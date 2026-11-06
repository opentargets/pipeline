"""Command line of the AACT mechanism-of-action experiment.

Each subcommand is one stage. It reads the files earlier stages wrote to ``--run-dir`` and
the inputs its own flags name, and writes its results back to ``--run-dir``::

    texts -> pools -> link -> pilot -> extract -> ground -> judge -> evaluate

``qc`` is separate: it checks a finished release. ``diff-snapshots`` compares two staged snapshots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from loguru import logger

from pts.transformers.clinical_mira.moa import (
    benchmark,
    grounding,
    judge,
    pilot,
    pooling,
    qc,
    quotes,
    report,
    staging,
    vertex,
)
from pts.transformers.clinical_mira.moa.claude import run_batch_cached
from pts.transformers.clinical_mira.moa.llm import run_cached
from pts.transformers.clinical_mira.moa.schemas import JudgeVerdict, MoaExtraction, load_prompt

if TYPE_CHECKING:
    from pydantic import BaseModel

STRATA = ('phase_group', 'maximumClinicalStage', 'drugType')
DEFAULT_MODELS = {
    ('openai', 'extract'): 'gpt-5-mini',
    ('openai', 'judge'): 'gpt-5',
    ('vertex', 'extract'): vertex.GPT_OSS_120B,
    ('vertex', 'judge'): vertex.GPT_OSS_120B,
    ('anthropic', 'extract'): 'claude-opus-5-5',
    ('anthropic', 'judge'): 'claude-opus-5-5',
}


def _provider_options(args: argparse.Namespace, stage: str) -> dict:
    """The ``run_cached`` options for the chosen provider."""
    model = args.model or DEFAULT_MODELS[(args.provider, stage)]
    if args.provider == 'vertex':
        project, location = args.vertex_project, args.vertex_location
        return {
            'model': model,
            'api': 'chat',
            'client_factory': lambda: vertex.vertex_client(project, location),
        }
    return {'model': model}


def _run_llm(
    args: argparse.Namespace,
    run: Path,
    stage: Literal['extract', 'judge'],
    requests: list[tuple[str, str]],
    response_model: type[BaseModel],
    cached_only: bool = False,
) -> tuple[dict, dict[str, str]]:
    """Run one stage's requests with the chosen provider, through the stage's cache."""
    if args.provider == 'anthropic':
        if cached_only:
            return run_cached(
                requests,
                system_prompt=load_prompt(stage),
                response_model=response_model,
                model=args.model or DEFAULT_MODELS[('anthropic', stage)],
                cache_path=run / f'{stage}_cache.jsonl',
                cached_only=True,
            )
        return run_batch_cached(
            requests,
            system_prompt=load_prompt(stage),
            response_model=response_model,
            model=args.model or DEFAULT_MODELS[('anthropic', stage)],
            cache_path=run / f'{stage}_cache.jsonl',
            state_uri=str(run / f'{stage}_batches.json'),
            effort=args.effort,
        )
    return run_cached(
        requests,
        system_prompt=load_prompt(stage),
        response_model=response_model,
        cache_path=run / f'{stage}_cache.jsonl',
        concurrency=args.concurrency,
        cached_only=cached_only,
        **_provider_options(args, stage),
    )


LABELS = ('relation', 'statementType', 'effect_wording', 'action_other')


def _read(path: str | Path, columns: list[str] | None = None) -> pl.DataFrame:
    path = Path(path)
    source = str(path / '*.parquet') if path.is_dir() else str(path)
    return pl.read_parquet(source, columns=columns)


def _columns(path: str | Path) -> set[str]:
    path = Path(path)
    source = str(path / '*.parquet') if path.is_dir() else str(path)
    return set(pl.scan_parquet(source).collect_schema().names())


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def cmd_texts(run: Path, args: argparse.Namespace) -> None:
    extraction = _read(args.extraction, ['id', 'prompt_sha256'])
    scratch = Path(args.scratch or run / 'scratch')
    scratch.mkdir(parents=True, exist_ok=True)
    texts = pooling.trial_texts(str(args.archive), str(scratch))
    matched, counts = pooling.match_texts(texts, extraction)
    matched.write_parquet(run / 'texts.parquet')
    _write_json(run / 'text_counts.json', {'trials_in_extraction': extraction.height, **counts})
    logger.info(f'texts: {counts}')


def cmd_pools(run: Path, args: argparse.Namespace) -> None:
    extraction = _read(args.extraction, ['id', 'investigated_drugs'])
    texts = pl.read_parquet(run / 'texts.parquet')
    trials = pooling.single_drug_trials(extraction)
    pools = pooling.build_pools(trials, texts, char_budget=args.char_budget)
    pools.write_parquet(run / 'pools.parquet')
    counts = {
        'single_drug_trials': trials.height,
        'single_drug_trials_with_text': trials.filter(pl.col('nct_id').is_in(texts['id'].to_list())).height,
        'pools': pools.height,
    }
    _write_json(run / 'pool_counts.json', counts)
    logger.info(f'pools: {counts}')


def cmd_link(run: Path, args: argparse.Namespace) -> None:
    pools = pl.read_parquet(run / 'pools.parquet')
    report_rows = _read(args.clinical_report, ['id', 'source', 'drugs', 'clinicalStage'])
    molecule = _read(args.molecule, ['id', 'maximumClinicalStage', 'drugType'])
    moa = _read(args.moa, ['mechanismOfAction', 'actionType', 'chemblIds', 'targets'])
    links = benchmark.link_pools(pools, report_rows, molecule)
    links.write_parquet(run / 'links.parquet')
    benchmark.benchmark_targets(moa).write_parquet(run / 'bench.parquet')
    moa.explode('chemblIds').rename({'chemblIds': 'drugId'}).write_parquet(run / 'moa.parquet')
    logger.info(f'links: {dict(links.group_by("link").len().iter_rows())}')


def cmd_pilot(run: Path, args: argparse.Namespace) -> None:
    sample = pilot.pilot_sample(
        pl.read_parquet(run / 'links.parquet'), pl.read_parquet(run / 'bench.parquet'), seed=args.seed
    )
    sample.write_parquet(run / 'pilot.parquet')
    logger.info(f'pilot: {sample.height} pools')


def cmd_extract(run: Path, args: argparse.Namespace) -> None:
    pools = pl.read_parquet(run / 'pools.parquet')
    if not args.all:
        pools = pools.join(pl.read_parquet(run / 'pilot.parquet'), on='pool_id', how='semi')
    pools = pools.sort('pool_id')
    if args.limit is not None:
        pools = pools.head(args.limit)
    requests = [
        (pool_id, staging.request_text(name, text))
        for pool_id, name, text in pools.select('pool_id', 'drug_name', 'text').iter_rows()
    ]
    results, errors = _run_llm(args, run, 'extract', requests, MoaExtraction, cached_only=args.cached_only)
    extractions = {k: v for k, v in results.items() if isinstance(v, MoaExtraction)}
    texts = pl.read_parquet(run / 'texts.parquet')
    mechanisms = quotes.check_mechanisms(extractions, pools, texts)
    mechanisms.write_parquet(run / 'mechanisms.parquet')
    quotes.pool_status(pools, extractions, mechanisms).write_parquet(run / 'statuses.parquet')
    _write_json(run / 'extract_errors.json', errors)
    logger.info(f'extract: {len(extractions)} pools, {mechanisms.height} mechanisms, {len(errors)} failed')


def cmd_ground(run: Path, args: argparse.Namespace) -> None:
    available = _columns(args.target_view)
    located = [f for f in grounding.LOCATION_FIELDS if f in available]
    index = grounding.build_label_index(_read(args.target_view, ['id', *grounding.FIELDS, *located]))
    grounded = grounding.ground_mechanisms(
        pl.read_parquet(run / 'mechanisms.parquet'),
        index,
        suggestion_min_confidence=args.suggestion_min_confidence,
    )
    grounded.write_parquet(run / 'grounded.parquet')
    logger.info(f'ground: {dict(grounded.group_by("grounding").len().iter_rows())}')


def _scored(run: Path) -> pl.DataFrame:
    return benchmark.score(
        pl.read_parquet(run / 'grounded.parquet'),
        pl.read_parquet(run / 'links.parquet'),
        pl.read_parquet(run / 'bench.parquet'),
    )


def cmd_judge(run: Path, args: argparse.Namespace) -> None:
    scored = _scored(run)
    strata = pl.read_parquet(run / 'links.parquet').select('pool_id', 'maximumClinicalStage')
    selected = judge.select_for_judging(scored, seed=args.seed, n_matches=args.n_matches, strata=strata)
    rows = scored.join(selected, on=['pool_id', 'mechanism_idx'], how='semi')
    pool_text = dict(pl.read_parquet(run / 'pools.parquet').select('pool_id', 'text').iter_rows())
    drug_of = dict(pl.read_parquet(run / 'links.parquet').select('pool_id', 'drugId').iter_rows())
    moa = pl.read_parquet(run / 'moa.parquet')
    chembl = {
        drug: group.select('mechanismOfAction', 'actionType').to_dicts() for (drug,), group in moa.group_by('drugId')
    }
    requests = [
        (
            judge.request_id(row['pool_id'], row['mechanism_idx']),
            judge.judge_input(
                row,
                pool_text[row['pool_id']],
                chembl.get(drug_of.get(row['pool_id']), []) if args.show_chembl else None,
            ),
        )
        for row in rows.iter_rows(named=True)
    ]
    results, errors = _run_llm(args, run, 'judge', requests, JudgeVerdict)
    verdicts = [
        {
            'pool_id': key.rsplit(':', 1)[0],
            'mechanism_idx': int(key.rsplit(':', 1)[1]),
            'targetSupported': v.target_supported,
            'actionSupported': v.action_supported,
            'primaryMechanism': v.primary_mechanism,
            'rationale': v.rationale,
        }
        for key, v in results.items()
        if isinstance(v, JudgeVerdict)
    ]
    pl.DataFrame(
        verdicts,
        schema={
            'pool_id': pl.String,
            'mechanism_idx': pl.UInt32,
            'targetSupported': pl.String,
            'actionSupported': pl.String,
            'primaryMechanism': pl.String,
            'rationale': pl.String,
        },
    ).write_parquet(run / 'verdicts.parquet')
    _write_json(run / 'judge_errors.json', errors)
    logger.info(f'judge: {len(verdicts)} verdicts, {len(errors)} failed')


def cmd_evaluate(run: Path, args: argparse.Namespace) -> None:
    scored = _scored(run)
    verdicts_path = run / 'verdicts.parquet'
    if verdicts_path.exists():
        adjusted = judge.adjust(scored, pl.read_parquet(verdicts_path))
    else:
        adjusted = scored.with_columns(
            targetSupported=pl.lit(None, dtype=pl.String),
            actionSupported=pl.lit(None, dtype=pl.String),
            primaryMechanism=pl.lit(None, dtype=pl.String),
            rationale=pl.lit(None, dtype=pl.String),
            judge_correct=pl.lit(None, dtype=pl.Boolean),
            adjusted_correct=pl.col('correct'),
        )
    adjusted.write_parquet(run / 'scored.parquet')

    links = pl.read_parquet(run / 'links.parquet')
    statuses = pl.read_parquet(run / 'statuses.parquet')
    bench = pl.read_parquet(run / 'bench.parquet')
    for by in STRATA:
        benchmark.summarise(adjusted, links, statuses, bench, by=[by]).write_parquet(run / f'summary_{by}.parquet')
        benchmark.summarise(adjusted, links, statuses, bench, by=[by], adjusted=True).write_parquet(
            run / f'summary_{by}_adjusted.parquet'
        )
    judge.calibration(adjusted).write_parquet(run / 'calibration.parquet')
    benchmark.precision_by_grounding(adjusted).write_parquet(run / 'precision_by_grounding.parquet')
    labelled = adjusted.with_columns(action_other=pl.col('actionType') == 'OTHER')
    for label in LABELS:
        benchmark.precision_by(labelled, label).write_parquet(run / f'precision_by_label_{label}.parquet')
    benchmark.precision_by_threshold(adjusted, links).write_parquet(run / 'precision_by_threshold.parquet')
    judge.handcheck(adjusted, pl.read_parquet(run / 'pools.parquet'), links, seed=args.seed).write_csv(
        run / 'handcheck.csv'
    )
    _write_json(run / 'usage.json', {stage: _usage(run / f'{stage}_cache.jsonl') for stage in ('extract', 'judge')})
    (run / 'report.md').write_text(report.render_report(run), encoding='utf-8')
    logger.info(f'evaluate: report at {run / "report.md"}')


def cmd_qc(_run: Path, args: argparse.Namespace) -> None:
    go = qc.run_qc(args.release, args.without, args.out, args.min_confidence, args.exclude_stages, args.min_scorable)
    logger.info(f'qc: {go}')


def _snapshot(path: str) -> pl.DataFrame:
    """Read the ``data.parquet`` of a snapshot directory or URI, whatever its tag."""
    return pl.read_parquet(f'{path.rstrip("/")}/data.parquet')


def cmd_diff_snapshots(_run: Path, args: argparse.Namespace) -> None:
    diff = qc.diff_snapshots(_snapshot(args.old), _snapshot(args.new))
    args.out.mkdir(parents=True, exist_ok=True)
    diff.with_columns(
        gained=pl.col('gained').list.join(';'),
        lost=pl.col('lost').list.join(';'),
    ).write_csv(args.out / 'snapshot_diff.csv')
    only_gained = diff.filter((pl.col('gained').list.len() > 0) & (pl.col('lost').list.len() == 0)).height
    only_lost = diff.filter((pl.col('lost').list.len() > 0) & (pl.col('gained').list.len() == 0)).height
    changed = diff.height - only_gained - only_lost
    logger.info(f'diff-snapshots: drugs gained: {only_gained}, lost: {only_lost}, changed: {changed}')


def _usage(cache_path: Path) -> dict[str, int]:
    """Sum the token counts recorded in a cache (Claude batches record them; other providers do not)."""
    totals = {'requests': 0, 'with_usage': 0}
    if not cache_path.exists():
        return totals
    for line in cache_path.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        totals['requests'] += 1
        if usage := row.get('usage'):
            totals['with_usage'] += 1
            for field, value in usage.items():
                totals[field] = totals.get(field, 0) + int(value)
    return totals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m pts.transformers.clinical_mira.moa', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    def command(name: str, handler: object) -> argparse.ArgumentParser:
        sub = commands.add_parser(name)
        sub.add_argument('--run-dir', required=True, type=Path)
        sub.set_defaults(handler=handler)
        return sub

    texts = command('texts', cmd_texts)
    texts.add_argument('--archive', required=True)
    texts.add_argument('--extraction', required=True)
    texts.add_argument('--scratch', default=None, help='where AACT is restored; keep the path short on macOS')

    pools = command('pools', cmd_pools)
    pools.add_argument('--extraction', required=True)
    pools.add_argument('--char-budget', type=int, default=60_000)

    link = command('link', cmd_link)
    link.add_argument('--clinical-report', required=True)
    link.add_argument('--molecule', required=True)
    link.add_argument('--moa', required=True)

    command('pilot', cmd_pilot).add_argument('--seed', type=int, default=0)

    extract = command('extract', cmd_extract)
    extract.add_argument('--concurrency', type=int, default=20)
    extract.add_argument('--limit', type=int, default=None)
    extract.add_argument('--all', action='store_true', help='every pool, not only the pilot')
    extract.add_argument('--cached-only', action='store_true', help='use cached results only; call no model')

    ground = command('ground', cmd_ground)
    ground.add_argument('--target-view', required=True)
    ground.add_argument(
        '--suggestion-min-confidence',
        type=float,
        default=None,
        help="ground mechanisms at or above this confidence to the model's suggestion even if the name disagrees",
    )

    judge_cmd = command('judge', cmd_judge)
    judge_cmd.add_argument('--concurrency', type=int, default=20)
    judge_cmd.add_argument('--n-matches', type=int, default=300)
    judge_cmd.add_argument('--seed', type=int, default=0)
    judge_cmd.add_argument('--show-chembl', action='store_true', help='show the judge ChEMBL mechanisms')

    for sub_parser in (extract, judge_cmd):
        sub_parser.add_argument('--provider', choices=['openai', 'vertex', 'anthropic'], default='openai')
        sub_parser.add_argument('--effort', default='medium', help='Claude effort (anthropic provider only)')
        sub_parser.add_argument('--model', default=None, help='default depends on the provider and the stage')
        sub_parser.add_argument('--vertex-project', default=vertex.DEFAULT_PROJECT)
        sub_parser.add_argument('--vertex-location', default=vertex.DEFAULT_LOCATION)

    qc_cmd = commands.add_parser('qc')
    qc_cmd.add_argument('--release', required=True, type=Path, help='release output root with the aact source')
    qc_cmd.add_argument('--without', required=True, type=Path, help='release output root without the aact source')
    qc_cmd.add_argument('--out', required=True, type=Path)
    qc_cmd.add_argument('--min-confidence', type=float, default=0.80)
    qc_cmd.add_argument('--exclude-stages', nargs='*', default=['APPROVAL'])
    qc_cmd.add_argument('--min-scorable', type=int, default=30, help='early-phase scorable candidates for a verdict')
    qc_cmd.set_defaults(handler=cmd_qc, run_dir=None)

    diff_cmd = commands.add_parser('diff-snapshots')
    diff_cmd.add_argument('--old', required=True, help='snapshot directory or URI holding data.parquet')
    diff_cmd.add_argument('--new', required=True, help='snapshot directory or URI holding data.parquet')
    diff_cmd.add_argument('--out', required=True, type=Path)
    diff_cmd.set_defaults(handler=cmd_diff_snapshots, run_dir=None)

    command('evaluate', cmd_evaluate).add_argument('--seed', type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run one stage."""
    args = build_parser().parse_args(argv)
    if args.run_dir is not None:
        args.run_dir.mkdir(parents=True, exist_ok=True)
    args.handler(args.run_dir, args)


if __name__ == '__main__':
    main()
