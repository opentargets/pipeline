"""Command line of the AACT mechanism-of-action experiment.

Each subcommand is one stage. It reads the files earlier stages wrote to ``--run-dir`` and
the inputs its own flags name, and writes its results back to ``--run-dir``::

    texts -> pools -> link -> pilot -> extract -> ground -> judge -> evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
from loguru import logger

from pts.transformers.clinical_mira.moa import benchmark, grounding, judge, pilot, pooling, quotes, report
from pts.transformers.clinical_mira.moa.llm import run_cached
from pts.transformers.clinical_mira.moa.schemas import JudgeVerdict, MoaExtraction, load_prompt

STRATA = ('phase_group', 'maximumClinicalStage', 'drugType')


def _read(path: str | Path, columns: list[str] | None = None) -> pl.DataFrame:
    path = Path(path)
    source = str(path / '*.parquet') if path.is_dir() else str(path)
    return pl.read_parquet(source, columns=columns)


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def cmd_texts(run: Path, args: argparse.Namespace) -> None:
    extraction = _read(args.extraction, ['id', 'prompt_sha256'])
    texts = pooling.trial_texts(str(args.archive), str(args.scratch or run / 'scratch'))
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
        (pool_id, f'Drug: {name}\n\n{text}')
        for pool_id, name, text in pools.select('pool_id', 'drug_name', 'text').iter_rows()
    ]
    results, errors = run_cached(
        requests,
        system_prompt=load_prompt('extract'),
        response_model=MoaExtraction,
        model=args.model,
        cache_path=run / 'extract_cache.jsonl',
        concurrency=args.concurrency,
    )
    extractions = {k: v for k, v in results.items() if isinstance(v, MoaExtraction)}
    texts = pl.read_parquet(run / 'texts.parquet')
    mechanisms = quotes.check_mechanisms(extractions, pools, texts)
    mechanisms.write_parquet(run / 'mechanisms.parquet')
    quotes.pool_status(pools, extractions, mechanisms).write_parquet(run / 'statuses.parquet')
    _write_json(run / 'extract_errors.json', errors)
    logger.info(f'extract: {len(extractions)} pools, {mechanisms.height} mechanisms, {len(errors)} failed')


def cmd_ground(run: Path, args: argparse.Namespace) -> None:
    index = grounding.build_label_index(_read(args.target_view, ['id', *grounding.FIELDS]))
    grounded = grounding.ground_mechanisms(pl.read_parquet(run / 'mechanisms.parquet'), index)
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
    selected = judge.select_for_judging(scored, seed=args.seed, n_matches=args.n_matches)
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
            judge.judge_input(row, pool_text[row['pool_id']], chembl.get(drug_of.get(row['pool_id']), [])),
        )
        for row in rows.iter_rows(named=True)
    ]
    results, errors = run_cached(
        requests,
        system_prompt=load_prompt('judge'),
        response_model=JudgeVerdict,
        model=args.model,
        cache_path=run / 'judge_cache.jsonl',
        concurrency=args.concurrency,
    )
    verdicts = [
        {
            'pool_id': key.rsplit(':', 1)[0],
            'mechanism_idx': int(key.rsplit(':', 1)[1]),
            'quoteSupports': v.quote_supports,
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
            'quoteSupports': pl.String,
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
            quoteSupports=pl.lit(None, dtype=pl.String),
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
    judge.handcheck(adjusted, pl.read_parquet(run / 'pools.parquet'), seed=args.seed).write_csv(run / 'handcheck.csv')
    (run / 'report.md').write_text(report.render_report(run), encoding='utf-8')
    logger.info(f'evaluate: report at {run / "report.md"}')


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
    extract.add_argument('--model', default='gpt-5-mini')
    extract.add_argument('--concurrency', type=int, default=20)
    extract.add_argument('--limit', type=int, default=None)
    extract.add_argument('--all', action='store_true', help='every pool, not only the pilot')

    command('ground', cmd_ground).add_argument('--target-view', required=True)

    judge_cmd = command('judge', cmd_judge)
    judge_cmd.add_argument('--model', default='gpt-5')
    judge_cmd.add_argument('--concurrency', type=int, default=20)
    judge_cmd.add_argument('--n-matches', type=int, default=100)
    judge_cmd.add_argument('--seed', type=int, default=0)

    command('evaluate', cmd_evaluate).add_argument('--seed', type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run one stage."""
    args = build_parser().parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.handler(args.run_dir, args)


if __name__ == '__main__':
    main()
