"""Transformer that stages the mechanisms of action the literature states for each clinical drug.

Each clinical drug's sentences are taken from the literature match of a published release and a
fixed number are selected per drug. Each drug is a record in the store, keyed by its selected
sentences, so a drug whose selection did not change is not computed again. A changed selection
whose new sentences have no mechanism wording carries the drug's previous output forward. Every
drug's output, computed or not, is checked against its current sentences before it is published as
a write-once snapshot beneath the release it was read from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger
from pydantic import BaseModel, ConfigDict

from pts.transformers.clinical_mira.aact_moa import (
    SNAPSHOT_COLUMNS,
    SeedTooSmall,
    _exists,
    _guarded,
    _write_parquet,
)
from pts.transformers.clinical_mira.moa import literature, literature_staging, quotes, schemas
from pts.transformers.clinical_mira.record_store import map_texts, new_run_id, publish_snapshot, read_records, write_run
from pts.transformers.clinical_mira.staged_outputs import MOA_LITERATURE
from pts.transformers.utils.dataset import scan_dataset

if TYPE_CHECKING:
    from otter.config.model import Config

LITERATURE_SNAPSHOT_COLUMNS = ['drugId', 'docIds'] + [c for c in SNAPSHOT_COLUMNS if c not in {'drugName', 'nctIds'}]
"""Columns of the published literature mechanisms."""

MOLECULE_COLUMNS = ['id', 'parentId', 'maximumClinicalStage', 'name']
"""Columns read from the drug molecules."""


class MoaLiteratureSettings(BaseModel):
    """Settings for the moa_literature transformer."""

    model_config = ConfigDict(extra='forbid')

    store_uri: str
    """Root of the record stores, an absolute URI. Sits outside the release directory so a drug seen in an
        earlier release is not computed again."""
    input_release: str
    """The release the literature match and drug molecules are read from. Prefixes the id of the runs this
        step writes, names the batch state file and the snapshot directory."""
    workers: int = 8
    """Match files reduced concurrently by the sentence pass."""
    shard_size: int = 2000
    """Drugs per shard. Each shard is written to the store as it finishes."""
    max_attempts: int = 3
    """Attempts per drug, within a run and across runs, before it is left out."""
    import_experiment_cache: str | None = None
    """The experiment's seed jsonl to import into the store, once, before computing."""
    import_for_tag: str | None = None
    """The MOA_LITERATURE tag the seed was computed for. The seed is imported only while it is the current tag."""
    min_seeded_share: float = 0.69
    """The lowest share of the current drugs the imported seed must cover, or the run aborts before computing."""
    billing_project: str | None = None
    """The project billed for reading a requester-pays literature match, or None to read it unbilled."""
    workspace_id_env: str = 'ANTHROPIC_GOOGLE_CLOUD_WORKSPACE_ID'
    """Name of the environment variable holding the Claude workspace id."""


def _metadata(input_release: str) -> dict[str, Any]:
    """The snapshot metadata of a run over the literature of ``input_release``."""
    return {
        'kind': MOA_LITERATURE.name,
        'model': MOA_LITERATURE.model,
        'tag': MOA_LITERATURE.tag(),
        'model_revisions': {MOA_LITERATURE.model: MOA_LITERATURE.tag()},
        'input_release': input_release,
    }


def _check_quotes(extractions: dict[str, schemas.MoaExtraction], requests: pl.DataFrame) -> pl.DataFrame:
    """Check each drug's mechanisms against that drug's own sentences of the documents it was shown."""
    docs = dict(requests.select('parent', 'docs').iter_rows())
    frames = [pl.DataFrame(schema=quotes.MECHANISM_SCHEMA)]
    for parent, extraction in extractions.items():
        shown = docs[parent]
        pools = pl.DataFrame(
            {'pool_id': [parent], 'nct_ids': [[d['doc'] for d in shown]]},
            schema={'pool_id': pl.String, 'nct_ids': pl.List(pl.String)},
        )
        texts = pl.DataFrame(
            {'id': [d['doc'] for d in shown], 'prompt': [d['text'] for d in shown]},
            schema={'id': pl.String, 'prompt': pl.String},
        )
        frames.append(quotes.check_mechanisms({parent: extraction}, pools, texts))
    return pl.concat(frames)


def _mechanism_rows(mechanisms: pl.DataFrame) -> pl.DataFrame:
    """The kept mechanisms with their drug and the documents their evidence cites."""
    return (
        mechanisms
        .filter(pl.col('kept'))
        .sort('pool_id', 'mechanism_idx')
        .with_columns(
            drugId='pool_id',
            docIds=pl.col('evidence').list.eval(pl.element().struct.field('nctId')).list.unique().list.sort(),
        )
        .select(LITERATURE_SNAPSHOT_COLUMNS)
    )


def _drug_table(requests: pl.DataFrame, status: pl.DataFrame, carried: set[str]) -> pl.DataFrame:
    """One row per drug: its name, sentences, whether it was extracted, what it states and whether it was carried."""
    return (
        requests
        .select('parent', 'name', n_sentences=pl.col('sentences').list.len())
        .join(status.rename({'pool_id': 'parent'}), on='parent', maintain_order='left')
        .select(
            'parent',
            'name',
            'n_sentences',
            'extracted',
            status=pl
            .when(~pl.col('extracted'))
            .then(pl.lit('failed'))
            .when(pl.col('not_stated'))
            .then(pl.lit('not_stated'))
            .otherwise(pl.lit('stated')),
            carried=pl.col('parent').is_in(sorted(carried)),
        )
    )


def stage_moa_literature(
    match: str,
    drug_molecule: str,
    destination: str,
    settings: MoaLiteratureSettings,
    config: Config,
    client: Any = None,
) -> dict[str, Any]:
    """Stage the mechanisms of action the literature states for each clinical drug and publish them as a snapshot.

    Args:
        match: The directory of literature match parquet files of ``input_release``.
        drug_molecule: The drug molecules of ``input_release``.
        destination: The directory holding one directory per input release, which receives the snapshot
            ``{input_release}/moa_literature/{tag}``.
        settings: The validated settings.
        config: The otter config.
        client: The Anthropic client, ``AnthropicGoogleCloud()`` when ``None``.

    Returns:
        dict[str, Any]: The number of drugs with candidate sentences and of those sentences, the number of
        drug requests (``pools``), seeded, carried and computed records, failed drugs, mechanisms kept and
        dropped by the quote check, and whether a snapshot was published.

    Raises:
        SeedTooSmall: If the seed is imported for the current tag and its import run covers less than
            ``min_seeded_share`` of the drug requests.
        ComputeAborted: If there are drugs to compute, no client is given and the workspace id variable is unset.
    """
    system_prompt = schemas.literature_system_prompt()
    scratch = config.work_path / 'moa_literature' / settings.input_release
    storage_options = (
        {'requester_pays': True, 'project': settings.billing_project} if settings.billing_project else None
    )

    molecules = scan_dataset(drug_molecule).select(MOLECULE_COLUMNS).collect()
    parents = literature.clinical_parents(molecules)
    cd_dir, gp_dir = literature.sentence_pass(match, parents, settings.workers, scratch, storage_options)
    sentences = literature.join_genes(cd_dir, gp_dir)
    picked = literature.select_sentences(sentences)
    requests = literature.requests(picked, molecules.select(parent='id', name='name'))
    n_drugs = sentences['parent'].n_unique()
    logger.info(f'{sentences.height} sentences of {n_drugs} drugs, {picked.height} picked, {requests.height} requests')

    store_uri = f'{settings.store_uri.rstrip("/")}/{MOA_LITERATURE.name}'
    tag = MOA_LITERATURE.tag()
    run_id = new_run_id(settings.input_release)

    seeded = 0
    importing = settings.import_experiment_cache is not None and settings.import_for_tag == tag
    if settings.import_experiment_cache is not None and not importing:
        logger.info(
            f'not importing {settings.import_experiment_cache}: computed for {settings.import_for_tag}, not {tag}'
        )
    if importing:
        seeded = literature_staging.import_experiment_cache(
            settings.import_experiment_cache, requests, system_prompt, store_uri, config
        )
        logger.info(f'imported {seeded} seed records into {store_uri}')

    records = read_records(store_uri, config, pl.String())
    if importing:
        covered = (
            records
            .filter(pl.col('run_id') == literature_staging.import_run_id())
            .filter(pl.col('text').is_in(requests['content_key'].implode()))['text']
            .n_unique()
        )
        if covered < settings.min_seeded_share * requests.height:
            msg = (
                f'{literature_staging.import_run_id()} covers {covered} of {requests.height} drugs, '
                f'below the minimum share {settings.min_seeded_share}'
            )
            raise SeedTooSmall(msg)

    carried = 0
    if MOA_LITERATURE.params['carry_forward']:
        carried_records = literature_staging.carry_forward(requests, records, tag, f'{run_id}-carry')
        if carried_records.height:
            write_run(carried_records, store_uri, f'{run_id}-carry', config, pl.String())
        carried = carried_records.height

    compute = literature_staging.literature_compute(
        requests,
        system_prompt,
        f'{store_uri}/batches/{settings.input_release}-{tag}.json',
        scratch,
        client=client,
    )
    mapped = map_texts(
        requests['content_key'],
        compute if client is not None else _guarded(compute, settings.workspace_id_env),
        store_uri,
        MOA_LITERATURE.name,
        tag,
        run_id,
        config,
        max_attempts=settings.max_attempts,
        output_dtype=pl.String(),
        allow_failures=True,
        shard_size=settings.shard_size,
    )

    outputs = {
        key: literature_staging.parse_output(output)
        for key, output in mapped.frame.select('text', 'output').iter_rows()
    }
    extractions = {}
    carried_drugs = set()
    for parent, key in requests.select('parent', 'content_key').iter_rows():
        if key in outputs:
            extractions[parent] = schemas.MoaExtraction.model_validate(outputs[key]['extraction'])
            if outputs[key]['carried_from']:
                carried_drugs.add(parent)

    mechanisms = _check_quotes(extractions, requests)
    status = quotes.pool_status(requests.select(pool_id='parent'), extractions, mechanisms)
    rows = _mechanism_rows(mechanisms)

    uri = f'{destination.rstrip("/")}/{settings.input_release}/moa_literature/{tag}'
    published = publish_snapshot(rows, uri, _metadata(settings.input_release), config)
    if published or not _exists(f'{uri}/drugs.parquet', config):
        _write_parquet(_drug_table(requests, status, carried_drugs), f'{uri}/drugs.parquet', config)

    return {
        'drugs': n_drugs,
        'sentences': sentences.height,
        'pools': requests.height,
        'seeded': seeded,
        'carried': carried,
        'computed': mapped.computed,
        'failed': mapped.failed,
        'mechanisms': rows.height,
        'dropped_by_quote_check': mechanisms.filter(~pl.col('kept')).height,
        'published': published,
    }


def moa_literature(
    source: dict[str, str], destination: dict[str, str], settings: dict[str, Any], config: Config
) -> None:
    """Stage the mechanisms of action stated in the literature of a published release, reusing earlier outputs.

    Args:
        source: Keys ``match``, the literature match directory, and ``drug_molecule``, the drug molecules,
            both of the previous release.
        destination: Key ``moa``, the directory the snapshot is published beneath.
        settings: See :py:class:`MoaLiteratureSettings`.
        config: The otter config.
    """
    s = MoaLiteratureSettings.model_validate(settings)
    stats = stage_moa_literature(source['match'], source['drug_molecule'], destination['moa'], s, config)
    logger.info(f'literature mechanisms staged: {stats}')
