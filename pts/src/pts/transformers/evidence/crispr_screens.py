"""CRISPR Brain screens (crispr_screen) evidence."""

from __future__ import annotations

import gzip
import json
import re
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.expressions import log10_rescaled_score
from pts.transformers.evidence.utils.run import run_evidence
from pts.transformers.utils.dataset import scan_dataset

LITERATURE_MAPPING = {
    'https://www.biorxiv.org/content/10.1101/2021.09.11.459904v1': ['ppr393667'],
    'https://pubmed.ncbi.nlm.nih.gov/30297964/': ['30297964'],
    'https://www.biorxiv.org/content/10.1101/2021.08.23.457400v1': ['PPR387167'],
    'https://doi.org/10.1101/2020.06.27.175679': ['34031600'],
    'https://pubmed.ncbi.nlm.nih.gov/31422865/': ['31422865'],
    'https://pubmed.ncbi.nlm.nih.gov/30449619/': ['30449619'],
}

#: Study file columns read by header name, with the type each is cast to.
STUDY_COLUMNS: dict[str, pl.DataType] = {
    'Gene': pl.String(),
    'Phenotype': pl.Float64(),
    'P Value': pl.Float32(),
    'Hit Class': pl.String(),
}

#: Screen fields read from the screens payload.
SCREEN_FIELDS = [
    'Screen Name', 'Libraries Screened', 'Cell Type', 'Genotype', 'Phenotype', 'Description', 'Reference Link'
]

CRISPR_SCREEN = EvidenceDatasource(
    datasource_id='crispr_screen',
    unique_fields=['targetId', 'targetFromSourceId', 'diseaseId', 'datasourceId', 'studyId'],
    score=log10_rescaled_score(pl.col('resourceScore'), 0.0, 0.5, 0.005),
)


def _paragraphs(markdown: str) -> list[str]:
    """Split markdown into paragraphs, a heading being one on its own, with wrapped lines joined."""
    blocks: list[list[str]] = [[]]
    for line in markdown.split('\n'):
        if not line.strip() or line.startswith('#'):
            blocks.append([])
        blocks[-1].append(line)
        if line.startswith('#'):
            blocks.append([])
    return [joined for block in blocks if (joined := ' '.join(' '.join(block).split()))]


def _section(paragraphs: list[str]) -> str | None:
    return '\n\n'.join(paragraphs) or None


def parsing_experiment(description: str | None) -> dict[str, str | None]:
    """Parse the title, experiment and analysis out of a screen description.

    A description is markdown: an intro, then `## Experiment` and `## Analysis` sections. The intro
    holds a one-sentence summary, either as its heading or as a paragraph after a screen name
    heading (sometimes followed by `## Description`). The title is that summary: the intro's first
    paragraph ending in a full stop, else its last, without `#`s. Experiment and analysis are their
    whole sections. Wrapped lines are joined, `*` emphasis removed, and a missing part is null.
    """
    text = (description or '').lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n').replace('*', '')
    parts = re.split(r'^#+\s*(Experiment|Analysis)\s*$', text, flags=re.MULTILINE)
    sections = dict(zip(parts[1::2], parts[2::2], strict=True))
    intro = [re.sub(r'^#+\s*', '', p) for p in _paragraphs(parts[0])]
    return {
        'title': next((p for p in intro if p.endswith('.')), intro[-1] if intro else None),
        'experiment': _section(_paragraphs(sections.get('Experiment', ''))),
        'analysis': _section(_paragraphs(sections.get('Analysis', ''))),
    }


def study_id(path: str) -> str:
    """The study id of a study file: its name without `.csv.gz`, with `%20` decoded to a space."""
    match = re.search(r'([^/]+)\.csv\.gz$', path)
    return (match.group(1) if match else '').replace('%20', ' ')


def _read_screens(path: str) -> pl.DataFrame:
    data, _ = StorageHandle(path).read()
    if path.endswith('.gz'):
        data = gzip.decompress(data)
    screens = [{**v, **v.get('metadata', {})} for v in json.loads(data).values() if isinstance(v, dict)]
    return pl.DataFrame(
        [{field: screen.get(field) for field in SCREEN_FIELDS} for screen in screens],
        schema=dict.fromkeys(SCREEN_FIELDS, pl.String),
    )


def _read_study(path: str) -> pl.LazyFrame | None:
    """The study's `STUDY_COLUMNS`, or None if it has no `Hit Class` (e.g. a CROP-seq screen)."""
    study = scan_dataset(path, format='csv', infer_schema=False)
    if 'Hit Class' not in study.collect_schema():
        logger.warning(f'skipping study file without a Hit Class column: {path}')
        return None
    return study.select(*STUDY_COLUMNS, studyId=pl.lit(study_id(path)))


def _read_genes(studies_dir: str) -> pl.LazyFrame:
    paths = sorted(StorageHandle(studies_dir).glob('*.csv.gz'))
    if not paths:
        raise ValueError(f'no *.csv.gz files found in {studies_dir}')
    studies = [study for path in paths if (study := _read_study(path)) is not None]
    if not studies:
        raise ValueError(f'no study file in {studies_dir} has a Hit Class column')
    return pl.concat(studies).with_columns(
        pl.col(name).str.strip_chars().cast(dtype, strict=False)
        for name, dtype in STUDY_COLUMNS.items()
        if dtype != pl.String
    )


def _studies(screens: pl.DataFrame, disease_mapping: pl.DataFrame) -> pl.DataFrame:
    literature = pl.DataFrame(
        {'Reference Link': list(LITERATURE_MAPPING), 'literature': list(LITERATURE_MAPPING.values())},
        schema={'Reference Link': pl.String, 'literature': pl.List(pl.String)},
    )
    return (
        screens
        .join(literature, on='Reference Link', how='left', maintain_order='left')
        .select(
            pl.col('Screen Name').alias('studyId'),
            pl.lit('affected_pathway').alias('datatypeId'),
            pl.lit('crispr_screen').alias('datasourceId'),
            pl.lit('crispr_brain').alias('projectId'),
            pl.col('Libraries Screened').alias('crisprScreenLibrary'),
            pl
            .col('Description')
            .map_elements(lambda d: parsing_experiment(d)['title'], return_dtype=pl.String)
            .alias('studyOverview'),
            pl.col('Cell Type').alias('cellType'),
            pl.when(pl.col('Genotype') != 'WT').then(pl.col('Genotype')).alias('geneticBackground'),
            'Phenotype',
            'literature',
        )
        .join(disease_mapping, on='studyId', how='left', maintain_order='left')
        .with_columns(pl.coalesce('contrast', 'Phenotype').alias('contrast'))
        .drop('Phenotype')
        .filter(pl.col('diseaseFromSourceMappedId').is_not_null())
        .unique(maintain_order=True)
    )


def generate(screens: str, studies_dir: str, disease_mapping: str) -> pl.LazyFrame:
    """Build CRISPR Brain evidence from its screens, per-study hits and disease mappings.

    Args:
        screens: location of the gzipped json screens payload.
        studies_dir: directory of per-study `.csv.gz` files, named after the study.
        disease_mapping: location of the study to disease mapping tsv.

    Returns:
        Raw CRISPR Brain evidence, one row per hit and mapped disease of a mapped study.
    """
    studies = _studies(
        _read_screens(screens),
        scan_dataset(disease_mapping, format='tsv', infer_schema=False).collect(),
    )
    hit_class = pl.col('Hit Class')
    genes = (
        _read_genes(studies_dir)
        .filter(hit_class.is_in(['Positive Hit', 'Negative Hit']))
        .select(
            pl.col('Gene').alias('targetFromSourceId'),
            pl.col('P Value').alias('resourceScore'),
            pl
            .when(hit_class == 'Positive Hit')
            .then(pl.lit('upper tail'))
            .when(hit_class == 'Negative Hit')
            .then(pl.lit('lower tail'))
            .alias('statisticalTestTail'),
            pl.col('Phenotype').alias('log2FoldChangeValue'),
            'studyId',
        )
        .unique(maintain_order=True)
    )
    return (
        studies
        .lazy()
        .join(genes, on='studyId', how='inner', maintain_order='left_right')
        .with_columns(pl.col('diseaseFromSourceMappedId').str.split(', '))
        .explode('diseaseFromSourceMappedId', keep_nulls=False)
    )


def crispr_screens(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process CRISPR Brain evidence.

    Args:
        source: `screens`, `studies_dir`, `disease_mapping`, `disease`, `target` and `literature`
            locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    run_evidence(
        generate(source['screens'], source['studies_dir'], source['disease_mapping']),
        CRISPR_SCREEN,
        disease=source['disease'],
        target=source['target'],
        literature=source['literature'],
        destination=destination,
    )
