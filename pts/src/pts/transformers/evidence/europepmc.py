"""Europe PMC literature evidence, from target-disease cooccurrences in publications."""

from __future__ import annotations

from typing import Any

import polars as pl
from otter.config.model import Config

from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.expressions import divide
from pts.transformers.evidence.utils.run import run_evidence
from pts.transformers.utils.dataset import scan_dataset

EXCLUDED_TARGET_TERMS = [
    'TEC',
    'TECS',
    'Tec',
    'tec',
    "'",
    '(',
    ')',
    '-',
    '-S',
    'S',
    'S-',
    'SS',
    'SSS',
    'Ss',
    'Ss-',
    's',
    's-',
    'ss',
    'U3',
    'U6',
    'u6',
    'SNORA70',
    'U2',
    'U8',
]
SECTIONS_OF_INTEREST = [
    'title',
    'abstract',
    'intro',
    'case',
    'figure',
    'table',
    'discuss',
    'concl',
    'results',
    'appendix',
    'other',
]

EUROPEPMC = EvidenceDatasource(
    datasource_id='europepmc',
    unique_fields=['targetId', 'targetFromSourceId', 'diseaseId', 'datasourceId', 'literature'],
    # min_horizontal skips a null or NaN resourceScore, so it scores 1.0
    score=pl.min_horizontal(divide(pl.col('resourceScore'), 100.0), pl.lit(1.0)),
)


def compute_evidence(cooccurrences: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate gene-disease cooccurrences into evidence, one row per publication, target, disease and year.

    Lists are sorted sets and the score is summed one by one in ascending order.

    Args:
        cooccurrences: cooccurrences in the literature `Cooccurrence` schema.

    Returns:
        Raw Europe PMC evidence.
    """
    pmid = pl.col('pmid').cast(pl.String).str.strip_chars(' ')
    return (
        cooccurrences
        .filter(pl.col('section').is_in(SECTIONS_OF_INTEREST))
        .with_columns(pmid.alias('pmid'))
        .with_columns(pl.coalesce('pmid', 'pmcid').alias('publicationIdentifier'))
        .filter(
            (pl.col('type') == 'GP-DS')
            & pl.col('isMapped')
            & pl.col('publicationIdentifier').is_not_null()
            & (pl.col('text').str.len_chars() < 600)
            & ~pl.col('label1').is_in(EXCLUDED_TARGET_TERMS)
        )
        .rename({'mappedId1': 'targetFromSourceId', 'mappedId2': 'diseaseFromSourceMappedId'})
        .group_by('publicationIdentifier', 'targetFromSourceId', 'diseaseFromSourceMappedId', 'year')
        .agg(
            pl.col('pmcid').drop_nulls().unique().sort().alias('pmcIds'),
            pl.col('pmid').drop_nulls().unique().sort().alias('literature'),
            pl
            .struct(
                'text',
                pl.col('start1').alias('tStart'),
                pl.col('end1').alias('tEnd'),
                pl.col('start2').alias('dStart'),
                pl.col('end2').alias('dEnd'),
                'section',
            )
            .unique()
            .sort()
            .alias('textMiningSentences'),
            pl.col('evidenceScore').sort().cum_sum().last().alias('resourceScore'),
        )
        .filter(pl.col('resourceScore') > 1)
        .select(
            pl.lit('europepmc').alias('datasourceId'),
            pl.lit('literature').alias('datatypeId'),
            'targetFromSourceId',
            'diseaseFromSourceMappedId',
            'resourceScore',
            'literature',
            'textMiningSentences',
            pl.when(pl.col('pmcIds').list.len() != 0).then(pl.col('pmcIds')).alias('pmcIds'),
            pl.col('year').alias('publicationYear'),
        )
    )


def europepmc(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process Europe PMC evidence.

    Args:
        source: `cooccurrence`, `disease`, `target` and `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    run_evidence(
        compute_evidence(scan_dataset(source['cooccurrence'])),
        EUROPEPMC,
        disease=source['disease'],
        target=source['target'],
        literature=source['literature'],
        destination=destination,
    )
