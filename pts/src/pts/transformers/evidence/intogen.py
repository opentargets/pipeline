"""intOGen cancer driver gene evidence."""

from __future__ import annotations

from typing import Any

import polars as pl
from otter.config.model import Config

from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.expressions import direction_when_disease_present, log10_rescaled_score
from pts.transformers.evidence.utils.luts import build_luts
from pts.transformers.evidence.utils.run import write_evidence
from pts.transformers.utils.dataset import scan_dataset

DOI_TO_PMID_MAPPING = {
    '10.1038/ng.2529': '23334666',
    '10.1038/s41588-023-01321-1': '36928603',
    '10.1038/ng.3940': '28825729',
    '10.1101/162784': '32025007',
    '10.1172/JCI128227': '31483290',
    '10.1038/s41591-019-0582-4': '31570822',
    '10.1002/hep.27198': '24798001',
    '10.1038/ng.3520': '26928227',
    '10.1038/s41588-018-0078-z': '29610475',
}
#: `ROLE` values and the functional consequence they map to.
ROLE_CONSEQUENCES = {'Act': 'SO_0002053', 'LoF': 'SO_0002054'}

INTOGEN = EvidenceDatasource(
    datasource_id='intogen',
    unique_fields=['targetId', 'targetFromSourceId', 'diseaseId', 'datasourceId', 'cohortShortName'],
    score=log10_rescaled_score(pl.col('resourceScore'), 0.25, 0.1, 1e-10),
)
DIRECTION_ON_TRAIT = direction_when_disease_present('risk')


def _direction_on_target() -> pl.Expr:
    """LoF or GoF when a matching consequence is among `mutatedSamples`, LoF first."""
    consequences = pl.col('mutatedSamples').list.eval(pl.element().struct.field('functionalConsequenceId'))
    return (
        pl
        .when(consequences.list.contains('SO_0002054'))
        .then(pl.lit('LoF'))
        .when(consequences.list.contains('SO_0002053'))
        .then(pl.lit('GoF'))
        .otherwise(None)
    )


DIRECTION_ON_TARGET = _direction_on_target()


def parse_source(source: pl.Expr) -> pl.Expr:
    """Turn a cohort `REFERENCE` into a one-element literature list, mapping known DOIs to PMIDs.

    The text after the first colon, stripped of spaces; null for `WEB` references and for
    references without a colon.
    """
    processed = (
        pl
        .when(source.str.starts_with('WEB'))
        .then(None)
        .otherwise(source.str.split(':').list.get(1, null_on_oob=True).str.strip_chars(' '))
    )
    mapped = processed.replace(DOI_TO_PMID_MAPPING)
    return pl.when(mapped.is_not_null()).then(pl.concat_list(mapped)).otherwise(None)


def _read_tsv(path: str) -> pl.DataFrame:
    return scan_dataset(path, format='tsv', infer_schema=False).collect()


def generate(genes: str, cohorts: str, disease_mapping: str) -> pl.LazyFrame:
    """Build intOGen evidence from its driver genes, cohorts and cancer type mappings.

    Args:
        genes: location of the intOGen driver genes tsv.
        cohorts: location of the intOGen cohorts tsv.
        disease_mapping: location of the cancer type to disease mapping tsv.

    Returns:
        Raw intOGen evidence, one row per driver gene, cohort and mapped disease.
    """
    samples = pl.col('SAMPLES').cast(pl.Float64)
    evidence = (
        _read_tsv(genes)
        .join(_read_tsv(cohorts).drop('SAMPLES', strict=False), on='COHORT', how='inner')
        .select(
            pl.lit('somatic_mutation').alias('datatypeId'),
            pl.lit('intogen').alias('datasourceId'),
            pl.col('SYMBOL').alias('targetFromSourceId'),
            pl.col('COHORT').alias('cohortId'),
            pl.col('COHORT_NICK').alias('cohortShortName'),
            pl.col('COHORT_NAME').alias('cohortDescription'),
            pl.col('METHODS').str.split(',').alias('significantDriverMethods'),
            pl.concat_list(
                pl.struct(
                    pl.col('ROLE').replace_strict(ROLE_CONSEQUENCES, default=None).alias('functionalConsequenceId'),
                    samples.alias('numberSamplesTested'),
                    samples.alias('numberMutatedSamples'),
                )
            ).alias('mutatedSamples'),
            pl.col('CANCER').alias('Cancer_type_acronym'),
            pl.col('CANCER_NAME').alias('diseaseFromSource'),
            pl.col('QVALUE_COMBINATION').cast(pl.Float64).alias('resourceScore'),
            parse_source(pl.col('REFERENCE')).alias('literature'),
        )
    )
    mapping = _read_tsv(disease_mapping).select(
        'Cancer_type_acronym',
        pl.col('EFO_id').str.strip_chars(' ').alias('diseaseFromSourceMappedId'),
    )
    return (
        evidence
        .join(mapping, on='Cancer_type_acronym', how='left', maintain_order='left')
        .drop('Cancer_type_acronym')
        .lazy()
    )


def intogen(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process intOGen evidence.

    Args:
        source: `genes`, `cohorts`, `disease_mapping`, `disease`, `target` and `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    raw = generate(source['genes'], source['cohorts'], source['disease_mapping'])
    luts = build_luts(raw, disease=source['disease'], target=source['target'], literature=source['literature'])
    evidence = (
        INTOGEN
        .postprocess(raw, luts)
        .assign_direction_on_trait(DIRECTION_ON_TRAIT)
        .assign_direction_on_target(DIRECTION_ON_TARGET, luts.target)
        .hash_long_variant_identifiers()
    )
    write_evidence(evidence, destination)
