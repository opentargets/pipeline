"""Gene2Phenotype disease panel evidence.

Parses the Gene2Phenotype panels into evidence and post-processes it.
Diseases are mapped with the OnToma lookup tables through `add_efo_mapping`, by label and by
MONDO or OMIM id.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.schemas.gene2phenotype import G2P_INPUT_SCHEMA
from pts.schemas.outputs.evidence_gene2phenotype import EvidenceGene2phenotype
from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.expressions import confidence_score, direction_when_disease_present
from pts.transformers.evidence.utils.luts import build_luts
from pts.transformers.evidence.utils.ontology import add_efo_mapping
from pts.transformers.evidence.utils.run import write_evidence
from pts.transformers.utils.spark_csv import read_spark_csv
from pts.transformers.utils.spark_strings import spark_trim

#: The `source` keys of the panel files, in the order they are read.
PANELS = [
    'cancer_panel',
    'cardiac_panel',
    'developmental_panel',
    'ear_panel',
    'eye_panel',
    'skeletal_panel',
    'skin_panel',
]

#: Variant consequence terms by rank, lowest first; a row keeps its highest-ranked term.
CONSEQUENCE_RANKS = {
    'uncertain': 'SO_0002220',
    'cis-regulatory or promotor mutation': 'SO_0001566',
    '5_prime or 3_prime UTR mutation': 'SO_0001622',
    'increased gene product level': 'SO_0002315',
    'decreased gene product level': 'SO_0002316',
    'altered gene product structure': 'SO_0002318',
    'absent gene product': 'SO_0002317',
}

#: What python's `str.strip()` removes; none is above U+3000.
PYTHON_WHITESPACE = ''.join(chr(c) for c in range(0x3001) if chr(c).isspace())

CONFIDENCE_SCORES: dict[str, float] = {
    'definitive': 1.0,
    'both RD and IF': 1.0,
    'strong': 1.0,
    'moderate': 0.5,
    'limited': 0.01,
}

GENE2PHENOTYPE = EvidenceDatasource(
    datasource_id='gene2phenotype',
    # order feeds the id, as in the spark config
    unique_fields=[
        'targetId',
        'targetFromSourceId',
        'diseaseId',
        'datasourceId',
        'diseaseFromSource',
        'allelicRequirements',
        'studyId',
    ],
    score=confidence_score(CONFIDENCE_SCORES),
    schema=EvidenceGene2phenotype,
)
DIRECTION_ON_TRAIT = direction_when_disease_present('risk')
DIRECTION_ON_TARGET = (
    pl
    .when(pl.col('variantFunctionalConsequenceId') == 'SO_0002315')
    .then(pl.lit('GoF'))
    .when(pl.col('variantFunctionalConsequenceId') == 'SO_0002317')
    .then(pl.lit('LoF'))
    .otherwise(None)
)

_RANK_OF = {term: rank for rank, term in enumerate(CONSEQUENCE_RANKS)}
_CONSEQUENCE_OF_RANK = dict(enumerate(CONSEQUENCE_RANKS.values()))
#: Java's `.` excludes line terminators.
_RELATED_PREFIX = r'[^\n\r\x{0085}\x{2028}\x{2029}]+-related '


def _functional_consequence(column: str) -> pl.Expr:
    """The highest-ranked mapped term of a semicolon-separated list, null if none maps."""
    ranks = (
        pl
        .col(column)
        .str.split(';')
        .list.eval(pl.element().str.strip_chars(PYTHON_WHITESPACE).replace_strict(_RANK_OF, default=None))
    )
    return ranks.list.max().replace_strict(_CONSEQUENCE_OF_RANK, default=None, return_dtype=pl.String)


def _read_panel(path: str) -> pl.DataFrame:
    return read_spark_csv(path, new_columns=list(G2P_INPUT_SCHEMA)).cast(G2P_INPUT_SCHEMA, strict=False)


def parse_gene2phenotype(paths: list[str]) -> pl.LazyFrame:
    """Raw Gene2Phenotype evidence from the panel files, before disease mapping.

    The panels are read by column position, as the files' own header names are not stable.
    """
    panels = pl.concat([_read_panel(path) for path in paths])
    return panels.lazy().select(
        literature=pl
        .when(pl.col('publications').is_not_null())
        .then(pl.col('publications').str.split(';').list.eval(spark_trim(pl.element()))),
        targetFromSourceId=pl.col('gene symbol'),
        studyId=pl.col('panel'),
        confidence=pl.col('confidence'),
        allelicRequirements=pl
        .when(pl.col('allelic requirement').is_not_null())
        .then(pl.concat_list('allelic requirement')),
        diseaseFromSourceId=pl
        .when(pl.col('disease MONDO').is_not_null())
        .then(pl.col('disease MONDO'))
        .when(pl.col('disease mim').is_not_null() & ~pl.col('disease mim').str.contains('No disease mim', literal=True))
        .then(pl.concat_str(pl.lit('OMIM:'), pl.col('disease mim')))
        .otherwise(None),
        diseaseFromSource=pl.col('disease name').str.replace_all(_RELATED_PREFIX, ''),
        variantFunctionalConsequenceId=_functional_consequence('variant consequence'),
        datasourceId=pl.lit('gene2phenotype'),
        datatypeId=pl.lit('genetic_literature'),
    )


def gene2phenotype(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process Gene2Phenotype evidence.

    Args:
        source: the `PANELS` files, `ontoma_disease_label_lut`, `ontoma_disease_id_lut`, `disease`,
            `target` and `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    logger.info(f'load gene2phenotype panels from {[source[p] for p in PANELS]}')
    raw = add_efo_mapping(
        parse_gene2phenotype([source[p] for p in PANELS]),
        disease_label_lut_path=source['ontoma_disease_label_lut'],
        disease_id_lut_path=source['ontoma_disease_id_lut'],
    ).collect()
    luts = build_luts(raw.lazy(), disease=source['disease'], target=source['target'], literature=source['literature'])
    evidence = (
        GENE2PHENOTYPE
        .postprocess(raw.lazy(), luts)
        .assign_direction_on_trait(DIRECTION_ON_TRAIT)
        .assign_direction_on_target(DIRECTION_ON_TARGET, luts.target)
        .hash_long_variant_identifiers()
    )
    write_evidence(evidence, destination, schema=GENE2PHENOTYPE.schema)
