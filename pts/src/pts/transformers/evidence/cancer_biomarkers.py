"""Cancer Biomarkers evidence."""

from __future__ import annotations

import re
from typing import Any

import polars as pl
from otter.config.model import Config

from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.run import run_evidence
from pts.transformers.utils.dataset import scan_dataset

ALTERATIONTYPE2FUNCTIONCSQ: dict[str, str | None] = {
    # TODO: Map BIA
    'MUT': 'SO_0001777',  # somatic_variant
    'CNA': 'SO_0001563',  # copy_number_change
    'FUS': 'SO_0001882',  # feature_fusion
    'EXPR': None,
    'BIA': None,
}

DRUGRESPONSE2EFO: dict[str, str] = {
    'Responsive': 'GO_0042493',  # response to drug
    'Not Responsive': 'EFO_0020002',  # lack of efficacy
    'Resistant': 'EFO_0020001',  # drug resistance
    'Increased Toxicity': 'EFO_0020003',  # drug toxicity
    'Increased Toxicity (Myelosupression)': 'EFO_0007053',  # myelosuppression
    'Increased Toxicity (Ototoxicity)': 'EFO_0006951',  # ototoxicity
    'Increased Toxicity (Hyperbilirubinemia)': 'HP_0002904',  # Hyperbilirubinemia
    'Increased Toxicity (Haemolytic Anemia)': 'EFO_0005558',  # hemolytic anemia
}

GENENAMESOVERRIDE: dict[str, str] = {
    # Correct gene names to use their approved symbol
    'C15orf55': 'NUTM1',
    'MLL': 'KMT2A',
    'MLL2': 'KMT2D',
}

#: Fields identifying one evidence; every other field is collected over them.
EVIDENCE_KEYS = [
    'datasourceId',
    'datatypeId',
    'drugFromSource',
    'drugId',
    'drugResponse',
    'targetFromSourceId',
    'diseaseFromSource',
    'diseaseFromSourceMappedId',
    'confidence',
    'biomarkerName',
]

CANCER_BIOMARKERS = EvidenceDatasource(
    datasource_id='cancer_biomarkers',
    unique_fields=[
        'targetId',
        'targetFromSourceId',
        'diseaseId',
        'datasourceId',
        'biomarkerName',
        'confidence',
        'diseaseFromSource',
        'drugFromSource',
        'drugResponse',
    ],
    score=pl.lit(1.0),
)


def get_variant_id(gdna: str) -> str | None:
    """Converts the genomic coordinates to the CHROM_POS_REF_ALT notation.

    Ex.: 'chr14:g.105243048G_T' --> '14_105243048_G_T'
    """
    translate_dct = {'chr': '', ':g.': '_', '>': '_', 'del': '_', 'ins': '_'}
    try:
        for k, v in translate_dct.items():
            gdna = gdna.replace(k, v)
        _, head, tail = re.split(r'^(.*?_\d+)', gdna)
        if bool(re.search(r'\d+', tail)):
            tail = re.split(r'^(_\d+_)', tail)[-1]
        return head + '_' + tail
    except AttributeError:
        return None


def initcap(value: pl.Expr) -> pl.Expr:
    """Spark's `initcap`: lower-case, then upper-case the first character and every one after a space."""
    return (
        value.str
        .to_lowercase()
        .str.split(' ')
        .list.eval(pl.element().str.slice(0, 1).str.to_uppercase() + pl.element().str.slice(1))
        .list.join(' ')
    )


def regexp_extract(value: pl.Expr, pattern: str, group: int) -> pl.Expr:
    """Spark's `regexp_extract`: the group's match, empty if there is none, null for a null value."""
    return pl.when(value.is_not_null()).then(value.str.extract(pattern, group).fill_null(''))


def explode_alterations(frame: pl.DataFrame) -> pl.DataFrame:
    """One row per element of `alterations`, as `alteration`, paired with an `alteration_type`.

    Types are taken in turn from `alteration_types`, cycling when there are fewer types than
    alterations. Rows whose alterations or types are null or empty give no rows.
    """
    n_types = pl.col('alteration_types').list.len()
    return (
        frame
        .filter(n_types > 0)
        .with_columns(position=pl.int_ranges(0, pl.col('alterations').list.len()))
        .explode('alterations', 'position', keep_nulls=False)
        .with_columns(alteration_type=pl.col('alteration_types').list.get(pl.col('position') % n_types))
        .rename({'alterations': 'alteration'})
        .drop('position')
    )


def _split(column: str, separator: str) -> pl.Expr:
    return pl.col(column).str.split(separator).list.unique(maintain_order=True)


def _clean_alteration() -> pl.Expr:
    alteration = pl.col('alteration')
    return (
        pl
        .when(alteration == 'NRAS:.12.,.13.,.59.,.61.,.117.,.146.')
        .then(pl.col('Biomarker'))
        .when(alteration.str.contains(':.', literal=True))
        .then(alteration.str.replace_all('[:.]', ''))
        .when(alteration.str.contains('__', literal=True) & ~pl.col('Biomarker').str.contains('+', literal=True))
        .then(pl.col('Biomarker'))
        .otherwise(alteration)
    )


def _gene_expression_id() -> pl.Expr:
    expr = pl.col('alteration_type') == 'EXPR'
    alteration = pl.col('alteration')
    return (
        pl
        .when(expr & alteration.str.contains('over', literal=True))
        .then(pl.lit('GO_0010628'))
        .when(expr & alteration.str.contains('under', literal=True))
        .then(pl.lit('GO_0010629'))
        .when(expr & alteration.str.contains('norm', literal=True))
        .then(pl.lit('GO_0010467'))
    )


def _collect_set(column: str) -> pl.Expr:
    values = pl.col(column).drop_nulls().unique().sort()
    return pl.when(values.len() > 0).then(values.implode()).alias(column)


def _enrich(
    biomarkers: pl.DataFrame, sources: pl.DataFrame, diseases: pl.DataFrame, drugs: pl.DataFrame
) -> pl.DataFrame:
    source = pl.col('source')
    drug = pl.col('drug')
    exploded = (
        biomarkers
        .select(
            'Biomarker',
            'IndividualMutation',
            _split('Alteration', ';').alias('alterations'),
            _split('Gene', ';').alias('gene'),
            pl.col('AlterationType').str.split(';').alias('alteration_types'),
            _split('PrimaryTumorTypeFullName', ';').alias('tumor_type_full_name'),
            pl.col('Drug').str.replace_all(',', ';', literal=True).str.split(';').list.unique(maintain_order=True),
            'DrugFullName',
            'Association',
            'gDNA',
            _split('EvidenceLevel', ',').alias('confidence'),
            _split('Source', ';').alias('source'),
        )
        .explode('confidence', keep_nulls=False)
        .explode('tumor_type_full_name', keep_nulls=False)
        .with_columns(tumor_type=pl.col('tumor_type_full_name').str.replace_all('[ -]', ''))
        .explode('Drug', keep_nulls=False)
        .with_columns(drug=pl.col('Drug').str.replace_all(r'[\[\]]', ''))
        .drop('Drug')
        .explode('gene', keep_nulls=False)
        .with_columns(pl.col('gene').replace(GENENAMESOVERRIDE).str.to_uppercase())
    )
    return (
        explode_alterations(exploded)
        .with_columns(alteration=pl.coalesce('IndividualMutation', 'alteration'))
        .with_columns(alteration=_clean_alteration())
        .explode('source', keep_nulls=False)
        .with_columns(source=regexp_extract(source, r'(PMID:[0-9]+)|([A-Za-z0-9_ ]+)', 0).str.strip_chars(' '))
        .join(sources, on='source', how='left', maintain_order='left')
        .with_columns(
            literature=pl.when(source.str.starts_with('PMID')).then(regexp_extract(source, '(PMID:)([0-9]+)', 2)),
            urls=pl
            .when(source.str.starts_with('NCT'))
            .then(
                pl.struct(
                    niceName=pl.lit('Clinical Trials'), url=pl.lit('https://clinicaltrials.gov/ct2/show/') + source
                )
            )
            .otherwise(pl.struct('niceName', 'url')),
        )
        .rechunk()
        .with_columns(urls=pl.when(pl.col('urls').struct.field('niceName').is_not_null()).then('urls'))
        .with_columns(
            functionalConsequenceId=pl.col('alteration_type').replace(ALTERATIONTYPE2FUNCTIONCSQ),
            Association=pl.col('Association').replace(DRUGRESPONSE2EFO),
        )
        .join(diseases, on='tumor_type', how='left', maintain_order='left')
        .with_columns(drug=drug.str.to_uppercase())
        .with_columns(drug=pl.when(drug.is_null() | (drug.str.len_chars() == 0)).then('DrugFullName').otherwise(drug))
        .join(drugs, on='drug', how='left', maintain_order='left')
        .with_columns(
            drug=initcap(drug),
            variantId=pl.col('gDNA').map_elements(get_variant_id, return_dtype=pl.String, skip_nulls=True),
            geneExpressionId=_gene_expression_id(),
        )
        .with_columns(
            geneticVariation=pl.when(pl.col('alteration_type') != 'EXPR').then(
                pl.struct('functionalConsequenceId', id='variantId', name='alteration')
            ),
            geneExpression=pl.when(pl.col('alteration_type') == 'EXPR').then(
                pl.struct(id='geneExpressionId', name='alteration')
            ),
        )
    )


def generate(associations: str, source: str, biomarker_disease: str, drugs: str) -> pl.LazyFrame:
    """Build Cancer Biomarkers evidence from its associations, sources and disease mappings.

    Args:
        associations: location of the biomarker associations tsv.
        source: location of the json lines of source names and urls.
        biomarker_disease: location of the json lines mapping tumour types to diseases.
        drugs: location of the drug molecule dataset.

    Returns:
        Raw Cancer Biomarkers evidence, one row per drug, target, tumour type, evidence level and
        biomarker, with its literature, urls and alterations collected and sorted.
    """
    sources = scan_dataset(source, format='ndjson', infer_schema_length=None).select(
        pl.col('label').alias('niceName'), 'source', 'url'
    )
    diseases = scan_dataset(biomarker_disease, format='ndjson', infer_schema_length=None).select(
        pl.col('name').str.replace_all('_', '', literal=True).alias('tumor_type'),
        regexp_extract(pl.col('url'), '[^/]+$', 0).alias('diseaseFromSourceMappedId'),
    )
    drug_names = scan_dataset(drugs).select(pl.col('id').alias('drugId'), pl.col('name').alias('drug'))
    enriched = _enrich(
        scan_dataset(associations, format='tsv', infer_schema=False).collect(),
        sources.collect(),
        diseases.collect(),
        drug_names.collect(),
    )
    return (
        enriched
        .select(
            pl.lit('cancer_biomarkers').alias('datasourceId'),
            pl.lit('affected_pathway').alias('datatypeId'),
            pl.col('drug').alias('drugFromSource'),
            'drugId',
            pl.col('Association').alias('drugResponse'),
            pl.col('gene').alias('targetFromSourceId'),
            pl.col('tumor_type_full_name').alias('diseaseFromSource'),
            'diseaseFromSourceMappedId',
            'confidence',
            pl.col('Biomarker').alias('biomarkerName'),
            'literature',
            'urls',
            'geneticVariation',
            'geneExpression',
        )
        .group_by(EVIDENCE_KEYS, maintain_order=True)
        .agg(_collect_set(c) for c in ['literature', 'urls', 'geneticVariation', 'geneExpression'])
        .select(*EVIDENCE_KEYS, 'literature', 'urls', biomarkers=pl.struct('geneExpression', 'geneticVariation'))
        .lazy()
    )


def cancer_biomarkers(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process Cancer Biomarkers evidence.

    Args:
        source: `associations`, `source`, `biomarker_disease`, `drugs`, `disease`, `target` and
            `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    run_evidence(
        generate(source['associations'], source['source'], source['biomarker_disease'], source['drugs']),
        CANCER_BIOMARKERS,
        disease=source['disease'],
        target=source['target'],
        literature=source['literature'],
        destination=destination,
    )
