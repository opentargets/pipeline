"""IMPC mouse model evidence.

Polars port of the `pyspark impc` parser together with its `evidence_postprocess` step. Mouse
models are linked to human genes through MGI and HGNC, and to diseases by the phenotypes they
share. Diseases are mapped with the OnToma lookup tables through `add_efo_mapping`, by label and by
id; then only the top-scoring evidence of each model, gene and disease is kept.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.schemas.impc import DISEASE_MODEL_SUMMARY_SCHEMA, MGI_PUBMED_COLUMNS
from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.expressions import direction_when_disease_present, divide
from pts.transformers.evidence.utils.luts import build_luts
from pts.transformers.evidence.utils.ontology import add_efo_mapping
from pts.transformers.evidence.utils.run import write_evidence
from pts.transformers.utils.spark_csv import read_spark_csv

#: The `source` keys of the IMPC inputs.
SOURCES = [
    'solr_gene_gene',
    'solr_ontology_ontology',
    'solr_mouse_model',
    'solr_disease',
    'solr_disease_model_summary',
    'solr_ontology',
    'hgnc_gene_mappings',
    'mouse_gene_mappings',
    'mouse_pubmed_refs',
]

#: Working column: the source model id, kept until deduplication.
MODEL_ID = 'model_id'

#: Evidence fields of which only the top-scoring row is kept.
UNIQUE_FIELDS = [
    'diseaseFromSource',
    'targetInModel',
    'biologicalModelAllelicComposition',
    'biologicalModelGeneticBackground',
    'diseaseFromSourceMappedId',
    'targetFromSourceId',
]

#: Ascending keys, nulls last, choosing among rows tied on the top `resourceScore`.
TIE_BREAK = [MODEL_ID, 'diseaseFromSourceId', 'targetInModelMgiId', 'targetInModelEnsemblId']

IMPC = EvidenceDatasource(
    datasource_id='impc',
    # order feeds the id, as in the spark config
    unique_fields=[
        'targetId',
        'targetFromSourceId',
        'diseaseId',
        'datasourceId',
        'biologicalModelGeneticBackground',
        'targetInModel',
        'biologicalModelAllelicComposition',
        'diseaseFromSource',
    ],
    score=divide(pl.col('resourceScore'), 100),
)
DIRECTION_ON_TRAIT = direction_when_disease_present('risk')
DIRECTION_ON_TARGET = direction_when_disease_present('LoF')

#: Java's `$` also matches before a final line terminator.
_MGI_MODEL_RE = r'^MGI:[0-9]+(?:\r\n|[\n\r\x{0085}\x{2028}\x{2029}])?$'

#: Evidence columns in the order the spark step produced them.
_EVIDENCE_COLUMNS = [
    'targetInModelMgiId',
    'diseaseFromSourceId',
    'biologicalModelGeneticBackground',
    'biologicalModelAllelicComposition',
    'diseaseFromSource',
    'resourceScore',
    'targetInModel',
    'targetInModelEnsemblId',
    'targetFromSourceId',
    'diseaseModelAssociatedModelPhenotypes',
    'diseaseModelAssociatedHumanPhenotypes',
    'literature',
    'biologicalModelId',
]


def biological_model_id(model_id: pl.Expr) -> pl.Expr:
    """The model id before its first `#`, if it is an MGI id; null otherwise."""
    cleaned = model_id.str.split('#').list.first()
    return pl.when(cleaned.str.contains(_MGI_MODEL_RE)).then(cleaned)


def _explode_non_empty(lf: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """Spark's `explode`: rows with a null or empty list are dropped."""
    return lf.filter(pl.col(column).list.len() > 0).explode(column)


def _phenotype_ids(lf: pl.LazyFrame, column: str, prefix: str, id_column: str) -> pl.LazyFrame:
    """One row per `<prefix>:<digits>` id in `column`; rows without one are dropped."""
    return _explode_non_empty(
        lf.with_columns(pl.col(column).str.extract_all(f'{prefix}:[0-9]+').alias(id_column)), id_column
    )


def _ontology_terms(ontology: pl.LazyFrame, name: str) -> pl.LazyFrame:
    prefix = name.lower()
    return ontology.filter(pl.col('ontology') == name).select(
        pl.col('phenotype_id').alias(f'{prefix}_id'), pl.col('phenotype_term').alias(f'{prefix}_term')
    )


def _phenotype_set(id_column: str, term_column: str) -> pl.Expr:
    return pl.struct(pl.col(id_column).alias('id'), pl.col(term_column).alias('label')).unique().sort()


def _gene_mapping(mgi: pl.LazyFrame, mouse_to_human: pl.LazyFrame, hgnc: pl.LazyFrame) -> pl.LazyFrame:
    """MGI gene id to mouse symbol, mouse Ensembl id and human Ensembl id, through HGNC."""
    return (
        mgi
        .select(
            targetInModelMgiId=pl.col('1. MGI accession id'),
            targetInModel=pl.col('3. marker symbol'),
            targetInModelEnsemblId=pl.col('11. Ensembl gene id'),
        )
        .filter(pl.col('targetInModelEnsemblId').is_not_null())
        .join(
            mouse_to_human.select(targetInModelMgiId=pl.col('gene_id'), hgnc_gene_id=pl.col('hgnc_gene_id')),
            on='targetInModelMgiId',
            how='inner',
            maintain_order='left',
        )
        .join(
            hgnc.select(hgnc_gene_id=pl.col('hgnc_id'), targetFromSourceId=pl.col('ensembl_gene_id')),
            on='hgnc_gene_id',
            how='inner',
            maintain_order='left',
        )
        .filter(pl.col('targetFromSourceId').is_not_null())
        .select('targetInModelMgiId', 'targetInModel', 'targetInModelEnsemblId', 'targetFromSourceId')
    )


def _literature(mgi_pubmed: pl.LazyFrame, summary: pl.LazyFrame, model_mouse: pl.LazyFrame) -> pl.LazyFrame:
    """Sorted distinct PMIDs per model and MGI gene, from the gene's references for the model's phenotypes."""
    references = mgi_pubmed.with_columns(pl.col('literature', 'targetInModelMgiId').str.split('|'))
    references = _explode_non_empty(_explode_non_empty(references, 'literature'), 'targetInModelMgiId')
    return (
        summary
        .select(MODEL_ID, 'targetInModelMgiId')
        .unique(maintain_order=True)
        .join(model_mouse, on=MODEL_ID, how='inner', maintain_order='left')
        .join(
            references.select('mp_id', 'literature', 'targetInModelMgiId').unique(maintain_order=True),
            on=['targetInModelMgiId', 'mp_id'],
            how='inner',
            maintain_order='left',
        )
        .group_by(MODEL_ID, 'targetInModelMgiId')
        .agg(pl.col('literature').unique().sort())
    )


def _read(path: str, **options: Any) -> pl.LazyFrame:
    return read_spark_csv(path, **options).lazy()


def parse_impc(source: dict[str, str], score_cutoff: float) -> pl.LazyFrame:
    """Raw IMPC evidence, before disease mapping, carrying the working `MODEL_ID` column.

    Args:
        source: the `SOURCES` locations.
        score_cutoff: the lowest `resourceScore` kept; a null score is dropped.
    """
    model_mouse = _phenotype_ids(_read(source['solr_mouse_model']), 'model_phenotypes', 'MP', 'mp_id').select(
        MODEL_ID, 'mp_id'
    )
    disease_human = (
        _phenotype_ids(_read(source['solr_disease']), 'disease_phenotypes', 'HP', 'hp_id')
        .select('disease_id', 'hp_id')
        .unique(maintain_order=True)
    )
    summary = (
        _read(source['solr_disease_model_summary'])
        .select(list(DISEASE_MODEL_SUMMARY_SCHEMA))
        .cast(DISEASE_MODEL_SUMMARY_SCHEMA, strict=False)
        .select(
            MODEL_ID,
            biologicalModelGeneticBackground=pl.col('model_genetic_background'),
            biologicalModelAllelicComposition=pl.col('model_description'),
            disease_id=pl.col('disease_id'),
            disease_term=pl.col('disease_term'),
            targetInModelMgiId=pl.col('marker_id'),
            resourceScore=pl.col('disease_model_avg_norm'),
        )
        .unique(maintain_order=True)
    )
    ontology = _read(source['solr_ontology'])
    mp_terms, hp_terms = _ontology_terms(ontology, 'MP'), _ontology_terms(ontology, 'HP')
    gene_mapping = _gene_mapping(
        _read(source['mouse_gene_mappings'], separator='\t', null_values=['null']),
        _read(source['solr_gene_gene']),
        _read(source['hgnc_gene_mappings'], separator='\t', null_values=['null']),
    )
    literature = _literature(
        _read(
            source['mouse_pubmed_refs'],
            separator='\t',
            has_header=False,
            new_columns=MGI_PUBMED_COLUMNS,
            null_values=['null'],
        ),
        summary,
        model_mouse,
    )

    model_phenotypes = (
        model_mouse
        .join(mp_terms, on='mp_id', how='inner', maintain_order='left')
        .group_by(MODEL_ID)
        .agg(diseaseModelAssociatedModelPhenotypes=_phenotype_set('mp_id', 'mp_term'))
    )
    model_human = model_mouse.join(
        _read(source['solr_ontology_ontology']), on='mp_id', how='inner', maintain_order='left'
    ).select(MODEL_ID, 'hp_id')
    human_phenotypes = (
        summary
        .select(MODEL_ID, 'disease_id')
        .join(disease_human, on='disease_id', how='inner', maintain_order='left')
        .join(model_human, on=[MODEL_ID, 'hp_id'], how='inner', maintain_order='left')
        .join(hp_terms, on='hp_id', how='inner', maintain_order='left')
        .group_by(MODEL_ID, 'disease_id')
        .agg(diseaseModelAssociatedHumanPhenotypes=_phenotype_set('hp_id', 'hp_term'))
    )

    return (
        summary
        .filter((pl.col('resourceScore') < score_cutoff).not_())
        .join(gene_mapping, on='targetInModelMgiId', how='inner', maintain_order='left')
        .join(model_phenotypes, on=MODEL_ID, how='left', maintain_order='left')
        .join(human_phenotypes, on=[MODEL_ID, 'disease_id'], how='left', maintain_order='left')
        .join(literature, on=[MODEL_ID, 'targetInModelMgiId'], how='left', maintain_order='left')
        .rename({'disease_id': 'diseaseFromSourceId', 'disease_term': 'diseaseFromSource'})
        .with_columns(biologicalModelId=biological_model_id(pl.col(MODEL_ID)))
        .select(*_EVIDENCE_COLUMNS, MODEL_ID)
    )


def deduplicate(evidence: pl.DataFrame) -> pl.DataFrame:
    """Keep the top `resourceScore` row of each `UNIQUE_FIELDS` group, ties broken by `TIE_BREAK`.

    Logs the groups whose top score is shared by distinct rows, then drops `MODEL_ID` and adds the
    datasource and datatype.
    """
    top = pl.col('resourceScore') == pl.col('resourceScore').max().over(UNIQUE_FIELDS)
    candidates = evidence.drop(MODEL_ID, strict=False).unique().filter(top)
    tied = candidates.group_by(UNIQUE_FIELDS).len().filter(pl.col('len') > 1).height
    groups = evidence.select(UNIQUE_FIELDS).unique().height
    logger.info(f'{tied} of {groups} impc evidence partitions tie on the top resourceScore')
    return (
        evidence
        .sort(['resourceScore', *TIE_BREAK], descending=[True] + [False] * len(TIE_BREAK), nulls_last=True)
        .unique(subset=UNIQUE_FIELDS, keep='first', maintain_order=True)
        .drop(MODEL_ID, strict=False)
        .with_columns(datasourceId=pl.lit('impc'), datatypeId=pl.lit('animal_model'))
    )


def impc(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process IMPC evidence.

    Args:
        source: the `SOURCES`, `ontoma_disease_label_lut`, `ontoma_disease_id_lut`, `disease`, `target`
            and `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: `score_cutoff`, the lowest `resourceScore` kept.
        config: unused.
    """
    logger.info(f'load impc data from {[source[s] for s in SOURCES]}')
    mapped = add_efo_mapping(
        parse_impc(source, float(settings.get('score_cutoff', 0))),
        disease_label_lut_path=source['ontoma_disease_label_lut'],
        disease_id_lut_path=source['ontoma_disease_id_lut'],
    ).collect()
    raw = deduplicate(mapped)
    luts = build_luts(raw.lazy(), disease=source['disease'], target=source['target'], literature=source['literature'])
    evidence = (
        IMPC
        .postprocess(raw.lazy(), luts)
        .assign_direction_on_trait(DIRECTION_ON_TRAIT)
        .assign_direction_on_target(DIRECTION_ON_TARGET, luts.target)
        .hash_long_variant_identifiers()
    )
    write_evidence(evidence, destination)
