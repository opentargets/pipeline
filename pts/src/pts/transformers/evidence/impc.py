"""IMPC mouse model evidence.

Parses the IMPC mouse model files into evidence and post-processes it. Mouse models are linked to
human genes through MGI and HGNC, and to diseases by the phenotypes they share. Diseases are mapped
with the OnToma lookup tables through `add_efo_mapping`, by label and by id; then only the
top-scoring evidence of each model, gene and disease is kept.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.schemas.impc import DISEASE_MODEL_SUMMARY_SCHEMA
from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.expressions import direction_when_disease_present, divide
from pts.transformers.evidence.utils.luts import build_luts
from pts.transformers.evidence.utils.ontology import add_efo_mapping
from pts.transformers.evidence.utils.run import write_evidence
from pts.transformers.impc_common import (
    MODEL_ID,
    biological_model_id,
    literature,
    model_mouse_phenotypes,
    ontology_terms,
    phenotype_ids,
    read_gene_mapping,
    read_impc_csv,
    read_mgi_pubmed,
)

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


def _phenotype_set(id_column: str, term_column: str) -> pl.Expr:
    return pl.struct(pl.col(id_column).alias('id'), pl.col(term_column).alias('label')).unique().sort()


def parse_impc(source: dict[str, str], score_cutoff: float) -> pl.LazyFrame:
    """Raw IMPC evidence, before disease mapping, carrying the working `MODEL_ID` column.

    Args:
        source: the `SOURCES` locations.
        score_cutoff: the lowest `resourceScore` kept; a null score is dropped.
    """
    model_mouse = model_mouse_phenotypes(source['solr_mouse_model'])
    disease_human = (
        phenotype_ids(read_impc_csv(source['solr_disease']), 'disease_phenotypes', 'HP', 'hp_id')
        .select('disease_id', 'hp_id')
        .unique(maintain_order=True)
    )
    summary = (
        read_impc_csv(source['solr_disease_model_summary'])
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
    ontology = read_impc_csv(source['solr_ontology'])
    mp_terms, hp_terms = ontology_terms(ontology, 'MP'), ontology_terms(ontology, 'HP')
    gene_mapping = read_gene_mapping(source)
    model_literature = literature(read_mgi_pubmed(source['mouse_pubmed_refs']), summary, model_mouse)

    model_phenotypes = (
        model_mouse
        .join(mp_terms, on='mp_id', how='inner', maintain_order='left')
        .group_by(MODEL_ID)
        .agg(diseaseModelAssociatedModelPhenotypes=_phenotype_set('mp_id', 'mp_term'))
    )
    model_human = model_mouse.join(
        read_impc_csv(source['solr_ontology_ontology']), on='mp_id', how='inner', maintain_order='left'
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
        .join(model_literature, on=[MODEL_ID, 'targetInModelMgiId'], how='left', maintain_order='left')
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
    candidates = evidence.filter(top).drop(MODEL_ID, strict=False).unique()
    tied = candidates.group_by(UNIQUE_FIELDS).len().filter(pl.col('len') > 1).height
    deduplicated = (
        evidence
        # the unique fields make the output order total
        .sort(
            ['resourceScore', *TIE_BREAK, *UNIQUE_FIELDS],
            descending=[True] + [False] * (len(TIE_BREAK) + len(UNIQUE_FIELDS)),
            nulls_last=True,
        )
        .unique(subset=UNIQUE_FIELDS, keep='first', maintain_order=True)
        .drop(MODEL_ID, strict=False)
        .with_columns(datasourceId=pl.lit('impc'), datatypeId=pl.lit('animal_model'))
    )
    logger.info(f'{tied} of {deduplicated.height} impc evidence partitions tie on the top resourceScore')
    return deduplicated


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
