"""Search index generation for diseases, targets, drugs, variants and studies.

Ported from the pyspark `search` job, which was itself a port of the platform-etl-backend
Search step (`Search.scala`). Builds search-index documents with ranked terms, keywords,
prefixes and ngrams for each entity type.

This step is a SINGLE task, and its name in `config.yaml` decides where it runs: a PTS step
goes to Dataproc if and only if one of its tasks is NAMED `pyspark …`. Renaming the task is
what moves this whole step onto a plain GCE VM -- do not reintroduce that prefix.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.search.disease import build_disease_index
from pts.transformers.search.drug import build_drug_index
from pts.transformers.search.lookups import (
    association_scores,
    disease_lut,
    drug_associations,
    drug_associations_from_evidence,
    drug_lut,
    nct_by,
    phenotype_names,
    resolve_ta_labels,
    scored_drug_associations,
    target_lut,
)
from pts.transformers.search.study import build_study_index
from pts.transformers.search.target import build_target_index
from pts.transformers.search.variant import build_variant_index
from pts.transformers.utils.dataset import scan_dataset, scan_datasets, write_dataset


def _drugs_with_mechanisms(drug_raw: pl.LazyFrame, mechanism: pl.LazyFrame, indication: pl.LazyFrame) -> pl.LazyFrame:
    """Attach mechanism-of-action rows and indication ids to the drug frame."""
    mechanisms = (
        mechanism.filter(pl.col('chemblIds').list.len() > 0)
        .explode('chemblIds', empty_as_null=False)
        .rename({'chemblIds': 'drugId'})
        .group_by('drugId')
        .agg(pl.struct('mechanismOfAction', 'references', 'targetName', 'targets').alias('rows'))
    )
    indications = indication.group_by(pl.col('drugId')).agg(pl.col('diseaseId').alias('indications'))
    return (
        drug_raw.rename({'id': 'drugId'})
        .join(mechanisms, on='drugId', how='left')
        .join(indications, on='drugId', how='left')
    )


def search(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Run the search index generation pipeline.

    Args:
        source: paths keyed by `association`, `drug`, `evidence`, `indication`, `mechanism`,
            `target`, `credible_sets`, `disease`, `disease_hpo`, `hpo`, `studies`, `variants`.
            Note `disease_hpo` points at the `disease_phenotype` dataset and `hpo` at the
            `disease_hpo` one -- the config keys are crossed relative to the dataset names.
        destination: paths keyed by `diseases`, `targets`, `drugs`, `variants`, `studies`.
        settings: unused; reserved for future configuration.
        config: unused; part of the transformer contract.
    """
    logger.info('loading input data')
    diseases = resolve_ta_labels(scan_dataset(source['disease']).rename({'id': 'diseaseId'}))
    targets = scan_dataset(source['target']).rename({'id': 'targetId'})
    drugs = _drugs_with_mechanisms(
        scan_dataset(source['drug']),
        scan_dataset(source['mechanism']),
        scan_dataset(source['indication']),
    )
    indication = scan_dataset(source['indication'])
    variants = scan_dataset(source['variants']).select(
        'variantId', 'rsIds', 'hgvsId', 'dbXrefs', 'chromosome', 'position', 'transcriptConsequences'
    )
    studies = scan_dataset(source['studies']).select(
        'studyId', 'traitFromSource', 'pubmedId', 'publicationFirstAuthor', 'diseaseIds', 'nSamples', 'geneId'
    )
    # Most evidence datasets carry no `drugId`; the union null-fills it.
    evidence = scan_datasets(source['evidence']).select('drugId', 'targetId', 'diseaseId')

    logger.info('building lookup tables')
    d_lut = disease_lut(diseases)
    t_lut = target_lut(targets)
    dr_lut = drug_lut(drugs)
    phenotypes = phenotype_names(scan_dataset(source['disease_hpo']), scan_dataset(source['hpo']))

    logger.info('building association scores')
    scores = association_scores(scan_dataset(source['association']))
    from_evidence = drug_associations_from_evidence(evidence).collect().lazy()
    # Counted BEFORE the join with scores, as spark does. See the function's docstring.
    total = from_evidence.select(pl.len()).collect().item()
    scored_drugs = scored_drug_associations(from_evidence, scores).collect().lazy()
    drug_assocs = drug_associations(scored_drugs, total)

    logger.info('building search index for diseases')
    write_dataset(
        build_disease_index(
            diseases, phenotypes, scores, scored_drugs, t_lut, dr_lut, studies, nct_by(indication, 'diseaseId')
        ),
        destination['diseases'],
    )

    logger.info('building search index for targets')
    write_dataset(build_target_index(targets, scores, d_lut, variants, scored_drugs, dr_lut), destination['targets'])

    logger.info('building search index for drugs')
    write_dataset(
        build_drug_index(drugs, drug_assocs, t_lut, d_lut, nct_by(indication, 'drugId')),
        destination['drugs'],
    )

    logger.info('building search index for variants')
    write_dataset(build_variant_index(variants), destination['variants'])

    logger.info('building search index for studies')
    write_dataset(
        build_study_index(studies, targets, scan_dataset(source['credible_sets'])),
        destination['studies'],
    )
