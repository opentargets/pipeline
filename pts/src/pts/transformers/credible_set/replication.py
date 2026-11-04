"""The REPLICATED rule: credible sets whose lead is reported on one of their traits by two independent reports."""

from __future__ import annotations

import polars as pl

from pts.schemas.study import StudyType
from pts.transformers.credible_set.study_lookup import qtl_genes
from pts.transformers.disease.coalescing import MEASUREMENT


def gwas_traits(study: pl.LazyFrame, disease: pl.LazyFrame) -> pl.LazyFrame:
    """The traits a GWAS study replicates on, by the kind of its `diseaseIds` terms, as on the `main` branch.

    - only measurements: one trait per measurement, so one shared measurement is enough;
    - measurements and diseases: one trait, the sorted diseases, the measurements set aside;
    - only diseases: one trait, the sorted diseases.

    A term is a measurement when the disease index files it under the measurement therapeutic area;
    a term absent from the disease index is a disease.

    Examples:
        >>> study = pl.LazyFrame({
        ...     'studyId': ['m', 'md', 'd', 'q'],
        ...     'studyType': ['gwas', 'gwas', 'gwas', 'eqtl'],
        ...     'diseaseIds': [['bmi', 'height'], ['bmi', 'ibd'], ['ra', 'ibd'], []],
        ... })
        >>> disease = pl.LazyFrame({'id': ['bmi', 'height'], 'therapeuticAreas': [[MEASUREMENT]] * 2})
        >>> gwas_traits(study, disease).sort('studyId', '_trait').collect().rows()
        [('d', ['ibd', 'ra']), ('m', ['bmi']), ('m', ['height']), ('md', ['ibd'])]
    """
    measurement = (
        disease
        .filter(pl.col('therapeuticAreas').list.contains(MEASUREMENT))
        .select(pl.col('id').alias('_term'), pl.lit(True).alias('_is_measurement'))
        .unique()
    )
    term = (
        study
        .filter(pl.col('studyType') == StudyType.GWAS.value)
        .select('studyId', pl.col('diseaseIds').alias('_term'))
        .explode('_term')
        .drop_nulls('_term')
        .unique()
        .join(measurement, on='_term', how='left')
    )
    disease_trait = (
        term.filter(pl.col('_is_measurement').is_null()).group_by('studyId').agg(pl.col('_term').sort().alias('_trait'))
    )
    measurement_trait = term.join(disease_trait, on='studyId', how='anti').select(
        'studyId', pl.concat_list('_term').alias('_trait')
    )
    return pl.concat([disease_trait, measurement_trait])


def gwas_reports(credible_set: pl.LazyFrame, study: pl.LazyFrame, disease: pl.LazyFrame) -> pl.LazyFrame:
    """One row per GWAS set and trait of `gwas_traits`: `id`, `leadVariantId`, `_trait` and `_evidence`.

    The evidence is the study's cohorts, publication and populations, so studies sharing all
    three are one report.

    Examples:
        >>> credible_set = pl.LazyFrame({'id': ['a', 'b'], 'studyId': ['g1', 'g2'], 'leadVariantId': ['1_1_A_T'] * 2})
        >>> study = pl.LazyFrame({
        ...     'studyId': ['g1', 'g2'],
        ...     'studyType': ['gwas', 'gwas'],
        ...     'diseaseIds': [['ra'], ['ra']],
        ...     'pubmedId': ['p', 'p'],
        ...     'cohorts': [['UKB'], ['UKB']],
        ...     'ldPopulationStructure': [None, None],
        ... })
        >>> disease = pl.LazyFrame({'id': ['ra'], 'therapeuticAreas': [['EFO_area']]})
        >>> report = gwas_reports(credible_set, study, disease).collect()
        >>> report['_evidence'].n_unique()
        1
    """
    evidence = study.select('studyId', pl.struct('cohorts', 'pubmedId', 'ldPopulationStructure').alias('_evidence'))
    return (
        credible_set
        .select('id', 'studyId', 'leadVariantId')
        .join(gwas_traits(study, disease), on='studyId')
        .join(evidence, on='studyId')
        .select('id', 'leadVariantId', '_trait', '_evidence')
    )


def qtl_reports(credible_set: pl.LazyFrame, study: pl.LazyFrame) -> pl.LazyFrame:
    """One row per QTL set of a study with a `geneId`: `id`, `leadVariantId`, `_trait` and `_evidence`.

    The trait is the gene, and the evidence is the credible set itself, so two sets of one study
    are two reports.

    Examples:
        >>> credible_set = pl.LazyFrame({'id': ['a', 'b'], 'studyId': ['q', 'q'], 'leadVariantId': ['1_1_A_T'] * 2})
        >>> study = pl.LazyFrame({'studyId': ['q'], 'studyType': ['eqtl'], 'geneId': ['ENSG1']})
        >>> qtl_reports(credible_set, study).sort('id').collect().rows()
        [('a', '1_1_A_T', 'ENSG1', 'a'), ('b', '1_1_A_T', 'ENSG1', 'b')]
    """
    return (
        credible_set
        .select('id', 'studyId', 'leadVariantId')
        .join(qtl_genes(study), on='studyId')
        .select('id', 'leadVariantId', pl.col('geneId').alias('_trait'), pl.col('id').alias('_evidence'))
    )


def replicated(report: pl.LazyFrame) -> pl.LazyFrame:
    """The ids of the reports whose lead and trait have at least two distinct pieces of evidence.

    Examples:
        >>> report = pl.LazyFrame({
        ...     'id': ['a', 'b', 'c'],
        ...     'leadVariantId': ['1_1_A_T'] * 3,
        ...     '_trait': ['ra', 'ra', 'ibd'],
        ...     '_evidence': ['p1', 'p2', 'p3'],
        ... })
        >>> replicated(report).sort('id').collect()['id'].to_list()
        ['a', 'b']
    """
    independent_report_count = pl.col('_evidence').n_unique().over('leadVariantId', '_trait')
    return report.filter(independent_report_count >= 2).select('id')


def replicated_sets(credible_set: pl.LazyFrame, study: pl.LazyFrame, disease: pl.LazyFrame) -> pl.LazyFrame:
    """The ids of the GWAS and QTL sets that are `replicated`, on `gwas_reports` and `qtl_reports`.

    This is the REPLICATED rule of the `main` branch.

    Examples:
        >>> credible_set = pl.LazyFrame({
        ...     'id': ['a', 'b', 'c'],
        ...     'studyId': ['s1', 's2', 's3'],
        ...     'leadVariantId': ['1_1_A_T'] * 3,
        ... })
        >>> study = pl.LazyFrame({
        ...     'studyId': ['s1', 's2', 's3'],
        ...     'studyType': ['gwas'] * 3,
        ...     'diseaseIds': [['ra'], ['ra'], ['ibd']],
        ...     'geneId': [None] * 3,
        ...     'pubmedId': ['p1', 'p2', 'p3'],
        ...     'cohorts': [None] * 3,
        ...     'ldPopulationStructure': [None] * 3,
        ... })
        >>> disease = pl.LazyFrame({'id': ['ra'], 'therapeuticAreas': [['EFO_area']]})
        >>> replicated_sets(credible_set, study, disease).sort('id').collect()['id'].to_list()
        ['a', 'b']
    """
    gwas = replicated(gwas_reports(credible_set, study, disease))
    qtl = replicated(qtl_reports(credible_set, study))
    return pl.concat([gwas, qtl]).unique()
