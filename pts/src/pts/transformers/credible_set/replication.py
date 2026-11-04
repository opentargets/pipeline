"""The REPLICATED rule: credible sets whose lead is reported on one of their traits by two independent reports.

A report is one row of `REPORT_COLUMNS`: a credible set, its lead, a trait it reports the lead on,
and the evidence that tells independent reports apart. Each kind of set builds its own reports, and
one rule, `independently_reported_ids`, applies to all of them.
"""

from __future__ import annotations

import polars as pl

from pts.schemas.study import StudyType
from pts.transformers.credible_set.study_lookup import qtl_genes
from pts.transformers.disease.coalescing import MEASUREMENT

REPORT_COLUMNS = ('id', 'leadVariantId', 'trait', 'evidence')
"""The columns of a report: the credible set, its lead, the trait and the evidence of the report."""

MIN_INDEPENDENT_REPORTS = 2
"""The number of independent reports of a lead on a trait that make it replicated."""


def measurement_term(disease: pl.LazyFrame) -> pl.LazyFrame:
    """The `id` of the disease index terms filed under the measurement therapeutic area.

    Examples:
        >>> disease = pl.LazyFrame({'id': ['bmi', 'ra'], 'therapeuticAreas': [[MEASUREMENT], ['EFO_area']]})
        >>> measurement_term(disease).collect()['id'].to_list()
        ['bmi']
    """
    return disease.filter(pl.col('therapeuticAreas').list.contains(MEASUREMENT)).select('id').unique()


def gwas_traits(study: pl.LazyFrame, disease: pl.LazyFrame) -> pl.LazyFrame:
    """The traits a GWAS study replicates on, from the kind of its `diseaseIds` terms.

    - A study with at least one disease term has one trait: its sorted disease terms. Its
      measurement terms, if any, are set aside: next to a disease, a measurement mostly describes
      the context of the association (an interaction partner, a covariate, the population or a
      treatment, as in a BMI-adjusted study of IBD), not a second trait, so the study must not
      replicate the plain GWAS of that measurement.
    - A study with only measurement terms has one trait per measurement, so one shared measurement
      is enough.

    A term absent from the disease index is a disease. A study without `diseaseIds` has no trait.

    Examples:
        >>> study = pl.LazyFrame({
        ...     'studyId': ['m', 'md', 'd', 'q'],
        ...     'studyType': ['gwas', 'gwas', 'gwas', 'eqtl'],
        ...     'diseaseIds': [['bmi', 'height'], ['bmi', 'ibd'], ['ra', 'ibd'], []],
        ... })
        >>> disease = pl.LazyFrame({'id': ['bmi', 'height'], 'therapeuticAreas': [[MEASUREMENT]] * 2})
        >>> gwas_traits(study, disease).sort('studyId', 'trait').collect().rows()
        [('d', ['ibd', 'ra']), ('m', ['bmi']), ('m', ['height']), ('md', ['ibd'])]
    """
    is_measurement = measurement_term(disease).select(pl.col('id').alias('term'), pl.lit(True).alias('is_measurement'))
    term_by_study = (
        study
        .filter(pl.col('studyType') == StudyType.GWAS.value)
        .select('studyId', pl.col('diseaseIds').alias('term'))
        .explode('term')
        .drop_nulls('term')
        .unique()
        .join(is_measurement, on='term', how='left')
        .group_by('studyId')
        .agg(
            pl.col('term').filter(pl.col('is_measurement').is_null()).sort().alias('disease_term'),
            pl.col('term').filter(pl.col('is_measurement')).alias('measurement_term'),
        )
    )
    has_disease = pl.col('disease_term').list.len() > 0
    disease_trait = term_by_study.filter(has_disease).select('studyId', pl.col('disease_term').alias('trait'))
    measurement_trait = (
        term_by_study
        .filter(~has_disease)
        .explode('measurement_term')
        .select('studyId', pl.concat_list('measurement_term').alias('trait'))
    )
    return pl.concat([disease_trait, measurement_trait])


def gwas_reports(credible_set: pl.LazyFrame, study: pl.LazyFrame, disease: pl.LazyFrame) -> pl.LazyFrame:
    """The reports of the GWAS sets: one per set and trait of `gwas_traits`.

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
        >>> report['evidence'].n_unique()
        1
    """
    evidence = study.select('studyId', pl.struct('cohorts', 'pubmedId', 'ldPopulationStructure').alias('evidence'))
    return (
        credible_set
        .select('id', 'studyId', 'leadVariantId')
        .join(gwas_traits(study, disease), on='studyId')
        .join(evidence, on='studyId')
        .select(REPORT_COLUMNS)
    )


def qtl_reports(credible_set: pl.LazyFrame, study: pl.LazyFrame) -> pl.LazyFrame:
    """The reports of the QTL sets of a study with a `geneId`: one per set, on that gene.

    The evidence is the credible set itself, so two sets of one study are two reports.

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
        .select('id', 'leadVariantId', pl.col('geneId').alias('trait'), pl.col('id').alias('evidence'))
        .select(REPORT_COLUMNS)
    )


def independently_reported_ids(report: pl.LazyFrame) -> pl.LazyFrame:
    """The ids of the reports whose lead and trait have `MIN_INDEPENDENT_REPORTS` or more distinct evidence values.

    Examples:
        >>> report = pl.LazyFrame({
        ...     'id': ['a', 'b', 'c'],
        ...     'leadVariantId': ['1_1_A_T'] * 3,
        ...     'trait': ['ra', 'ra', 'ibd'],
        ...     'evidence': ['p1', 'p2', 'p3'],
        ... })
        >>> independently_reported_ids(report).sort('id').collect()['id'].to_list()
        ['a', 'b']
    """
    independent_report_count = pl.col('evidence').n_unique().over('leadVariantId', 'trait')
    return report.filter(independent_report_count >= MIN_INDEPENDENT_REPORTS).select('id')


def replicated_sets(credible_set: pl.LazyFrame, study: pl.LazyFrame, disease: pl.LazyFrame) -> pl.LazyFrame:
    """The ids of the GWAS and QTL sets whose lead is reported on one of their traits by independent reports.

    The reports are those of `gwas_reports` and `qtl_reports`, and the rule is
    `independently_reported_ids`.

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
    gwas = independently_reported_ids(gwas_reports(credible_set, study, disease))
    qtl = independently_reported_ids(qtl_reports(credible_set, study))
    return pl.concat([gwas, qtl]).unique()
