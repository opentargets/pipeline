"""The REPLICATED rule: credible sets whose lead is reported on one of their traits by two independent reports.

A lead found once may be chance; found again on the same trait by an independent study, it is
replicated. A report is one row of `_REPORT_COLUMNS`: a credible set, its lead, a trait it reports
the lead on, and the evidence that tells independent reports apart.

`replicated_sets` builds the reports of each kind of set, then applies one rule to all of them
(`_independently_reported_ids`):

- a GWAS set reports its lead on each of its study's traits (`_gwas_traits`), and two reports are
  independent when they come from different publications, cohorts or populations;
- a QTL set reports its lead on its study's gene, and every QTL study is independent of another.
"""

from __future__ import annotations

import polars as pl

from pts.schemas.study import StudyType
from pts.transformers.credible_set.source import qtl_genes
from pts.transformers.disease.coalescing import MEASUREMENT

_REPORT_COLUMNS = ('id', 'leadVariantId', 'trait', 'evidence')
"""The columns of a report: the credible set, its lead, the trait and the evidence of the report."""


_MIN_INDEPENDENT_REPORTS = 2
"""The number of independent reports of a lead on a trait that make it replicated."""


def replicated_sets(credible_set: pl.LazyFrame, study: pl.LazyFrame, disease: pl.LazyFrame) -> pl.LazyFrame:
    """The ids of the GWAS and QTL sets whose lead is reported on one of their traits by independent reports.

    The reports are those of `_gwas_reports` and `_qtl_reports`, and the rule is
    `_independently_reported_ids`.

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
    gwas = credible_set.pipe(_gwas_reports, study, disease).pipe(_independently_reported_ids)
    qtl = credible_set.pipe(_qtl_reports, study).pipe(_independently_reported_ids)
    return pl.concat([gwas, qtl]).unique()


def _gwas_reports(credible_set: pl.LazyFrame, study: pl.LazyFrame, disease: pl.LazyFrame) -> pl.LazyFrame:
    """The reports of the GWAS sets: one per set and trait of `_gwas_traits`.

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
        >>> report = _gwas_reports(credible_set, study, disease).collect()
        >>> report['evidence'].n_unique()
        1
    """
    evidence = study.select('studyId', pl.struct('cohorts', 'pubmedId', 'ldPopulationStructure').alias('evidence'))
    return (
        credible_set
        .select('id', 'studyId', 'leadVariantId')
        .join(_gwas_traits(study, disease), on='studyId')
        .join(evidence, on='studyId')
        .select(_REPORT_COLUMNS)
    )


def _gwas_traits(study: pl.LazyFrame, disease: pl.LazyFrame) -> pl.LazyFrame:
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
        `bmi` and `height` are measurements, `ibd` and `ra` diseases. Each study is named after its
        case, and lists the traits it replicates on:

        >>> disease = pl.LazyFrame({
        ...     'id': ['bmi', 'height', 'ibd', 'ra'],
        ...     'therapeuticAreas': [[MEASUREMENT], [MEASUREMENT], ['EFO_area'], ['EFO_area']],
        ... })
        >>> study = pl.LazyFrame({
        ...     'studyId': [
        ...         'one measurement, one disease',
        ...         'one disease',
        ...         'many diseases',
        ...         'many measurements',
        ...         'one disease, many measurements',
        ...         'many diseases, one measurement',
        ...         'many diseases, many measurements',
        ...         'empty list',
        ...     ],
        ...     'studyType': ['gwas'] * 8,
        ...     'diseaseIds': [
        ...         ['bmi', 'ibd'],
        ...         ['ibd'],
        ...         ['ra', 'ibd'],
        ...         ['height', 'bmi'],
        ...         ['ibd', 'bmi', 'height'],
        ...         ['ra', 'ibd', 'bmi'],
        ...         ['ra', 'bmi', 'ibd', 'height'],
        ...         [],
        ...     ],
        ... })
        >>> trait = _gwas_traits(study, disease).group_by('studyId').agg(pl.col('trait').sort())
        >>> by_study = study.select('studyId').join(trait, on='studyId', how='left', maintain_order='left')
        >>> for study_id, study_trait in by_study.collect().rows():
        ...     print(f'{study_id}: {study_trait or []}')
        one measurement, one disease: [['ibd']]
        one disease: [['ibd']]
        many diseases: [['ibd', 'ra']]
        many measurements: [['bmi'], ['height']]
        one disease, many measurements: [['ibd']]
        many diseases, one measurement: [['ibd', 'ra']]
        many diseases, many measurements: [['ibd', 'ra']]
        empty list: []
    """
    is_measurement = _measurement_term(disease).select(pl.col('id').alias('term'), pl.lit(True).alias('is_measurement'))
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
            pl.col('term').filter(pl.col('is_measurement')).alias('_measurement_term'),
        )
    )
    has_disease = pl.col('disease_term').list.len() > 0
    disease_trait = term_by_study.filter(has_disease).select('studyId', pl.col('disease_term').alias('trait'))
    measurement_trait = (
        term_by_study
        .filter(~has_disease)
        .explode('_measurement_term')
        .select('studyId', pl.concat_list('_measurement_term').alias('trait'))
    )
    return pl.concat([disease_trait, measurement_trait])


def _measurement_term(disease: pl.LazyFrame) -> pl.LazyFrame:
    """The `id` of the disease index terms filed under the measurement therapeutic area.

    Examples:
        >>> disease = pl.LazyFrame({'id': ['bmi', 'ra'], 'therapeuticAreas': [[MEASUREMENT], ['EFO_area']]})
        >>> _measurement_term(disease).collect()['id'].to_list()
        ['bmi']
    """
    return disease.filter(pl.col('therapeuticAreas').list.contains(MEASUREMENT)).select('id').unique()


def _qtl_reports(credible_set: pl.LazyFrame, study: pl.LazyFrame) -> pl.LazyFrame:
    """The reports of the QTL sets of a study with a `geneId`: one per set, on that gene.

    The evidence is the credible set itself, so two sets of one study are two reports.

    Examples:
        >>> credible_set = pl.LazyFrame({'id': ['a', 'b'], 'studyId': ['q', 'q'], 'leadVariantId': ['1_1_A_T'] * 2})
        >>> study = pl.LazyFrame({'studyId': ['q'], 'studyType': ['eqtl'], 'geneId': ['ENSG1']})
        >>> _qtl_reports(credible_set, study).sort('id').collect().rows()
        [('a', '1_1_A_T', 'ENSG1', 'a'), ('b', '1_1_A_T', 'ENSG1', 'b')]
    """
    return (
        credible_set
        .select('id', 'studyId', 'leadVariantId')
        .join(qtl_genes(study), on='studyId')
        .select('id', 'leadVariantId', pl.col('geneId').alias('trait'), pl.col('id').alias('evidence'))
        .select(_REPORT_COLUMNS)
    )


def _independently_reported_ids(report: pl.LazyFrame) -> pl.LazyFrame:
    """The ids of the reports whose lead and trait have `_MIN_INDEPENDENT_REPORTS` or more distinct evidence values.

    Examples:
        >>> report = pl.LazyFrame({
        ...     'id': ['a', 'b', 'c'],
        ...     'leadVariantId': ['1_1_A_T'] * 3,
        ...     'trait': ['ra', 'ra', 'ibd'],
        ...     'evidence': ['p1', 'p2', 'p3'],
        ... })
        >>> _independently_reported_ids(report).sort('id').collect()['id'].to_list()
        ['a', 'b']
    """
    independent_report_count = pl.col('evidence').n_unique().over('leadVariantId', 'trait')
    return report.filter(independent_report_count >= _MIN_INDEPENDENT_REPORTS).select('id')
