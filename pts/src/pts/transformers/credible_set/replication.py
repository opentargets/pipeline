"""The REPLICATED rule: credible sets whose lead is reported on one of their traits by two independent reports."""

from __future__ import annotations

import polars as pl

from pts.schemas.study import StudyType
from pts.transformers.credible_set.study_lookup import qtl_genes
from pts.transformers.disease.coalescing import MEASUREMENT

EVIDENCE = pl.Struct({
    'cohorts': pl.List(pl.String),
    'pubmedId': pl.String,
    'ldPopulationStructure': pl.List(pl.Struct({'ldPopulation': pl.String, 'relativeSampleSize': pl.Float64})),
    'credibleSetId': pl.String,
})
"""What counts as one independent report of a lead on a trait: a GWAS study by its cohorts,
publication and populations, or a QTL credible set."""


def _evidence(**column_of_field: str) -> pl.Expr:
    """An `EVIDENCE` struct of the given fields, read from the given columns, with the other fields null."""
    struct_field = (
        (pl.col(column_of_field[name]) if name in column_of_field else pl.lit(None)).cast(dtype).alias(name)
        for name, dtype in EVIDENCE.to_schema().items()
    )
    return pl.struct(*struct_field).alias('_evidence')


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


def replication_reports(credible_set: pl.LazyFrame, study: pl.LazyFrame, disease: pl.LazyFrame) -> pl.LazyFrame:
    """One row per credible set and trait it reports its lead on, with what makes the report independent.

    The columns are `id`, `leadVariantId`, `_trait` and `_evidence`.

    - A GWAS set reports on every trait of `gwas_traits`; its evidence is its study's cohorts,
      publication and populations, so studies sharing all three are one report.
    - A QTL set of a study with a `geneId` reports on that gene; its evidence is the credible set,
      so two sets of one study are two reports.

    Examples:
        >>> credible_set = pl.LazyFrame({'id': ['a', 'b'], 'studyId': ['g', 'q'], 'leadVariantId': ['1_1_A_T'] * 2})
        >>> study = pl.LazyFrame({
        ...     'studyId': ['g', 'q'],
        ...     'studyType': ['gwas', 'eqtl'],
        ...     'diseaseIds': [['ra'], []],
        ...     'geneId': [None, 'ENSG1'],
        ...     'pubmedId': ['p', None],
        ...     'cohorts': [None, None],
        ...     'ldPopulationStructure': [None, None],
        ... })
        >>> disease = pl.LazyFrame({'id': ['ra'], 'therapeuticAreas': [['EFO_area']]})
        >>> report = replication_reports(credible_set, study, disease).sort('id').collect()
        >>> evidence = pl.col('_evidence').struct.field('pubmedId', 'credibleSetId')
        >>> report.select('id', '_trait', evidence).rows()
        [('a', ['ra'], 'p', None), ('b', ['ENSG1'], None, 'b')]
    """
    lead = credible_set.select('id', 'studyId', 'leadVariantId')
    study_evidence = study.select(
        'studyId', _evidence(cohorts='cohorts', pubmedId='pubmedId', ldPopulationStructure='ldPopulationStructure')
    )
    gwas = (
        lead
        .join(gwas_traits(study, disease), on='studyId')
        .join(study_evidence, on='studyId')
        .select('id', 'leadVariantId', '_trait', '_evidence')
    )
    qtl = lead.join(qtl_genes(study), on='studyId').select(
        'id', 'leadVariantId', pl.concat_list('geneId').alias('_trait'), _evidence(credibleSetId='id')
    )
    return pl.concat([gwas, qtl])


def replicated_sets(credible_set: pl.LazyFrame, study: pl.LazyFrame, disease: pl.LazyFrame) -> pl.LazyFrame:
    """The ids of the sets whose lead is reported on one of their traits by at least two independent reports.

    The reports and what makes them independent are those of `replication_reports`. This is the
    REPLICATED rule of the `main` branch.

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
    report = replication_reports(credible_set, study, disease)
    independent_report_count = pl.col('_evidence').n_unique().over('leadVariantId', '_trait')
    return report.filter(independent_report_count >= 2).select('id').unique()
