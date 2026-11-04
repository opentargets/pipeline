"""Annotate the validated credible sets and variants, and publish them."""

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import chi2

from pts.schemas.study import StudyType
from pts.transformers.credible_set.enums import FinemappingMethod
from pts.transformers.credible_set.flags import CredibleSetConfidenceClasses, CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.variant_id import variant_chromosome, variant_position
from pts.transformers.disease.coalescing import MEASUREMENT
from pts.transformers.study import QualityFlag as StudyQualityFlag
from pts.transformers.utils.dataset import scan_dataset, sink_dataset
from pts.transformers.utils.quality_flags import flag_matching_ids, flag_rows


class AnnotationSource(BaseModel):
    """The validated sets and variants of every source, the study index and the variant index."""

    model_config = ConfigDict(extra='forbid')

    credible_set: str
    credible_set_variant: str
    study: str
    """The study index, for the populations of each study."""
    variant: str
    """The variant index, for the gnomAD allele frequencies."""
    target: str
    """The target index, for the chromosome and TSS of the gene of a QTL study."""
    disease: str
    """The disease index, for the therapeutic areas that mark a term as a measurement."""


class AnnotationDestination(BaseModel):
    """Paths of the published credible sets and variants."""

    model_config = ConfigDict(extra='forbid')

    credible_set: str
    credible_set_variant: str


class AnnotationSettings(BaseModel):
    """Settings of the annotation."""

    model_config = ConfigDict(extra='forbid')

    trans_qtl_threshold: int = Field(default=5_000_000, ge=0)
    """The distance from the gene TSS beyond which a QTL variant is trans."""


def _study_of_set(sets: pl.LazyFrame) -> pl.LazyFrame:
    """The `studyId` of each credible set, keyed by `credibleSetId` for a join to its variants."""
    return sets.select(pl.col('id').alias('credibleSetId'), 'studyId')


def _qtl_genes(studies: pl.LazyFrame) -> pl.LazyFrame:
    """The `geneId` of each QTL study that has one."""
    qtl = (pl.col('studyType') != StudyType.GWAS.value) & pl.col('geneId').is_not_null()
    return studies.filter(qtl).select('studyId', 'geneId')


def qtl_gene_locations(studies: pl.LazyFrame, targets: pl.LazyFrame) -> pl.LazyFrame:
    """The chromosome and TSS of the gene of each QTL study, null for a gene absent from the target index.

    Examples:
        >>> studies = pl.LazyFrame({'studyId': ['q', 'g'], 'studyType': ['eqtl', 'gwas'], 'geneId': ['ENSG1', None]})
        >>> targets = pl.LazyFrame({'id': ['ENSG1'], 'tss': [100], 'genomicLocation': [{'chromosome': '1'}]})
        >>> qtl_gene_locations(studies, targets).collect().rows()
        [('q', '1', 100)]
    """
    genes = targets.select(
        pl.col('id').alias('geneId'),
        pl.col('genomicLocation').struct.field('chromosome').alias('_gene_chromosome'),
        pl.col('tss').alias('_gene_tss'),
    )
    return _qtl_genes(studies).join(genes, on='geneId', how='left').select('studyId', '_gene_chromosome', '_gene_tss')


def is_trans(threshold: int) -> pl.Expr:
    """Whether a variant is trans to its gene: on another chromosome, or more than `threshold` from the TSS.

    It reads `_variant_chromosome`, `_variant_position`, `_gene_chromosome` and `_gene_tss`. A
    comparison with a missing value counts as not trans.

    Examples:
        >>> variants = pl.DataFrame({
        ...     '_variant_chromosome': ['2', '1', '1', '1'],
        ...     '_variant_position': [100, 6_000_101, 5_000_100, 100],
        ...     '_gene_chromosome': ['1', '1', '1', None],
        ...     '_gene_tss': [100, 100, 100, None],
        ... })
        >>> variants.select(is_trans(5_000_000))['_trans'].to_list()
        [True, True, False, False]
    """
    other_chromosome = pl.col('_variant_chromosome') != pl.col('_gene_chromosome')
    far_from_tss = (pl.col('_variant_position') - pl.col('_gene_tss')).abs() > threshold
    return (other_chromosome | far_from_tss).fill_null(False).alias('_trans')


def trans_qtls(
    sets: pl.LazyFrame, variants: pl.LazyFrame, studies: pl.LazyFrame, targets: pl.LazyFrame, threshold: int
) -> pl.LazyFrame:
    """The ids of the QTL sets whose every variant `is_trans` to the gene of their study.

    GWAS sets and sets of a study without a `geneId` are never trans. This is the `isTransQtl`
    rule of the `main` branch.

    Examples:
        >>> sets = pl.LazyFrame({'id': ['far', 'near'], 'studyId': ['s', 's']})
        >>> variants = pl.LazyFrame({
        ...     'credibleSetId': ['far', 'near', 'near'],
        ...     'variantId': ['2_10_A_T', '2_10_A_T', '1_9_A_T'],
        ... })
        >>> studies = pl.LazyFrame({'studyId': ['s'], 'studyType': ['eqtl'], 'geneId': ['g']})
        >>> targets = pl.LazyFrame({'id': ['g'], 'tss': [10], 'genomicLocation': [{'chromosome': '1'}]})
        >>> trans_qtls(sets, variants, studies, targets, 5_000_000).collect()['id'].to_list()
        ['far']
    """
    variant_id = pl.col('variantId')
    variant_locations = variants.select(
        pl.col('credibleSetId').alias('id'),
        variant_chromosome(variant_id).alias('_variant_chromosome'),
        variant_position(variant_id).alias('_variant_position'),
    )
    return (
        variant_locations
        .join(sets.select('id', 'studyId'), on='id')
        .join(qtl_gene_locations(studies, targets), on='studyId')
        .group_by('id')
        .agg(is_trans(threshold).all())
        .filter('_trans')
        .select('id')
    )


EVIDENCE = pl.Struct({
    'cohorts': pl.List(pl.String),
    'pubmedId': pl.String,
    'ldPopulationStructure': pl.List(pl.Struct({'ldPopulation': pl.String, 'relativeSampleSize': pl.Float64})),
    'credibleSetId': pl.String,
})
"""What counts as one independent report of a lead on a trait: a GWAS study by its cohorts,
publication and populations, or a QTL credible set."""


def _evidence(**columns: str) -> pl.Expr:
    """An `EVIDENCE` struct of the given fields, read from the given columns, with the other fields null."""
    fields = (
        (pl.col(columns[name]) if name in columns else pl.lit(None)).cast(dtype).alias(name)
        for name, dtype in EVIDENCE.to_schema().items()
    )
    return pl.struct(*fields).alias('_evidence')


def gwas_traits(studies: pl.LazyFrame, diseases: pl.LazyFrame) -> pl.LazyFrame:
    """The traits a GWAS study replicates on, by the kind of its `diseaseIds` terms, as on the `main` branch.

    - only measurements: one trait per measurement, so one shared measurement is enough;
    - measurements and diseases: one trait, the sorted diseases, the measurements set aside;
    - only diseases: one trait, the sorted diseases.

    A term is a measurement when the disease index files it under the measurement therapeutic area;
    a term absent from the disease index is a disease.

    Examples:
        >>> studies = pl.LazyFrame({
        ...     'studyId': ['m', 'md', 'd', 'q'],
        ...     'studyType': ['gwas', 'gwas', 'gwas', 'eqtl'],
        ...     'diseaseIds': [['bmi', 'height'], ['bmi', 'ibd'], ['ra', 'ibd'], []],
        ... })
        >>> diseases = pl.LazyFrame({'id': ['bmi', 'height'], 'therapeuticAreas': [[MEASUREMENT]] * 2})
        >>> gwas_traits(studies, diseases).sort('studyId', '_trait').collect().rows()
        [('d', ['ibd', 'ra']), ('m', ['bmi']), ('m', ['height']), ('md', ['ibd'])]
    """
    measurements = (
        diseases
        .filter(pl.col('therapeuticAreas').list.contains(MEASUREMENT))
        .select(pl.col('id').alias('_term'), pl.lit(True).alias('_is_measurement'))
        .unique()
    )
    terms = (
        studies
        .filter(pl.col('studyType') == StudyType.GWAS.value)
        .select('studyId', pl.col('diseaseIds').alias('_term'))
        .explode('_term')
        .drop_nulls('_term')
        .unique()
        .join(measurements, on='_term', how='left')
    )
    disease_traits = (
        terms
        .filter(pl.col('_is_measurement').is_null())
        .group_by('studyId')
        .agg(pl.col('_term').sort().alias('_trait'))
    )
    measurement_traits = terms.join(disease_traits, on='studyId', how='anti').select(
        'studyId', pl.concat_list('_term').alias('_trait')
    )
    return pl.concat([disease_traits, measurement_traits])


def replication_reports(sets: pl.LazyFrame, studies: pl.LazyFrame, diseases: pl.LazyFrame) -> pl.LazyFrame:
    """One row per credible set and trait it reports its lead on, with what makes the report independent.

    The columns are `id`, `leadVariantId`, `_trait` and `_evidence`.

    - A GWAS set reports on every trait of `gwas_traits`; its evidence is its study's cohorts,
      publication and populations, so studies sharing all three are one report.
    - A QTL set of a study with a `geneId` reports on that gene; its evidence is the credible set,
      so two sets of one study are two reports.

    Examples:
        >>> sets = pl.LazyFrame({'id': ['a', 'b'], 'studyId': ['g', 'q'], 'leadVariantId': ['1_1_A_T'] * 2})
        >>> studies = pl.LazyFrame({
        ...     'studyId': ['g', 'q'],
        ...     'studyType': ['gwas', 'eqtl'],
        ...     'diseaseIds': [['ra'], []],
        ...     'geneId': [None, 'ENSG1'],
        ...     'pubmedId': ['p', None],
        ...     'cohorts': [None, None],
        ...     'ldPopulationStructure': [None, None],
        ... })
        >>> diseases = pl.LazyFrame({'id': ['ra'], 'therapeuticAreas': [['EFO_area']]})
        >>> reports = replication_reports(sets, studies, diseases).sort('id').collect()
        >>> evidence = pl.col('_evidence').struct.field('pubmedId', 'credibleSetId')
        >>> reports.select('id', '_trait', evidence).rows()
        [('a', ['ra'], 'p', None), ('b', ['ENSG1'], None, 'b')]
    """
    leads = sets.select('id', 'studyId', 'leadVariantId')
    study_evidence = studies.select(
        'studyId', _evidence(cohorts='cohorts', pubmedId='pubmedId', ldPopulationStructure='ldPopulationStructure')
    )
    gwas = (
        leads
        .join(gwas_traits(studies, diseases), on='studyId')
        .join(study_evidence, on='studyId')
        .select('id', 'leadVariantId', '_trait', '_evidence')
    )
    qtl = leads.join(_qtl_genes(studies), on='studyId').select(
        'id', 'leadVariantId', pl.concat_list('geneId').alias('_trait'), _evidence(credibleSetId='id')
    )
    return pl.concat([gwas, qtl])


def replicated_sets(sets: pl.LazyFrame, studies: pl.LazyFrame, diseases: pl.LazyFrame) -> pl.LazyFrame:
    """The ids of the sets whose lead is reported on one of their traits by at least two independent reports.

    The reports and what makes them independent are those of `replication_reports`. This is the
    REPLICATED rule of the `main` branch.

    Examples:
        >>> sets = pl.LazyFrame({
        ...     'id': ['a', 'b', 'c'],
        ...     'studyId': ['s1', 's2', 's3'],
        ...     'leadVariantId': ['1_1_A_T'] * 3,
        ... })
        >>> studies = pl.LazyFrame({
        ...     'studyId': ['s1', 's2', 's3'],
        ...     'studyType': ['gwas'] * 3,
        ...     'diseaseIds': [['ra'], ['ra'], ['ibd']],
        ...     'geneId': [None] * 3,
        ...     'pubmedId': ['p1', 'p2', 'p3'],
        ...     'cohorts': [None] * 3,
        ...     'ldPopulationStructure': [None] * 3,
        ... })
        >>> diseases = pl.LazyFrame({'id': ['ra'], 'therapeuticAreas': [['EFO_area']]})
        >>> replicated_sets(sets, studies, diseases).sort('id').collect()['id'].to_list()
        ['a', 'b']
    """
    reports = replication_reports(sets, studies, diseases)
    independent_reports = pl.col('_evidence').n_unique().over('leadVariantId', '_trait')
    return reports.filter(independent_reports >= 2).select('id').unique()


def assign_confidence(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Set `confidence` from the fine-mapping method and the LD and top hit flags.

    The conditions are tested in order:

    - SuSiE or SuSiE-inf without `OUT_OF_SAMPLE_LD`: 4, `FINEMAPPED_IN_SAMPLE_LD`;
    - SuSiE or SuSiE-inf with `OUT_OF_SAMPLE_LD`: 3, `FINEMAPPED_OUT_OF_SAMPLE_LD`;
    - PICS without `TOP_HIT`: 2, `PICSED_SUMMARY_STATS`;
    - PICS with `TOP_HIT`: 1, `PICSED_TOP_HIT`;
    - any other set: 0, `UNKNOWN`.

    Examples:
        >>> out_of_sample, top_hit = CredibleSetFlag.OUT_OF_SAMPLE_LD.value, CredibleSetFlag.TOP_HIT.value
        >>> sets = pl.LazyFrame({
        ...     'fineMappingMethod': ['SuSiE', 'SuSiE-inf', 'PICS', 'PICS', 'SuSie'],
        ...     'qualityControls': [[], [out_of_sample], [], [top_hit], []],
        ... })
        >>> assign_confidence(sets).collect()['confidence'].to_list()
        [4, 3, 2, 1, 0]
    """
    flags = pl.col('qualityControls')
    susie = pl.col('fineMappingMethod').is_in([method.value for method in FinemappingMethod.susie()])
    pics = pl.col('fineMappingMethod') == FinemappingMethod.PICS.value
    classes = CredibleSetConfidenceClasses
    confidence = (
        pl
        .when(susie & ~flags.list.contains(CredibleSetFlag.OUT_OF_SAMPLE_LD.value))
        .then(classes.FINEMAPPED_IN_SAMPLE_LD.value)
        .when(susie)
        .then(classes.FINEMAPPED_OUT_OF_SAMPLE_LD.value)
        .when(pics & ~flags.list.contains(CredibleSetFlag.TOP_HIT.value))
        .then(classes.PICSED_SUMMARY_STATS.value)
        .when(pics)
        .then(classes.PICSED_TOP_HIT.value)
        .otherwise(classes.UNKNOWN.value)
    )
    return frame.with_columns(confidence.cast(pl.Int8).alias('confidence'))


def gnomad_frequencies(
    variants: pl.LazyFrame, sets: pl.LazyFrame, studies: pl.LazyFrame, variant_index: pl.LazyFrame
) -> pl.LazyFrame:
    """The sample-weighted gnomAD frequency of each (studyId, variantId) of `variants`.

    The frequency over the study populations p of `ldPopulationStructure` is
    `sum(w_p * AF_p) / sum(w_p)`, where `w_p` is `relativeSampleSize` and `AF_p` the `{p}_adj`
    entry of the variant `alleleFrequencies`. A single population gives its own frequency,
    whatever its weight. The frequency is null when the study has no population, or when a
    population has no gnomAD frequency or, among several, no weight.

    Examples:
        >>> variants = pl.LazyFrame({'credibleSetId': ['cs', 'cs'], 'variantId': ['v', 'w']})
        >>> sets = pl.LazyFrame({'id': ['cs'], 'studyId': ['s']})
        >>> studies = pl.LazyFrame({
        ...     'studyId': ['s'],
        ...     'ldPopulationStructure': [[
        ...         {'ldPopulation': 'nfe', 'relativeSampleSize': 0.8},
        ...         {'ldPopulation': 'eas', 'relativeSampleSize': 0.2},
        ...     ]],
        ... })
        >>> index = pl.LazyFrame({
        ...     'variantId': ['v', 'w'],
        ...     'alleleFrequencies': [
        ...         [
        ...             {'populationName': 'nfe_adj', 'alleleFrequency': 0.3},
        ...             {'populationName': 'eas_adj', 'alleleFrequency': 0.05},
        ...         ],
        ...         [{'populationName': 'nfe_adj', 'alleleFrequency': 0.3}],
        ...     ],
        ... })
        >>> frequencies = gnomad_frequencies(variants, sets, studies, index)
        >>> frequencies.with_columns(pl.col('_gnomad').round(4)).sort('variantId').collect().rows()
        [('s', 'v', 0.25), ('s', 'w', None)]
    """
    pairs = (
        variants
        .select('credibleSetId', 'variantId')
        .join(_study_of_set(sets), on='credibleSetId')
        .select('studyId', 'variantId')
        .unique()
    )
    populations = (
        studies
        .select('studyId', pl.col('ldPopulationStructure').alias('_population'))
        .explode('_population', empty_as_null=True)
        .select(
            'studyId',
            (pl.col('_population').struct.field('ldPopulation') + '_adj').alias('populationName'),
            pl.col('_population').struct.field('relativeSampleSize').alias('_weight'),
        )
    )
    frequencies = (
        variant_index
        .select('variantId', 'alleleFrequencies')
        .join(pairs.select('variantId').unique(), on='variantId', how='semi')
        .explode('alleleFrequencies')
        .select('variantId', pl.col('alleleFrequencies').struct.unnest())
    )
    af, weight = pl.col('alleleFrequency'), pl.col('_weight')
    single = pl.len() == 1
    missing = af.is_null().any() | (~single & weight.is_null().any())
    weighted = ((weight * af).sum() / weight.sum()).fill_nan(None)
    return (
        pairs
        .join(populations, on='studyId', how='left')
        .join(frequencies, on=['variantId', 'populationName'], how='left')
        .group_by('studyId', 'variantId')
        .agg(pl.when(missing).then(None).when(single).then(af.first()).otherwise(weighted).alias('_gnomad'))
    )


def effect_allele_frequency(
    variants: pl.LazyFrame, sets: pl.LazyFrame, studies: pl.LazyFrame, variant_index: pl.LazyFrame
) -> pl.LazyFrame:
    """Set `effectAlleleFrequency` from the source, otherwise from gnomAD, and flag its origin.

    A source frequency is valid when it is in [0, 1] and is not NaN. Every row gets exactly one of:

    - `EAF_FROM_SOURCE`: the valid source frequency is kept;
    - `EAF_FROM_GNOMAD`: the sample-weighted gnomAD frequency of `gnomad_frequencies`;
    - `MISSING_EAF`: neither, and the frequency is null.

    An invalid source frequency also gets `INVALID_EAF`. The rows and their order are unchanged.

    Examples:
        >>> variants = pl.LazyFrame(
        ...     {'credibleSetId': ['cs'] * 3, 'variantId': ['v'] * 3, 'effectAlleleFrequency': [0.3, 1.4, None]},
        ...     schema_overrides={'effectAlleleFrequency': pl.Float32},
        ... ).with_columns(pl.lit([], dtype=pl.List(pl.String)).alias('qualityControls'))
        >>> sets = pl.LazyFrame({'id': ['cs'], 'studyId': ['s']})
        >>> nfe = {'ldPopulation': 'nfe', 'relativeSampleSize': 1.0}
        >>> studies = pl.LazyFrame({'studyId': ['s'], 'ldPopulationStructure': [[nfe]]})
        >>> nfe_adj = {'populationName': 'nfe_adj', 'alleleFrequency': 0.2}
        >>> index = pl.LazyFrame({'variantId': ['v'], 'alleleFrequencies': [[nfe_adj]]})
        >>> result = effect_allele_frequency(variants, sets, studies, index).collect()
        >>> eafs = pl.col('effectAlleleFrequency').cast(pl.Float64).round(2)
        >>> for eaf, flags in result.select(eafs, 'qualityControls').rows():
        ...     print(eaf, [CredibleSetVariantFlag(flag).name for flag in flags])
        0.3 ['EAF_FROM_SOURCE']
        0.2 ['INVALID_EAF', 'EAF_FROM_GNOMAD']
        0.2 ['EAF_FROM_GNOMAD']
    """
    flags = CredibleSetVariantFlag
    eaf = pl.col('effectAlleleFrequency')
    valid = (eaf.is_between(0, 1) & eaf.is_not_nan()).fill_null(False)
    gnomad = gnomad_frequencies(variants.filter(~valid), sets, studies, variant_index)
    origin = (
        pl
        .when(valid)
        .then(pl.lit(flags.EAF_FROM_SOURCE.value))
        .when(pl.col('_gnomad').is_not_null())
        .then(pl.lit(flags.EAF_FROM_GNOMAD.value))
        .otherwise(pl.lit(flags.MISSING_EAF.value))
    )
    invalid = pl.when(eaf.is_not_null() & ~valid).then(pl.lit(flags.INVALID_EAF.value))
    return (
        variants
        .join(
            _study_of_set(sets),
            on='credibleSetId',
            how='left',
            maintain_order='left',
        )
        .join(gnomad, on=['studyId', 'variantId'], how='left', maintain_order='left')
        .with_columns(
            pl.when(valid).then(eaf).otherwise(pl.col('_gnomad')).cast(pl.Float32).alias('effectAlleleFrequency'),
            pl.concat_list('qualityControls', invalid, origin).list.drop_nulls().alias('qualityControls'),
        )
        .select(variants.collect_schema().names())
    )


CHI2_EXACT_MIN_EXPONENT = -300
"""The smallest p-value exponent with an exact chi-squared statistic; below it the p-value underflows."""
CHI2_APPROXIMATION_SLOPE = 4.596
"""The slope of the linear approximation of the chi-squared statistic from -log10(p), a constant."""
CHI2_APPROXIMATION_INTERCEPT = -5.367
"""The intercept of the linear approximation of the chi-squared statistic from -log10(p), a constant."""


def chi2_from_pvalue(mantissa: pl.Expr, exponent: pl.Expr) -> pl.Expr:
    """The chi-squared statistic (1 degree of freedom) of a p-value given as mantissa and exponent.

    From `CHI2_EXACT_MIN_EXPONENT` up it is the inverse survival function of the chi-squared
    distribution. Below it, where the p-value underflows, it is the linear approximation
    `CHI2_APPROXIMATION_SLOPE * -log10(p) + CHI2_APPROXIMATION_INTERCEPT`, whose slope 4.596 and
    intercept -5.367 are constants. It is null when the p-value is missing or invalid.

    Examples:
        >>> frame = pl.DataFrame({'mantissa': [5.0, 9.0, 9.0, None], 'exponent': [-8, -300, -301, -8]})
        >>> frame.select(chi2_from_pvalue(pl.col('mantissa'), pl.col('exponent')).round(2))['chi2'].to_list()
        [29.72, 1369.48, 1373.64, None]
    """
    neglog = -(mantissa.cast(pl.Float64).log10() + exponent)
    approximation = neglog * CHI2_APPROXIMATION_SLOPE + CHI2_APPROXIMATION_INTERCEPT
    p_value = mantissa.cast(pl.Float64) * pl.lit(10.0).pow(exponent)
    exact = p_value.map_batches(
        lambda values: pl.Series(chi2.isf(values.to_numpy(), df=1)), return_dtype=pl.Float64, is_elementwise=True
    )
    statistic = pl.when(exponent < CHI2_EXACT_MIN_EXPONENT).then(approximation).otherwise(exact).fill_nan(None)
    return pl.when(statistic.is_finite()).then(statistic).alias('chi2')


def rescaled_effect_size(variants: pl.LazyFrame, sets: pl.LazyFrame, studies: pl.LazyFrame) -> pl.LazyFrame:
    """Set the rescaled effect size of every variant from its p-value, frequency and study design.

    `|z|` is the square root of `chi2_from_pvalue`, `varG = 2 * AF * (1 - AF)` and N is the study
    `nSamples`. The trait class comes from the study design flag:

    - MEASUREMENT_STUDY_DESIGN, quantitative: `SE = sqrt(1 / (varG * N))`;
    - CASE_CONTROL_STUDY_DESIGN, binary, with the prevalence `nCases / nSamples`:
      `SE = sqrt(1 / (varG * N * prev * (1 - prev)))`.

    `rescaledAbsBeta = |z| * SE`, `rescaledStandardError = SE`, and `rescaledDirection` is the sign
    of `conditionalBeta`, otherwise of `reportedBeta`, relative to the effect allele. A rescaled row
    gets RESCALED_FROM_CONDITIONAL or RESCALED_FROM_MARGINAL after the origin of its direction. The
    rescaled columns are null, and no flag is set, without a p-value, a beta, a frequency in (0, 1),
    a supported design or a sample size. The rows and their order are unchanged.

    Examples:
        >>> variants = pl.LazyFrame({
        ...     'credibleSetId': ['cs', 'cs'], 'pValueMantissa': [1.0, 1.0], 'pValueExponent': [-10, -10],
        ...     'effectAlleleFrequency': [0.2, None], 'reportedBeta': [0.2, 0.2], 'conditionalBeta': [None, None],
        ...     'rescaledAbsBeta': [None, None],
        ...     'rescaledStandardError': [None, None],
        ...     'rescaledDirection': [None, None],
        ...     'qualityControls': [[], []],
        ... })
        >>> sets = pl.LazyFrame({'id': ['cs'], 'studyId': ['s']})
        >>> studies = pl.LazyFrame({
        ...     'studyId': ['s'], 'nSamples': [100_000], 'nCases': [None],
        ...     'qualityControls': [[StudyQualityFlag.MEASUREMENT_STUDY_DESIGN.value]],
        ... })
        >>> rescaled = rescaled_effect_size(variants, sets, studies).collect()
        >>> rounded = pl.col('rescaledAbsBeta', 'rescaledStandardError').cast(pl.Float64).round(5)
        >>> rescaled.select(rounded, 'rescaledDirection').rows()
        [(0.03615, 0.00559, 1), (None, None, None)]
    """
    study_flags = pl.col('qualityControls')
    designs = studies.select(
        'studyId',
        pl.col('nSamples').cast(pl.Float64).alias('_n'),
        (pl.col('nCases').cast(pl.Float64) / pl.col('nSamples').cast(pl.Float64)).alias('_prevalence'),
        study_flags.list.contains(StudyQualityFlag.MEASUREMENT_STUDY_DESIGN.value).alias('_quantitative'),
        study_flags.list.contains(StudyQualityFlag.CASE_CONTROL_STUDY_DESIGN.value).alias('_binary'),
    )
    af = pl.col('effectAlleleFrequency').cast(pl.Float64)
    var_g = 2 * af * (1 - af)
    n, prevalence = pl.col('_n'), pl.col('_prevalence')
    standard_error = (
        pl
        .when(pl.col('_quantitative'))
        .then((1 / (var_g * n)).sqrt())
        .when(pl.col('_binary'))
        .then((1 / (var_g * n * prevalence * (1 - prevalence))).sqrt())
    )
    beta = pl.coalesce(pl.col('conditionalBeta').cast(pl.Float64), pl.col('reportedBeta').cast(pl.Float64))
    z = chi2_from_pvalue(pl.col('pValueMantissa'), pl.col('pValueExponent')).sqrt()
    rescalable = (
        af.is_between(0, 1, closed='none') & (n > 0) & beta.is_not_null() & z.is_not_null() & standard_error.is_finite()
    ).fill_null(False)
    conditional = pl.col('conditionalBeta').is_not_null()
    return (
        variants
        .join(
            _study_of_set(sets),
            on='credibleSetId',
            how='left',
            maintain_order='left',
        )
        .join(designs, on='studyId', how='left', maintain_order='left')
        .with_columns(
            pl.when(rescalable).then(z * standard_error).cast(pl.Float32).alias('rescaledAbsBeta'),
            pl.when(rescalable).then(standard_error).cast(pl.Float32).alias('rescaledStandardError'),
            pl.when(rescalable).then(beta.sign()).cast(pl.Int8).alias('rescaledDirection'),
            rescalable.alias('_rescaled'),
        )
        .pipe(flag_rows, pl.col('_rescaled') & conditional, CredibleSetVariantFlag.RESCALED_FROM_CONDITIONAL)
        .pipe(flag_rows, pl.col('_rescaled') & ~conditional, CredibleSetVariantFlag.RESCALED_FROM_MARGINAL)
        .select(variants.collect_schema().names())
    )


def annotation(
    source: dict[str, str | list[str]],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Annotate the validated credible sets and write the published datasets.

    Args:
        source: The validated sets under `credible_set`, the validated variants under
            `credible_set_variant`, and the study, variant, target and disease indexes under `study`,
            `variant`, `target` and `disease`.
        destination: Published paths under `credible_set` and `credible_set_variant`.
        settings: The distance from the gene TSS beyond which a QTL variant is trans, under
            `trans_qtl_threshold`.
        config: Pipeline configuration.
    """
    sources = AnnotationSource.model_validate(source)
    destinations = AnnotationDestination.model_validate(destination)
    options = AnnotationSettings.model_validate(settings)
    validated_sets = scan_dataset(sources.credible_set)
    validated_variants = scan_dataset(sources.credible_set_variant)
    studies = scan_dataset(sources.study)
    targets = scan_dataset(sources.target)
    diseases = scan_dataset(sources.disease).select('id', 'therapeuticAreas')
    trans = trans_qtls(validated_sets, validated_variants, studies, targets, options.trans_qtl_threshold)
    replicated = replicated_sets(validated_sets, studies, diseases)
    sets = (
        validated_sets
        .pipe(assign_confidence)
        .pipe(flag_matching_ids, 'id', trans, CredibleSetFlag.TRANS_QTL)
        .pipe(flag_matching_ids, 'id', replicated, CredibleSetFlag.REPLICATED)
    )
    variants = validated_variants.pipe(
        effect_allele_frequency, validated_sets, studies, scan_dataset(sources.variant)
    ).pipe(rescaled_effect_size, validated_sets, studies)
    logger.info(
        'Writing credible sets to {} and variants to {}', destinations.credible_set, destinations.credible_set_variant
    )
    pl.collect_all([
        sink_dataset(sets, destinations.credible_set),
        sink_dataset(variants, destinations.credible_set_variant),
    ])
