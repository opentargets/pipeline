"""VIDRA step 1: the analysis-ready variant table, one row per variant, gene and disease.

Port of VIDRA's `prepare_analysis_input.py` onto the platform's own datasets. Four sources feed the
table (`GsourceLab`, see `vidra_utils.constants`):

- common variants (0): a GWAS credible set colocalising (COLOC H4 above the threshold) with an
  eQTL or pQTL credible set of the gene. The GWAS lead is the row's variant; `yc` is its rescaled
  effect in the GWAS, `xc` the QTL lead's reported effect in the QTL study (the two leads are in
  LD but need not be the same variant);
- rare variants (1): AstraZeneca ExWAS associations, `yc = log(OR)`;
- ClinVar variants (2): EVA evidence with a clinical significance score and no effect size;
- coding GWAS variants (3): genome-wide significant GWAS leads with a coding consequence on the
  gene, not in a colocalising set.

Every row gets the gene-disease burden test of the AstraZeneca collapsing analysis (`bO`, `bOse`),
and its protein-function annotations from the variant index (`vidra_utils.annotation`). Diseases
are platform EFO ids: GWAS studies through `study.diseaseIds`, ClinVar through `diseaseId`,
AstraZeneca phenotypes through the OnToma label lookup table, with obsolete ids mapped to their
current terms through the disease index.
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.common.session import Session
from pts.pyspark.vidra_utils.annotation import (
    CLASS_INFRAME_OR_SPLICE,
    CLASS_LOF,
    CLASS_MISSENSE,
    CLASS_OTHER,
    CODING_CONSEQUENCES,
    INFRAME_OR_SPLICE_CONSEQUENCES,
    LOF_CONSEQUENCES,
    LOFTEE_HIGH_CONFIDENCE,
    MISSENSE_CONSEQUENCE,
)
from pts.pyspark.vidra_utils.constants import (
    ANALYSIS_COLUMNS,
    ANNOTATION_COLUMNS,
    CLINICAL_SIGNIFICANCE_SCORE,
    DEFAULT_BO,
    DEFAULT_BOSE,
    DEFAULT_CLINICAL_SIGNIFICANCE,
    DEFAULT_XC,
    DEFAULT_XCSE,
    DEFAULT_YC,
    DEFAULT_YCSE,
    EVA_EXCLUDED_CONFIDENCE,
    PROVENANCE_COLUMNS,
    QTL_NONE,
    SOURCE_AZ_RARE,
    SOURCE_CLINVAR,
    SOURCE_CODING_GWAS,
    SOURCE_COMMON_QTL,
)
from pts.transformers.evidence.utils.ontology import MAPPED_ID_COLUMN, add_efo_mapping

CI_TO_SE = 3.92
"""Width of a 95% confidence interval in standard errors."""


class GpnStarTransform(BaseModel):
    """`as_gpnstar = 1 - sigmoid(a + b * llr)`; see `vidra_utils.gpn_star`."""

    model_config = ConfigDict(extra='forbid')

    a: float
    b: float


class VidraInputSettings(BaseModel):
    """Thresholds and vocabularies of the step."""

    model_config = ConfigDict(extra='forbid')

    coloc_h4_threshold: float = Field(default=0.7, ge=0, le=1)
    """A GWAS and a QTL credible set colocalise when COLOC's H4 exceeds this."""
    gwas_p_value_threshold: float = Field(default=5e-8, gt=0, le=1)
    """Coding GWAS leads are kept at or below this p-value."""
    qtl_study_types: dict[str, int] = Field(default_factory=lambda: {'eqtl': 0, 'sceqtl': 0, 'pqtl': 1})
    """Study types of the QTL side of a colocalisation, each with its `GqtlLab` (0 eQTL, 1 pQTL)."""
    blood_biosample_ancestors: list[str] = Field(
        default_factory=lambda: ['UBERON_0000178', 'UBERON_0001969', 'CL_0000988']
    )
    """QTL studies are kept when their biosample is one of these or descends from one; empty keeps every tissue."""
    burden_collapsing_models: list[str] = Field(default_factory=lambda: ['ptv', 'ptv5pcnt', 'ptvraredmg'])
    """AstraZeneca collapsing models whose tests provide the gene-disease burden effect."""
    gpn_star: GpnStarTransform


# ----------------------------------------------------------------------------------------------
# Reference tables
# ----------------------------------------------------------------------------------------------


def protein_coding_genes(target: DataFrame) -> DataFrame:
    """`geneId`, `approvedSymbol` of the protein-coding targets.

    Examples:
        >>> rows = [('g1', 'A', 'protein_coding'), ('g2', 'B', 'lncRNA')]
        >>> target = spark.createDataFrame(rows, ['id', 'approvedSymbol', 'biotype'])
        >>> protein_coding_genes(target).collect()
        [Row(geneId='g1', approvedSymbol='A')]
    """
    return (
        target
        .filter(f.col('biotype') == 'protein_coding')
        .select(f.col('id').alias('geneId'), 'approvedSymbol')
        .filter(f.col('geneId').isNotNull())
        .distinct()
    )


def obsolete_disease_map(disease: DataFrame) -> DataFrame:
    """`obsoleteId` to the current `diseaseId`, from the disease index's `obsoleteTerms`.

    Examples:
        >>> disease = spark.createDataFrame([('EFO_1', ['EFO_0', 'EFO_9']), ('EFO_2', None)], ['id', 'obsoleteTerms'])
        >>> sorted(obsolete_disease_map(disease).collect())
        [Row(obsoleteId='EFO_0', diseaseId='EFO_1'), Row(obsoleteId='EFO_9', diseaseId='EFO_1')]
    """
    return disease.select(
        f.explode('obsoleteTerms').alias('obsoleteId'), f.col('id').alias('diseaseId')
    ).dropDuplicates(['obsoleteId'])


def remap_diseases(frame: DataFrame, column: str, obsolete: DataFrame) -> DataFrame:
    """Replace obsolete disease ids in `column` with their current id.

    Examples:
        >>> obsolete = spark.createDataFrame([('EFO_0', 'EFO_1')], ['obsoleteId', 'diseaseId'])
        >>> frame = spark.createDataFrame([('a', 'EFO_0'), ('b', 'EFO_5')], ['k', 'd'])
        >>> sorted(remap_diseases(frame, 'd', obsolete).collect())
        [Row(k='a', d='EFO_1'), Row(k='b', d='EFO_5')]
    """
    current = obsolete.select(f.col('obsoleteId').alias('_obsolete'), f.col('diseaseId').alias('_current'))
    return (
        frame
        .join(current, frame[column] == current['_obsolete'], 'left')
        .withColumn(column, f.coalesce(f.col('_current'), f.col(column)))
        .drop('_obsolete', '_current')
    )


def study_diseases(study: DataFrame, obsolete: DataFrame) -> DataFrame:
    """`studyId`, `diseaseId`: one row per disease of each GWAS study, obsolete ids remapped.

    Examples:
        >>> rows = [('s1', 'gwas', ['EFO_0', 'EFO_2']), ('s2', 'eqtl', ['EFO_2'])]
        >>> study = spark.createDataFrame(rows, ['studyId', 'studyType', 'diseaseIds'])
        >>> obsolete = spark.createDataFrame([('EFO_0', 'EFO_1')], ['obsoleteId', 'diseaseId'])
        >>> sorted(study_diseases(study, obsolete).collect())
        [Row(studyId='s1', diseaseId='EFO_1'), Row(studyId='s1', diseaseId='EFO_2')]
    """
    exploded = (
        study
        .filter(f.col('studyType') == 'gwas')
        .select('studyId', f.explode('diseaseIds').alias('diseaseId'))
        .filter(f.col('diseaseId').isNotNull())
    )
    return remap_diseases(exploded, 'diseaseId', obsolete).distinct()


def log10_p_value(mantissa: str = 'pValueMantissa', exponent: str = 'pValueExponent') -> f.Column:
    """`log10(p)` from mantissa and exponent, which orders p-values that underflow a double."""
    return f.log10(f.col(mantissa).cast('double')) + f.col(exponent).cast('double')


def lead_statistics(credible_set: DataFrame, credible_set_variant: DataFrame) -> DataFrame:
    """The lead variant of every credible set with its statistics in that set.

    Columns: `credibleSetId`, `studyId`, `studyType`, `leadVariantId`, `log10PValue`, `reportedBeta`,
    `reportedStandardError`, `rescaledBeta` (signed) and `rescaledStandardError`.

    Examples:
        >>> sets = spark.createDataFrame([('cs1', 's1', 'gwas', 'v1')], ['id', 'studyId', 'studyType', 'leadVariantId'])
        >>> columns = [
        ...     'credibleSetId', 'variantId', 'pValueMantissa', 'pValueExponent', 'reportedBeta',
        ...     'reportedStandardError', 'rescaledAbsBeta', 'rescaledStandardError', 'rescaledDirection',
        ... ]
        >>> lead = ('cs1', 'v1', 2.0, -10, 0.3, 0.05, 0.25, 0.04, -1)
        >>> rows = [lead, ('cs1', 'v2', 1.0, -8, None, None, None, None, None)]
        >>> variants = spark.createDataFrame(rows, columns)
        >>> row = lead_statistics(sets, variants).collect()[0]
        >>> (row.leadVariantId, round(row.log10PValue, 3), row.rescaledBeta)
        ('v1', -9.699, -0.25)
    """
    leads = credible_set.select(
        f.col('id').alias('credibleSetId'),
        'studyId',
        f.col('studyType').cast('string').alias('studyType'),
        'leadVariantId',
    )
    statistics = credible_set_variant.select(
        'credibleSetId',
        f.col('variantId').alias('leadVariantId'),
        log10_p_value().alias('log10PValue'),
        f.col('reportedBeta').cast('double').alias('reportedBeta'),
        f.col('reportedStandardError').cast('double').alias('reportedStandardError'),
        (f.col('rescaledDirection').cast('double') * f.col('rescaledAbsBeta').cast('double')).alias('rescaledBeta'),
        f.col('rescaledStandardError').cast('double').alias('rescaledStandardError'),
    )
    return leads.join(statistics, on=['credibleSetId', 'leadVariantId'], how='left')


# ----------------------------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------------------------


def blood_biosamples(biosample: DataFrame, ancestors: list[str]) -> DataFrame:
    """`biosampleId` of the biosamples that are, or descend from, one of `ancestors`.

    Examples:
        >>> blood, monocyte = ('UBERON_0000178', []), ('CL_0000576', ['CL_0000988'])
        >>> rows = [blood, monocyte, ('UBERON_0002107', ['UBERON_0000062'])]
        >>> biosample = spark.createDataFrame(rows, ['biosampleId', 'ancestors'])
        >>> sorted(r.biosampleId for r in blood_biosamples(biosample, ['UBERON_0000178', 'CL_0000988']).collect())
        ['CL_0000576', 'UBERON_0000178']
    """
    wanted = f.array(*[f.lit(a) for a in ancestors])
    return (
        biosample
        .filter(
            f.col('biosampleId').isin(ancestors) | f.arrays_overlap(f.coalesce(f.col('ancestors'), f.array()), wanted)
        )
        .select('biosampleId')
        .distinct()
    )


def qtl_credible_sets(
    leads: DataFrame,
    study: DataFrame,
    genes: DataFrame,
    biosamples: DataFrame | None,
    settings: VidraInputSettings,
) -> DataFrame:
    """QTL credible sets of protein-coding genes with their lead's effect in the QTL study.

    Columns: `qtlCredibleSetId`, `qtlStudyId`, `qtlVariantId`, `as_gene`, `qtlBiosampleId`, `GqtlLab`,
    `xc`, `xcse`, `qtlLog10PValue`. Sets whose lead has no reported effect are dropped; with
    `biosamples`, sets of other tissues are dropped too.
    """
    qtl_label = f.create_map(*[f.lit(x) for pair in settings.qtl_study_types.items() for x in pair])
    qtl_studies = (
        study
        .filter(f.col('studyType').isin(list(settings.qtl_study_types)))
        .select(
            f.col('studyId').alias('qtlStudyId'),
            f.col('geneId').alias('as_gene'),
            f.col('biosampleId').alias('qtlBiosampleId'),
        )
        .join(genes.select(f.col('geneId').alias('as_gene')), on='as_gene', how='inner')
    )
    if biosamples is not None:
        qtl_studies = qtl_studies.join(
            biosamples.select(f.col('biosampleId').alias('qtlBiosampleId')), on='qtlBiosampleId'
        )
    return (
        leads
        .filter(f.col('studyType').isin(list(settings.qtl_study_types)))
        .filter(f.col('reportedBeta').isNotNull() & f.col('reportedStandardError').isNotNull())
        .select(
            f.col('credibleSetId').alias('qtlCredibleSetId'),
            f.col('studyId').alias('qtlStudyId'),
            f.col('leadVariantId').alias('qtlVariantId'),
            qtl_label[f.col('studyType')].cast('int').alias('GqtlLab'),
            f.col('reportedBeta').alias('xc'),
            f.col('reportedStandardError').alias('xcse'),
            f.col('log10PValue').alias('qtlLog10PValue'),
        )
        .join(qtl_studies, on='qtlStudyId', how='inner')
    )


def gwas_leads(leads: DataFrame) -> DataFrame:
    """GWAS credible sets whose lead has a rescaled effect.

    Columns: `gwasCredibleSetId`, `gwasStudyId`, `variant`, `yc`, `ycse`, `gwasLog10PValue`.
    """
    return (
        leads
        .filter(f.col('studyType') == 'gwas')
        .filter(f.col('rescaledBeta').isNotNull() & f.col('rescaledStandardError').isNotNull())
        .select(
            f.col('credibleSetId').alias('gwasCredibleSetId'),
            f.col('studyId').alias('gwasStudyId'),
            f.col('leadVariantId').alias('variant'),
            f.col('rescaledBeta').alias('yc'),
            f.col('rescaledStandardError').alias('ycse'),
            f.col('log10PValue').alias('gwasLog10PValue'),
        )
    )


def _empty_provenance() -> list[f.Column]:
    return [f.lit(None).cast('string').alias(c) for c in PROVENANCE_COLUMNS]


def common_variants(
    colocalisation: DataFrame,
    gwas: DataFrame,
    qtl: DataFrame,
    diseases: DataFrame,
    settings: VidraInputSettings,
) -> DataFrame:
    """Common variant rows (`GsourceLab` 0): GWAS leads colocalising with a QTL set of the gene.

    Either side of a colocalisation may be the GWAS set. One row per variant, gene and disease:
    the colocalisation with the lowest GWAS p-value, then the lowest QTL p-value, wins.
    """
    pairs = colocalisation.filter(f.col('h4') > settings.coloc_h4_threshold).select(
        'leftLocusId', 'rightLocusId', 'h4', 'betaRatioSignAverage'
    )
    oriented = pairs.select(
        f.col('leftLocusId').alias('gwasCredibleSetId'),
        f.col('rightLocusId').alias('qtlCredibleSetId'),
        'h4',
        'betaRatioSignAverage',
    ).unionByName(
        pairs.select(
            f.col('rightLocusId').alias('gwasCredibleSetId'),
            f.col('leftLocusId').alias('qtlCredibleSetId'),
            'h4',
            'betaRatioSignAverage',
        )
    )
    joined = (
        oriented
        .join(gwas, on='gwasCredibleSetId', how='inner')
        .join(qtl, on='qtlCredibleSetId', how='inner')
        .join(
            diseases.select(f.col('studyId').alias('gwasStudyId'), f.col('diseaseId').alias('as_disease')),
            on='gwasStudyId',
        )
    )
    # One row per QTL type: an eQTL and a pQTL colocalisation of the same lead are two observations
    # (the Bayesian step deduplicates on the same key).
    best = Window.partitionBy('variant', 'as_gene', 'as_disease', 'GqtlLab').orderBy(
        f.col('gwasLog10PValue').asc_nulls_last(),
        f.col('qtlLog10PValue').asc_nulls_last(),
        'gwasCredibleSetId',
        'qtlCredibleSetId',
    )
    return (
        joined
        .withColumn('_rank', f.row_number().over(best))
        .filter(f.col('_rank') == 1)
        .select(
            'variant',
            'as_gene',
            'as_disease',
            f.lit(SOURCE_COMMON_QTL).alias('GsourceLab'),
            'GqtlLab',
            'yc',
            'ycse',
            'xc',
            'xcse',
            f.lit(DEFAULT_CLINICAL_SIGNIFICANCE).alias('as_clinicalSignificance'),
            'gwasStudyId',
            'gwasCredibleSetId',
            'qtlStudyId',
            'qtlCredibleSetId',
            'qtlVariantId',
            'qtlBiosampleId',
            f.col('h4').cast('string').alias('colocH4'),
            f.col('betaRatioSignAverage').cast('string').alias('colocBetaRatioSignAverage'),
            f.lit(None).cast('string').alias('clinvarDirectionOnTarget'),
            f.lit(None).cast('string').alias('clinvarDirectionOnTrait'),
        )
    )


def gene_consequences(variant: DataFrame, genes: DataFrame) -> DataFrame:
    """The consequence of each indexed variant on the canonical transcript of each protein-coding gene.

    Columns: `variant`, `as_gene`, `consequences` (SO ids), `is_lof` (a LoF term, or LOFTEE high
    confidence) and `consequence_class` (see `vidra_utils.annotation`). The Ensembl canonical
    transcript is preferred, then a MANE Select one, then the highest consequence score.
    """
    exploded = (
        variant
        .select(f.col('variantId').alias('variant'), f.explode('transcriptConsequences').alias('tc'))
        .select(
            'variant',
            f.col('tc.targetId').alias('as_gene'),
            f.col('tc.variantFunctionalConsequenceIds').alias('consequences'),
            f.col('tc.lofteePrediction').alias('loftee'),
            f.coalesce(f.col('tc.isEnsemblCanonical'), f.lit(False)).alias('canonical'),
            f.col('tc.maneSelect').isNotNull().alias('mane'),
            f.coalesce(f.col('tc.consequenceScore'), f.lit(0.0)).alias('score'),
            f.coalesce(f.col('tc.biotype') == 'protein_coding', f.lit(False)).alias('proteinCoding'),
        )
        .filter(f.col('as_gene').isNotNull() & f.col('proteinCoding'))
        .join(genes.select(f.col('geneId').alias('as_gene')), on='as_gene', how='inner')
    )
    preferred = Window.partitionBy('variant', 'as_gene').orderBy(
        f.col('canonical').desc(), f.col('mane').desc(), f.col('score').desc()
    )
    terms = f.coalesce(f.col('consequences'), f.array().cast('array<string>'))
    lof = f.array(*[f.lit(x) for x in sorted(LOF_CONSEQUENCES)])
    inframe = f.array(*[f.lit(x) for x in sorted(INFRAME_OR_SPLICE_CONSEQUENCES)])
    is_lof = (f.col('loftee') == LOFTEE_HIGH_CONFIDENCE) | f.arrays_overlap(terms, lof)
    return (
        exploded
        .withColumn('_rank', f.row_number().over(preferred))
        .filter(f.col('_rank') == 1)
        .select(
            'variant',
            'as_gene',
            terms.alias('consequences'),
            f.coalesce(is_lof, f.lit(False)).alias('is_lof'),
            f
            .when(f.coalesce(is_lof, f.lit(False)), CLASS_LOF)
            .when(f.array_contains(terms, MISSENSE_CONSEQUENCE), CLASS_MISSENSE)
            .when(f.arrays_overlap(terms, inframe), CLASS_INFRAME_OR_SPLICE)
            .otherwise(CLASS_OTHER)
            .alias('consequence_class'),
        )
    )


def coding_gwas_variants(
    gwas: DataFrame,
    consequences: DataFrame,
    diseases: DataFrame,
    common: DataFrame,
    settings: VidraInputSettings,
) -> DataFrame:
    """Coding GWAS rows (`GsourceLab` 3): significant GWAS leads coding for a gene, not in a colocalising set.

    A lead is coding for a gene when its consequence on the gene's transcript is one of
    `CODING_CONSEQUENCES`. Variants of any common row are excluded whatever the gene or disease,
    as in the original pipeline. One row per variant, gene and disease, lowest GWAS p-value.
    """
    coding = f.array(*[f.lit(x) for x in sorted(CODING_CONSEQUENCES)])
    coding_genes = consequences.filter(f.arrays_overlap(f.col('consequences'), coding)).select('variant', 'as_gene')
    joined = (
        gwas
        .filter(f.col('gwasLog10PValue') <= math.log10(settings.gwas_p_value_threshold))
        .join(coding_genes, on='variant', how='inner')
        .join(common.select('variant').distinct(), on='variant', how='left_anti')
        .join(
            diseases.select(f.col('studyId').alias('gwasStudyId'), f.col('diseaseId').alias('as_disease')),
            on='gwasStudyId',
        )
    )
    best = Window.partitionBy('variant', 'as_gene', 'as_disease').orderBy(
        f.col('gwasLog10PValue').asc_nulls_last(), 'gwasCredibleSetId'
    )
    return (
        joined
        .withColumn('_rank', f.row_number().over(best))
        .filter(f.col('_rank') == 1)
        .select(
            'variant',
            'as_gene',
            'as_disease',
            f.lit(SOURCE_CODING_GWAS).alias('GsourceLab'),
            f.lit(QTL_NONE).alias('GqtlLab'),
            'yc',
            'ycse',
            f.lit(DEFAULT_XC).alias('xc'),
            f.lit(DEFAULT_XCSE).alias('xcse'),
            f.lit(DEFAULT_CLINICAL_SIGNIFICANCE).alias('as_clinicalSignificance'),
            'gwasStudyId',
            'gwasCredibleSetId',
            *[
                f.lit(None).cast('string').alias(c)
                for c in PROVENANCE_COLUMNS
                if c not in ('gwasStudyId', 'gwasCredibleSetId')
            ],
        )
    )


def clinical_significance_score() -> f.Column:
    """The score of each element of `clinicalSignificances`, as a map lookup on the lowercase label."""
    return f.create_map(*[
        f.lit(x) for label, score in CLINICAL_SIGNIFICANCE_SCORE.items() for x in (label, float(score))
    ])


def clinvar_variants(evidence_eva: DataFrame, genes: DataFrame, obsolete: DataFrame) -> DataFrame:
    """ClinVar rows (`GsourceLab` 2) from the EVA evidence, with `as_clinicalSignificance` on [0, 1].

    A submission's score is the highest of its clinical significance labels; unknown labels score 0.
    Submissions of excluded review statuses are dropped. Several submissions of one variant and
    disease stay separate rows: the Bayesian step deduplicates them by score.
    """
    scores = clinical_significance_score()
    labels = f.transform(
        f.coalesce(f.col('clinicalSignificances'), f.array().cast('array<string>')), lambda x: f.lower(f.trim(x))
    )
    rows = (
        evidence_eva
        .filter(f.col('datasourceId') == 'eva')
        .filter(f.col('variantId').isNotNull() & f.col('diseaseId').isNotNull())
        .filter(~f.coalesce(f.col('confidence'), f.lit('')).isin(list(EVA_EXCLUDED_CONFIDENCE)))
        .join(genes.select(f.col('geneId').alias('targetId')), on='targetId', how='inner')
        .select(
            f.col('variantId').alias('variant'),
            f.col('targetId').alias('as_gene'),
            f.col('diseaseId').alias('as_disease'),
            f.lit(SOURCE_CLINVAR).alias('GsourceLab'),
            f.lit(QTL_NONE).alias('GqtlLab'),
            f.lit(DEFAULT_YC).alias('yc'),
            f.lit(DEFAULT_YCSE).alias('ycse'),
            f.lit(DEFAULT_XC).alias('xc'),
            f.lit(DEFAULT_XCSE).alias('xcse'),
            f.coalesce(f.array_max(f.transform(labels, lambda x: f.coalesce(scores[x], f.lit(0.0)))), f.lit(0.0)).alias(
                'as_clinicalSignificance'
            ),
            *[
                f.lit(None).cast('string').alias(c)
                for c in PROVENANCE_COLUMNS
                if c not in ('clinvarDirectionOnTarget', 'clinvarDirectionOnTrait')
            ],
            f.col('directionOnTarget').alias('clinvarDirectionOnTarget'),
            f.col('directionOnTrait').alias('clinvarDirectionOnTrait'),
        )
    )
    return remap_diseases(rows, 'as_disease', obsolete)


def map_phenotypes(spark: SparkSession, phenotypes: DataFrame, label_lut: str) -> DataFrame:
    """`phenotype`, `diseaseId`: every EFO id the OnToma label lookup maps each AstraZeneca phenotype to.

    The distinct phenotype strings (a few thousand) are mapped on the driver with the same
    `add_efo_mapping` the gene burden evidence uses, so VIDRA and the platform agree on them.
    """
    distinct = [
        row.phenotype
        for row in phenotypes.select('phenotype').distinct().filter(f.col('phenotype').isNotNull()).collect()
    ]
    mapped = (
        add_efo_mapping(
            pl.LazyFrame({'diseaseFromSource': distinct}, schema={'diseaseFromSource': pl.String}),
            disease_label_lut_path=label_lut,
            disease_id_lut_path=None,
            id_col_name=None,
        )
        .filter(pl.col(MAPPED_ID_COLUMN).is_not_null())
        .select(pl.col('diseaseFromSource').alias('phenotype'), pl.col(MAPPED_ID_COLUMN).alias('diseaseId'))
        .unique()
        .collect()
    )
    logger.info('{} of {} AstraZeneca phenotypes map to a disease', mapped['phenotype'].n_unique(), len(distinct))
    schema = t.StructType([t.StructField('phenotype', t.StringType()), t.StructField('diseaseId', t.StringType())])
    return spark.createDataFrame(mapped.rows(), schema=schema)


def az_rare_variants(az_exwas: DataFrame, genes: DataFrame, phenotype_map: DataFrame, obsolete: DataFrame) -> DataFrame:
    """Rare variant rows (`GsourceLab` 1) from the kept ExWAS associations: `yc = log(OR)`, `ycse` from the CI."""
    rows = (
        az_exwas
        .join(
            genes.select(f.col('approvedSymbol').alias('geneSymbol'), f.col('geneId').alias('as_gene')), on='geneSymbol'
        )
        .join(phenotype_map.select(f.col('phenotype'), f.col('diseaseId').alias('as_disease')), on='phenotype')
        .filter((f.col('oddsRatio') > 0) & (f.col('oddsRatioLower') > 0) & (f.col('oddsRatioUpper') > 0))
        .select(
            f.col('variantId').alias('variant'),
            'as_gene',
            'as_disease',
            f.lit(SOURCE_AZ_RARE).alias('GsourceLab'),
            f.lit(QTL_NONE).alias('GqtlLab'),
            f.log(f.col('oddsRatio')).alias('yc'),
            ((f.log(f.col('oddsRatioUpper')) - f.log(f.col('oddsRatioLower'))) / CI_TO_SE).alias('ycse'),
            f.lit(DEFAULT_XC).alias('xc'),
            f.lit(DEFAULT_XCSE).alias('xcse'),
            f.lit(DEFAULT_CLINICAL_SIGNIFICANCE).alias('as_clinicalSignificance'),
            *_empty_provenance(),
        )
        .distinct()
    )
    return remap_diseases(rows, 'as_disease', obsolete)


def burden_tests(
    binary: DataFrame,
    quantitative: DataFrame,
    genes: DataFrame,
    phenotype_map: DataFrame,
    obsolete: DataFrame,
    settings: VidraInputSettings,
) -> DataFrame:
    """The burden test of each gene and disease: `as_gene`, `as_disease`, `bO`, `bOse`.

    Binary traits give `log(OR)` and its standard error from the CI; quantitative traits their
    beta and CI directly. Per gene and phenotype the lowest p-value wins; after the obsolete id
    remapping, per gene and disease the largest `|bO|` wins.
    """
    binary_rows = binary.select(
        'Gene',
        'Phenotype',
        'CollapsingModel',
        f.col('pValue').cast('double').alias('pValue'),
        f.log(f.col('BinOddsRatio').cast('double')).alias('bO'),
        (
            (f.log(f.col('BinOddsRatioUCI').cast('double')) - f.log(f.col('BinOddsRatioLCI').cast('double'))) / CI_TO_SE
        ).alias('bOse'),
    )
    quantitative_rows = quantitative.select(
        'Gene',
        'Phenotype',
        'CollapsingModel',
        f.col('pValue').cast('double').alias('pValue'),
        f.col('beta').cast('double').alias('bO'),
        ((f.col('UCI').cast('double') - f.col('LCI').cast('double')) / CI_TO_SE).alias('bOse'),
    )
    tests = (
        binary_rows
        .unionByName(quantitative_rows)
        .filter(f.col('CollapsingModel').isin(settings.burden_collapsing_models))
        .filter(f.col('bO').isNotNull() & f.col('bOse').isNotNull() & (f.col('bOse') > 0))
        .join(genes.select(f.col('approvedSymbol').alias('Gene'), f.col('geneId').alias('as_gene')), on='Gene')
        .join(
            phenotype_map.select(f.col('phenotype').alias('Phenotype'), f.col('diseaseId').alias('as_disease')),
            on='Phenotype',
        )
    )
    best_test = Window.partitionBy('as_gene', 'as_disease').orderBy(f.col('pValue').asc_nulls_last(), 'CollapsingModel')
    best = (
        tests
        .withColumn('_rank', f.row_number().over(best_test))
        .filter(f.col('_rank') == 1)
        .select('as_gene', 'as_disease', 'bO', 'bOse')
    )
    remapped = remap_diseases(best, 'as_disease', obsolete)
    largest = Window.partitionBy('as_gene', 'as_disease').orderBy(f.abs(f.col('bO')).desc())
    return remapped.withColumn('_rank', f.row_number().over(largest)).filter(f.col('_rank') == 1).drop('_rank')


# ----------------------------------------------------------------------------------------------
# Annotations
# ----------------------------------------------------------------------------------------------


def variant_effects(variant: DataFrame, method: str) -> DataFrame:
    """`variant`, `targetId`, `score` of the `variantEffect` entries of one method."""
    return (
        variant
        .select(f.col('variantId').alias('variant'), f.explode('variantEffect').alias('effect'))
        .filter(f.col('effect.method') == method)
        .select(
            'variant', f.col('effect.targetId').alias('targetId'), f.col('effect.score').cast('double').alias('score')
        )
    )


def annotate(rows: DataFrame, variant: DataFrame, consequences: DataFrame, settings: VidraInputSettings) -> DataFrame:
    """Add the protein-function annotations of each row's variant and gene.

    `as_gpnstar` is the calibrated GPN-Star transform of the variant (null when it has no score),
    `as_alphamissense` is `1 - pathogenicity` for the gene (null when not a missense change the
    predictor covers), and the consequence columns come from `gene_consequences`; a variant with
    no transcript of the gene gets class `unknown`.
    """
    gpn = variant_effects(variant, 'GPN-Star').groupBy('variant').agg(f.first('score').alias('llr'))
    transform = settings.gpn_star
    gpn = gpn.select(
        'variant', (1.0 - 1.0 / (1.0 + f.exp(-(transform.a + transform.b * f.col('llr'))))).alias('as_gpnstar')
    )
    alphamissense = (
        variant_effects(variant, 'AlphaMissense')
        .groupBy('variant', f.col('targetId').alias('as_gene'))
        .agg((1.0 - f.first('score')).alias('as_alphamissense'))
    )
    return (
        rows
        .join(gpn, on='variant', how='left')
        .join(alphamissense, on=['variant', 'as_gene'], how='left')
        .join(consequences, on=['variant', 'as_gene'], how='left')
        .withColumn('consequence_class', f.coalesce(f.col('consequence_class'), f.lit('unknown')))
        .withColumn('is_lof', f.coalesce(f.col('is_lof'), f.lit(False)))
        .withColumn('consequences', f.coalesce(f.col('consequences'), f.array().cast('array<string>')))
    )


# ----------------------------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------------------------


OUTPUT_COLUMNS = (*ANALYSIS_COLUMNS, *ANNOTATION_COLUMNS, *PROVENANCE_COLUMNS)


def assemble(sources: list[DataFrame], burden: DataFrame) -> DataFrame:
    """Union the source rows, attach the burden test of each gene and disease, and fill the defaults."""
    rows = sources[0]
    for other in sources[1:]:
        rows = rows.unionByName(other)
    return (
        rows
        .filter(f.col('as_gene').isNotNull() & f.col('as_disease').isNotNull() & f.col('variant').isNotNull())
        .join(burden, on=['as_gene', 'as_disease'], how='left')
        .fillna({
            'bO': DEFAULT_BO,
            'bOse': DEFAULT_BOSE,
            'yc': DEFAULT_YC,
            'ycse': DEFAULT_YCSE,
            'xc': DEFAULT_XC,
            'xcse': DEFAULT_XCSE,
            'as_clinicalSignificance': DEFAULT_CLINICAL_SIGNIFICANCE,
        })
    )


def vidra_input(source: dict[str, str], destination: str, settings: dict[str, Any], properties: dict[str, str]) -> None:
    """Build the VIDRA analysis-ready table and write it partitioned by gene.

    Args:
        source: `colocalisation`, `credible_set`, `credible_set_variant`, `study`, `variant`, `target`,
            `disease`, `biosample`, `evidence_eva`, `az_exwas`, `az_burden_binary`, `az_burden_quantitative`
            and `ontoma_disease_label_lut`.
        destination: the output directory, partitioned by `as_gene`.
        settings: `VidraInputSettings`.
        properties: Spark properties.
    """
    options = VidraInputSettings.model_validate(settings)
    session = Session(app_name='vidra_input', properties=properties)
    spark = session.spark

    genes = protein_coding_genes(session.load_data(source['target'])).cache()
    obsolete = obsolete_disease_map(session.load_data(source['disease'])).cache()
    study = session.load_data(source['study'])
    diseases = study_diseases(study, obsolete).cache()
    leads = lead_statistics(
        session.load_data(source['credible_set']), session.load_data(source['credible_set_variant'])
    )
    variant = session.load_data(source['variant'])
    consequences = gene_consequences(variant, genes)

    biosamples = None
    if options.blood_biosample_ancestors:
        biosamples = blood_biosamples(session.load_data(source['biosample']), options.blood_biosample_ancestors)
        logger.info('{} biosamples pass the tissue filter', biosamples.count())
    qtl = qtl_credible_sets(leads, study, genes, biosamples, options)
    gwas = gwas_leads(leads).cache()
    common = common_variants(session.load_data(source['colocalisation']), gwas, qtl, diseases, options).cache()
    logger.info('{} common variant rows', common.count())
    coding = coding_gwas_variants(gwas, consequences, diseases, common, options)
    logger.info('{} coding GWAS rows', coding.count())
    clinvar = clinvar_variants(session.load_data(source['evidence_eva']), genes, obsolete)
    logger.info('{} ClinVar rows', clinvar.count())

    az_exwas = session.load_data(source['az_exwas'])
    binary = session.load_data(source['az_burden_binary'])
    quantitative = session.load_data(source['az_burden_quantitative'])
    phenotypes = (
        az_exwas
        .select('phenotype')
        .unionByName(binary.select(f.col('Phenotype').alias('phenotype')))
        .unionByName(quantitative.select(f.col('Phenotype').alias('phenotype')))
    )
    phenotype_map = map_phenotypes(spark, phenotypes, source['ontoma_disease_label_lut']).cache()
    rare = az_rare_variants(az_exwas, genes, phenotype_map, obsolete)
    logger.info('{} AstraZeneca rare variant rows', rare.count())
    burden = burden_tests(binary, quantitative, genes, phenotype_map, obsolete, options)
    logger.info('{} burden tests', burden.count())

    rows = annotate(assemble([common, coding, clinvar, rare], burden), variant, consequences, options).select(
        *OUTPUT_COLUMNS
    )
    n_genes = rows.select('as_gene').distinct().count()
    logger.info('writing {} genes to {}', n_genes, destination)
    rows.repartition(max(n_genes, 1), 'as_gene').write.mode('overwrite').partitionBy('as_gene').parquet(destination)
    logger.info('{} analysis-ready rows written', spark.read.parquet(destination).count())
