"""Quality flags and confidence classes of the CredibleSet and CredibleSetVariant datasets."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class CredibleSetFlag(StrEnum):
    """Quality control flags recorded on credible sets.

    A flag the sources already write keeps the text they write.
    """

    WINDOW_CLUMPED = 'Explained by a more significant variant in the same window'
    LD_CLUMPED = 'Explained by a more significant variant in high LD'
    NO_POPULATION = 'Study does not have population annotation to resolve LD'
    UNRESOLVED_LD = 'Variant not found in LD reference'
    OUT_OF_SAMPLE_LD = 'Study locus finemapped without in-sample LD reference'
    SUBSIGNIFICANT_FLAG = 'Subsignificant p-value'
    COMPOSITE_FLAG = 'Composite association'
    INCONSISTENCY_FLAG = 'Variant inconsistency'
    PALINDROMIC_ALLELE_FLAG = 'Palindrome alleles - cannot harmonize'
    TOP_HIT = 'Study locus from curated top hit'
    TRANS_QTL = 'Trans QTL'
    REPLICATED = 'Lead variant is replicated in an independent study'
    MISSING_STUDY = 'Study not found in the study index'
    TOP_HIT_AND_SUMMARY_STATS = 'Curated top hit is flagged because summary statistics are available for study'
    IN_MHC = 'MHC region'
    INVALID_CHROMOSOME = 'Chromosome not in 1:22, X, Y or MT'
    REDUNDANT_PICS_TOP_HIT = 'PICS results from summary statistics available for this same study'
    EXPLAINED_BY_SUSIE = 'Study locus in region explained by a SuSiE credible set'
    DUPLICATED_CREDIBLE_SET_ID = 'Credible set identifier repeated within its source'
    DUPLICATED_CREDIBLE_SET_ID_ACROSS_SOURCES = 'Credible set identifier also in another source, whose copy is kept'
    INVALID_VARIANT_IN_SET = 'Credible set has a variant with an excluding quality flag'
    INVALID_CS_ID = 'Credible set identifier is not a 32-character hexadecimal md5'
    LEAD_CHROMOSOME_MISMATCH = 'Chromosome of the lead variant differs from the credible set chromosome'
    INVALID_FINEMAPPING_METHOD = 'Fine-mapping method is not supported'
    INVALID_STUDY_TYPE = 'Study type of the source is not a supported study type'
    STUDY_TYPE_MISMATCH = 'Study type of the credible set differs from the study index'
    LEAD_VARIANT_MISSING = 'Lead variant of the credible set has no row among its variants'


class CredibleSetVariantFlag(StrEnum):
    """Quality control flags recorded on credible set variants.

    A flag the sources already write keeps the text they write.
    """

    EAF_FROM_SOURCE = 'Allele frequency reported by the study'
    EAF_FROM_GNOMAD = 'Allele frequency taken from gnomAD'
    MISSING_EAF = 'Allele frequency is missing from the study and from gnomAD'
    INVALID_EAF = 'Allele frequency reported by the study is not in [0, 1]'
    RESCALED_FROM_CONDITIONAL = 'Rescaled effect derived from the conditional effect'
    RESCALED_FROM_MARGINAL = 'Rescaled effect derived from the marginal effect'
    NOT_IN_LOCUS = 'Variant in LD with the lead has no statistics in the locus'
    INVALID_VARIANT_ID = 'Variant identifier is not a valid chromosome_position_ref_alt'
    DUPLICATED_VARIANT = 'Variant appears more than once in its credible set'
    CREDIBLE_SET_EXCLUDED = 'Credible set of the variant is excluded'
    INVALID_CS_ID = 'Credible set identifier is not a 32-character hexadecimal md5'
    EXCEEDS_COVERAGE_95 = 'Variant is outside the 95% coverage of the credible set'
    ABNORMAL_PIP = 'Credible set PIPs do not sum to the expected coverage, or a PIP is not in (0, 1]'
    EFFECT_DIRECTION_MISMATCH = 'Reported and conditional effects have opposite signs'
    INCOMPLETE_REPORTED_STATS = 'Only one of the reported beta and standard error is present'
    INCOMPLETE_CONDITIONAL_STATS = 'Only one of the conditional beta and standard error is present'
    MISSING_PVALUE = 'P-value mantissa or exponent is missing'
    ORPHAN_VARIANT = 'Credible set of the variant is not among the credible sets of its source'
    VARIANT_CHROMOSOME_MISMATCH = 'Chromosome of the variant differs from the credible set chromosome'


class CredibleSetConfidenceClasses(IntEnum):
    """Confidence of a credible set, from its fine-mapping method and LD reference."""

    FINEMAPPED_IN_SAMPLE_LD = 4
    FINEMAPPED_OUT_OF_SAMPLE_LD = 3
    PICSED_SUMMARY_STATS = 2
    PICSED_TOP_HIT = 1
    UNKNOWN = 0
