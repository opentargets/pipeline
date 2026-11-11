"""Labels, defaults and column names shared by the VIDRA steps.

VIDRA (variant-informed dose-response analysis) regresses disease risk on protein function across
the variants of a gene-disease pair. Every row of its analysis-ready table is one variant
observation for one gene and one disease, labelled by where it came from.
"""

from __future__ import annotations

# `GsourceLab`: the evidence source of a row.
SOURCE_COMMON_QTL = 0
"""Common variant: a GWAS credible set colocalising with an eQTL or pQTL credible set of the gene."""
SOURCE_AZ_RARE = 1
"""Rare variant: AstraZeneca PheWAS Portal ExWAS association (allelic model)."""
SOURCE_CLINVAR = 2
"""ClinVar variant, from the EVA evidence."""
SOURCE_CODING_GWAS = 3
"""Coding GWAS variant: a genome-wide significant GWAS lead in the gene's coding sequence, not colocalising."""

# `GqtlLab`: the QTL branch of a row.
QTL_EQTL = 0
QTL_PQTL = 1
QTL_NONE = 2
"""Not a QTL branch (rare, ClinVar and coding GWAS rows)."""

# Fill values when an effect is not measured, as in the original pipeline.
DEFAULT_XC, DEFAULT_XCSE = 0.0, 0.1
"""No QTL effect."""
DEFAULT_YC, DEFAULT_YCSE = 0.0, 0.14
"""No GWAS effect."""
DEFAULT_BO, DEFAULT_BOSE = 0.0, 2.0
"""No burden test."""
DEFAULT_CLINICAL_SIGNIFICANCE = 0.0
"""Not a ClinVar row."""

CLINICAL_SIGNIFICANCE_ORDER = (
    'not provided',
    'association not found',
    'other',
    'benign',
    'likely benign',
    'low penetrance',
    'confers sensitivity',
    'uncertain risk allele',
    'drug response',
    'uncertain significance',
    'association',
    'affects',
    'likely risk allele',
    'risk factor',
    'established risk allele',
    'likely pathogenic',
    'pathogenic',
)
"""ClinVar clinical significance, from benign to pathogenic, scored as its index over 16."""

CLINICAL_SIGNIFICANCE_ALIASES = {
    # Composite ClinVar values scored as undetermined rather than falling through to benign.
    'conflicting interpretations of pathogenicity': 'uncertain significance',
    'conflicting classifications of pathogenicity': 'uncertain significance',
}

CLINICAL_SIGNIFICANCE_SCORE: dict[str, float] = {
    **{
        label: index / (len(CLINICAL_SIGNIFICANCE_ORDER) - 1) for index, label in enumerate(CLINICAL_SIGNIFICANCE_ORDER)
    },
}
for _alias, _target in CLINICAL_SIGNIFICANCE_ALIASES.items():
    CLINICAL_SIGNIFICANCE_SCORE[_alias] = CLINICAL_SIGNIFICANCE_SCORE[_target]

EVA_EXCLUDED_CONFIDENCE = (
    'criteria provided, conflicting interpretations',
    'criteria provided, conflicting classifications',
    'no assertion criteria provided',
    'no assertion provided',
    'no classification provided',
)
"""ClinVar review statuses whose submissions VIDRA does not use."""

ANALYSIS_COLUMNS = (
    'variant',
    'as_gene',
    'as_disease',
    'GsourceLab',
    'GqtlLab',
    'yc',
    'ycse',
    'xc',
    'xcse',
    'bO',
    'bOse',
    'as_clinicalSignificance',
)
"""The analysis-ready columns the Stan models consume, as named in the original pipeline."""

ANNOTATION_COLUMNS = ('as_gpnstar', 'as_alphamissense', 'consequence_class', 'consequences', 'is_lof')
"""Protein-function annotation columns, joined from the variant index per variant and gene."""

PROVENANCE_COLUMNS = (
    'gwasStudyId',
    'gwasCredibleSetId',
    'qtlStudyId',
    'qtlCredibleSetId',
    'qtlVariantId',
    'qtlBiosampleId',
    'colocH4',
    'colocBetaRatioSignAverage',
    'clinvarDirectionOnTarget',
    'clinvarDirectionOnTrait',
)
"""Where a row came from; not read by the models."""
