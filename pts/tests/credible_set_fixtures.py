"""Credible sets as their producers write them, for the stages and the tasks that read them.

Three real credible sets and their variants, taken from the 2026-10 new-schema PIS inputs:

- `ALT_CONTIG`: a GWAS Catalog curated top hit (`gcca`) with a member on an alternate contig.
- `MHC`: a GWAS Catalog PICS set (`gcss_pics`) with its lead in the MHC region.
- `CLEAN`: a FinnGen SuSiE set (`finngen`) that passes every check.

`published_credible_set` and `published_credible_set_variant` build the datasets validation and
annotation write, with every column and dtype: the tasks that read them are tested on these, so a
change of either side breaks their tests.
"""

from typing import Any

import polars as pl

from pts.schemas.credible_set import credible_set_variant_schema, validated_credible_set_schema


def published_credible_set(*row: dict[str, Any]) -> pl.DataFrame:
    """Credible sets as validation and annotation write them: every column, with its enum dtypes.

    A column missing from a row is null, and `qualityControls` is empty.
    """
    return pl.DataFrame(
        [
            {name: {'qualityControls': [], **set_row}.get(name) for name in validated_credible_set_schema}
            for set_row in row
        ],
        schema=validated_credible_set_schema,
    )


def published_credible_set_variant(*row: dict[str, Any]) -> pl.DataFrame:
    """Credible set variants as validation and annotation write them: every column, with its dtypes.

    A column missing from a row is null, and `qualityControls` is empty.
    """
    return pl.DataFrame(
        [
            {name: {'qualityControls': [], **variant_row}.get(name) for name in credible_set_variant_schema}
            for variant_row in row
        ],
        schema=credible_set_variant_schema,
    )


ALT_CONTIG_SET: dict[str, Any] = {
    'id': '97a0793ea0f67c070ac3970f00bb39c4',
    'chromosome': '14',
    'locusStart': 106012954,
    'locusEnd': 106012954,
    'studyId': 'GCST90240321',
    'studyType': 'gwas',
    'leadVariantId': '14_106012954_A_G',
    'fineMappingMethod': 'PICS',
    'logBayesFactor': None,
    'confidence': None,
    'purityMeanR2': None,
    'purityMinR2': None,
    'qualityControls': ['Study locus finemapped without in-sample LD reference', 'Study locus from curated top hit'],
}

ALT_CONTIG_VARIANTS: list[dict[str, Any]] = [
    {
        'credibleSetId': '97a0793ea0f67c070ac3970f00bb39c4',
        'variantId': '14_106006753_C_T',
        'pValueMantissa': None,
        'pValueExponent': None,
        'effectAlleleFrequency': None,
        'sampleSize': None,
        'reportedBeta': None,
        'reportedStandardError': None,
        'posteriorProbability': 0.04588742181658745,
        'logBayesFactor': None,
        'r2Overall': 0.9690424799919128,
        'conditionalBeta': None,
        'conditionalStandardError': None,
        'rescaledAbsBeta': None,
        'rescaledStandardError': None,
        'rescaledDirection': None,
        'qualityControls': ['Variant in LD with the lead has no statistics in the locus'],
    },
    {
        'credibleSetId': '97a0793ea0f67c070ac3970f00bb39c4',
        'variantId': '14_106007799_C_G',
        'pValueMantissa': None,
        'pValueExponent': None,
        'effectAlleleFrequency': None,
        'sampleSize': None,
        'reportedBeta': None,
        'reportedStandardError': None,
        'posteriorProbability': 0.0212023314088583,
        'logBayesFactor': None,
        'r2Overall': 0.9537929892539978,
        'conditionalBeta': None,
        'conditionalStandardError': None,
        'rescaledAbsBeta': None,
        'rescaledStandardError': None,
        'rescaledDirection': None,
        'qualityControls': ['Variant in LD with the lead has no statistics in the locus'],
    },
    {
        'credibleSetId': '97a0793ea0f67c070ac3970f00bb39c4',
        'variantId': '14_106008164_G_A',
        'pValueMantissa': None,
        'pValueExponent': None,
        'effectAlleleFrequency': None,
        'sampleSize': None,
        'reportedBeta': None,
        'reportedStandardError': None,
        'posteriorProbability': 0.05374649539589882,
        'logBayesFactor': None,
        'r2Overall': 0.972093939781189,
        'conditionalBeta': None,
        'conditionalStandardError': None,
        'rescaledAbsBeta': None,
        'rescaledStandardError': None,
        'rescaledDirection': None,
        'qualityControls': ['Variant in LD with the lead has no statistics in the locus'],
    },
    {
        'credibleSetId': '97a0793ea0f67c070ac3970f00bb39c4',
        'variantId': '14_106008855_C_T',
        'pValueMantissa': None,
        'pValueExponent': None,
        'effectAlleleFrequency': None,
        'sampleSize': None,
        'reportedBeta': None,
        'reportedStandardError': None,
        'posteriorProbability': 0.08084804564714432,
        'logBayesFactor': None,
        'r2Overall': 0.979771077632904,
        'conditionalBeta': None,
        'conditionalStandardError': None,
        'rescaledAbsBeta': None,
        'rescaledStandardError': None,
        'rescaledDirection': None,
        'qualityControls': ['Variant in LD with the lead has no statistics in the locus'],
    },
    {
        'credibleSetId': '97a0793ea0f67c070ac3970f00bb39c4',
        'variantId': '14_106012954_A_G',
        'pValueMantissa': 5.0,
        'pValueExponent': -61,
        'effectAlleleFrequency': None,
        'sampleSize': None,
        'reportedBeta': 0.65,
        'reportedStandardError': 0.03943864201161859,
        'posteriorProbability': 0.38174936175346375,
        'logBayesFactor': None,
        'r2Overall': 1.0,
        'conditionalBeta': None,
        'conditionalStandardError': None,
        'rescaledAbsBeta': None,
        'rescaledStandardError': None,
        'rescaledDirection': None,
        'qualityControls': [],
    },
    {
        'credibleSetId': '97a0793ea0f67c070ac3970f00bb39c4',
        'variantId': '14_106013509_A_C',
        'pValueMantissa': None,
        'pValueExponent': None,
        'effectAlleleFrequency': None,
        'sampleSize': None,
        'reportedBeta': None,
        'reportedStandardError': None,
        'posteriorProbability': 0.11534691601991653,
        'logBayesFactor': None,
        'r2Overall': 0.9860902428627014,
        'conditionalBeta': None,
        'conditionalStandardError': None,
        'rescaledAbsBeta': None,
        'rescaledStandardError': None,
        'rescaledDirection': None,
        'qualityControls': ['Variant in LD with the lead has no statistics in the locus'],
    },
    {
        'credibleSetId': '97a0793ea0f67c070ac3970f00bb39c4',
        'variantId': '14_106015262_C_G',
        'pValueMantissa': None,
        'pValueExponent': None,
        'effectAlleleFrequency': None,
        'sampleSize': None,
        'reportedBeta': None,
        'reportedStandardError': None,
        'posteriorProbability': 0.07689211517572403,
        'logBayesFactor': None,
        'r2Overall': 0.9788475632667542,
        'conditionalBeta': None,
        'conditionalStandardError': None,
        'rescaledAbsBeta': None,
        'rescaledStandardError': None,
        'rescaledDirection': None,
        'qualityControls': ['Variant in LD with the lead has no statistics in the locus'],
    },
    {
        'credibleSetId': '97a0793ea0f67c070ac3970f00bb39c4',
        'variantId': '14_106016546_G_A',
        'pValueMantissa': None,
        'pValueExponent': None,
        'effectAlleleFrequency': None,
        'sampleSize': None,
        'reportedBeta': None,
        'reportedStandardError': None,
        'posteriorProbability': 0.11902563273906708,
        'logBayesFactor': None,
        'r2Overall': 0.9866256713867188,
        'conditionalBeta': None,
        'conditionalStandardError': None,
        'rescaledAbsBeta': None,
        'rescaledStandardError': None,
        'rescaledDirection': None,
        'qualityControls': ['Variant in LD with the lead has no statistics in the locus'],
    },
    {
        'credibleSetId': '97a0793ea0f67c070ac3970f00bb39c4',
        'variantId': '14_KI270846v1_alt_497149_C_T',
        'pValueMantissa': None,
        'pValueExponent': None,
        'effectAlleleFrequency': None,
        'sampleSize': None,
        'reportedBeta': None,
        'reportedStandardError': None,
        'posteriorProbability': 0.06548985093832016,
        'logBayesFactor': None,
        'r2Overall': 0.9758524298667908,
        'conditionalBeta': None,
        'conditionalStandardError': None,
        'rescaledAbsBeta': None,
        'rescaledStandardError': None,
        'rescaledDirection': None,
        'qualityControls': ['Variant in LD with the lead has no statistics in the locus'],
    },
]

MHC_SET: dict[str, Any] = {
    'id': '00004b85ac7ad992735b4e39a90497cc',
    'chromosome': '6',
    'locusStart': 25592913,
    'locusEnd': 26592913,
    'studyId': 'GCST90019509',
    'studyType': 'gwas',
    'leadVariantId': '6_26092913_G_A',
    'fineMappingMethod': 'PICS',
    'logBayesFactor': None,
    'confidence': None,
    'purityMeanR2': None,
    'purityMinR2': None,
    'qualityControls': ['Study locus finemapped without in-sample LD reference'],
}

MHC_VARIANTS: list[dict[str, Any]] = [
    {
        'credibleSetId': '00004b85ac7ad992735b4e39a90497cc',
        'variantId': '6_26092913_G_A',
        'pValueMantissa': 5.25,
        'pValueExponent': -174,
        'effectAlleleFrequency': None,
        'sampleSize': None,
        'reportedBeta': -0.1299,
        'reportedStandardError': 0.0046,
        'posteriorProbability': 0.926308810710907,
        'logBayesFactor': None,
        'r2Overall': 1.0,
        'conditionalBeta': None,
        'conditionalStandardError': None,
        'rescaledAbsBeta': None,
        'rescaledStandardError': None,
        'rescaledDirection': None,
        'qualityControls': [],
    },
    {
        'credibleSetId': '00004b85ac7ad992735b4e39a90497cc',
        'variantId': '6_26098246_T_C',
        'pValueMantissa': 4.849999904632568,
        'pValueExponent': -173,
        'effectAlleleFrequency': None,
        'sampleSize': None,
        'reportedBeta': -0.13,
        'reportedStandardError': 0.0046,
        'posteriorProbability': 0.061798419803380966,
        'logBayesFactor': None,
        'r2Overall': 0.9847370386123657,
        'conditionalBeta': None,
        'conditionalStandardError': None,
        'rescaledAbsBeta': None,
        'rescaledStandardError': None,
        'rescaledDirection': None,
        'qualityControls': [],
    },
]

CLEAN_SET: dict[str, Any] = {
    'id': '016ac7f9dc03c404e9e3f0b517d80a4d',
    'chromosome': '1',
    'locusStart': 153651361,
    'locusEnd': 156651361,
    'studyId': 'FINNGEN_R12_M13_GOUT',
    'studyType': 'gwas',
    'leadVariantId': '1_155151361_G_C',
    'fineMappingMethod': 'SuSiE',
    'logBayesFactor': 22.110761110150023,
    'confidence': None,
    'purityMeanR2': 0.926893980844,
    'purityMinR2': 0.891256836096,
    'qualityControls': [],
}

CLEAN_VARIANTS: list[dict[str, Any]] = [
    {
        'credibleSetId': '016ac7f9dc03c404e9e3f0b517d80a4d',
        'variantId': '1_155151361_G_C',
        'pValueMantissa': 1.909000039100647,
        'pValueExponent': -16,
        'effectAlleleFrequency': 0.37560901045799255,
        'sampleSize': None,
        'reportedBeta': 0.113354,
        'reportedStandardError': 0.0137771,
        'posteriorProbability': 0.6104365587234497,
        'logBayesFactor': 31.311180114746094,
        'r2Overall': 1.0,
        'conditionalBeta': 0.11177290230989456,
        'conditionalStandardError': 0.013673300854861736,
        'rescaledAbsBeta': None,
        'rescaledStandardError': None,
        'rescaledDirection': None,
        'qualityControls': ['Allele frequency reported by the study'],
    },
    {
        'credibleSetId': '016ac7f9dc03c404e9e3f0b517d80a4d',
        'variantId': '1_155208991_T_A',
        'pValueMantissa': 5.117000102996826,
        'pValueExponent': -16,
        'effectAlleleFrequency': 0.3765780031681061,
        'sampleSize': None,
        'reportedBeta': 0.111731,
        'reportedStandardError': 0.0137791,
        'posteriorProbability': 0.2401031255722046,
        'logBayesFactor': 30.378074645996094,
        'r2Overall': 0.8987457156181335,
        'conditionalBeta': 0.11021674424409866,
        'conditionalStandardError': 0.013675245456397533,
        'rescaledAbsBeta': None,
        'rescaledStandardError': None,
        'rescaledDirection': None,
        'qualityControls': ['Allele frequency reported by the study'],
    },
    {
        'credibleSetId': '016ac7f9dc03c404e9e3f0b517d80a4d',
        'variantId': '1_155215184_G_A',
        'pValueMantissa': 9.682999610900879,
        'pValueExponent': -16,
        'effectAlleleFrequency': 0.3774769902229309,
        'sampleSize': None,
        'reportedBeta': 0.11068,
        'reportedStandardError': 0.0137819,
        'posteriorProbability': 0.13098196685314178,
        'logBayesFactor': 29.772066116333008,
        'r2Overall': 0.8912568092346191,
        'conditionalBeta': 0.10920562595129013,
        'conditionalStandardError': 0.013677997514605522,
        'rescaledAbsBeta': None,
        'rescaledStandardError': None,
        'rescaledDirection': None,
        'qualityControls': ['Allele frequency reported by the study'],
    },
]
