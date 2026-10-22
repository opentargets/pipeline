"""Credible set: the `credible_set` output.

Generated from the 26.09 release output (5,918,177 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from pts.schemas.dataset import DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import MD5_HEX, VARIANT_ID


class LdSet(StructModel):
    """`ldSet[]`."""

    tagVariantId: str = Field(
        min_length=1, description=('The variant ID for tag variants in LD with the credible set lead variant')
    )
    r2Overall: float = Field(
        ge=0.0,
        le=1.0000000000005778,
        description=('The R-squared value for the tag variants with the credible set lead variant'),
    )


class Locus(StructModel):
    """`locus[]`."""

    is95CredibleSet: bool = Field(description='Boolean for if the variant is part of the 95% credible set')
    is99CredibleSet: bool = Field(description='Boolean for if the variant is part of the 99% credible set')
    logBF: float | None = Field(
        ge=-0.886426304446791,
        le=4933615719.867019,
        description='Log (natural) Bayes factor for the variant from fine-mapping',
    )
    posteriorProbability: float = Field(
        ge=5.4428556424997e-06,
        le=1.0,
        description=('Posterior inclusion probability for the variant within this credible set'),
    )
    variantId: str = Field(min_length=1, description='Variant ID for this variant in the credible set')
    pValueMantissa: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        ge=1.0, le=10.0, description='Mantissa of the P-value for this variant in the credible set'
    )
    pValueExponent: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=-28539, le=-1, description='Exponent of the P-value for this variant in the credible set'
    )
    beta: float | None = Field(
        ge=-15750.792, le=78729.0, description='Beta coefficient of this variant in the credible set'
    )
    standardError: float | None = Field(
        ge=3.10052369577648e-14, le=3.56105, description='Standard error of this variant in the credible set'
    )
    r2Overall: float | None = Field(
        ge=0.501602615507378,
        le=1.0000000000005778,
        description='R-squared (LD) between this credible set variant and the lead variant',
    )


class CredibleSet(DatasetModel):
    """Credible set."""

    dataset_name: ClassVar[str] = 'credible_set'
    dataset_title: ClassVar[str] = 'Credible set'
    dataset_description: ClassVar[str] = (
        '95% credible sets for GWAS and molQTL studies. Credible sets include '
        'all variants in the credible set as well as the fine-mapping method '
        'and statistics used to estimate the credible set'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Genetics',)
    example: ClassVar[dict] = {
        'studyLocusId': '7a0db1aa41ad031279fff3bcc996be45',
        'studyId': ('qts000015_qtd000993_gtex_v10_majiq_spleen_ensg00000227232_s_17526-17742_17362-17603'),
        'variantId': '1_17005_A_G',
        'chromosome': '1',
        'position': 17005,
        'region': 'chr1:-970430-1029570',
        'beta': 1.21468,
        'zScore': -6.75,
        'pValueMantissa': 4.936999797821045,
        'pValueExponent': -6,
        'effectAlleleFrequencyFromSource': 0.008874669671058655,
        'standardError': 0.260147,
        'subStudyDescription': None,
        'qualityControls': ['Study has quality control flag(s)'],
        'finemappingMethod': 'SuSie',
        'credibleSetIndex': 2,
        'credibleSetlog10BF': 2.055420160293579,
        'purityMeanR2': 1.0,
        'purityMinR2': 1.0,
        'locusStart': 500715,
        'locusEnd': 978367,
        'sampleSize': 8282,
        'ldSet': [{'tagVariantId': '1_51427_T_G', 'r2Overall': 1.0}],
        'locus': [
            {
                'is95CredibleSet': True,
                'is99CredibleSet': True,
                'logBF': 10.947388135336865,
                'posteriorProbability': 0.9868013523090552,
                'variantId': '1_17005_A_G',
                'pValueMantissa': 4.936999797821045,
                'pValueExponent': -6,
                'beta': 1.21468,
                'standardError': 0.260147,
                'r2Overall': 1.0,
            }
        ],
        'confidence': 'SuSiE fine-mapped credible set with in-sample LD',
        'studyType': 'sqtl',
        'isTransQtl': False,
    }

    studyLocusId: Annotated[str, PrimaryKey()] = Field(
        pattern=MD5_HEX, description='Study-locus identifier for the credible set'
    )
    studyId: Annotated[str, Reference('study', 'studyId')] = Field(
        min_length=1, description='Identifier for the study which the credible set is derived'
    )
    variantId: Annotated[str, Reference('variant', 'variantId')] = Field(
        pattern=VARIANT_ID, description='The lead variant for the credible set, by posterior probability'
    )
    chromosome: str = Field(min_length=1, description='Chromosome which the credible set is located')
    position: Annotated[int, PolarsDtype(pl.Int32)] = Field(
        ge=10559, le=248944243, description='Position of the lead variant for the credible set (GRCh38)'
    )
    region: str | None = Field(min_length=1, description='Start and end positions of the region used for fine-mapping')
    beta: float | None = Field(ge=-15750.792, le=78729.0, description='Beta coefficient of the lead variant')
    zScore: float | None = Field(
        ge=-238.52091432335, le=361.45454545454544, description='Z-score of the lead variant from the GWAS'
    )
    pValueMantissa: Annotated[float, PolarsDtype(pl.Float32)] = Field(
        ge=1.0, le=10.0, description='Mantissa value of the lead variant P-value'
    )
    pValueExponent: Annotated[int, PolarsDtype(pl.Int32)] = Field(
        ge=-28539, le=-3, description='Exponent value of the lead variant P-value'
    )
    effectAlleleFrequencyFromSource: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        ge=0.0, le=1.0, description='Allele frequency of the lead variant from the GWAS'
    )
    standardError: float | None = Field(ge=0.0, le=9457.2, description='Standard error of the lead variant')
    subStudyDescription: str | None = Field(description='[Deprecated]')
    qualityControls: list[
        Literal[
            'Palindrome alleles - cannot harmonize',
            'Study has quality control flag(s)',
            'Study locus finemapped without in-sample LD reference',
            'Study locus from curated top hit',
            'Study locus with a sum of PIPs that not in the expected range [0.95,1]',
            'Variant not found in LD reference',
        ]
    ] = Field(min_length=1, description='Quality control flags for this credible set')
    finemappingMethod: Literal['PICS', 'SuSiE-inf', 'SuSie'] = Field(
        description='Method used for fine-mapping of credible set'
    )
    credibleSetIndex: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=1, le=10, description='Integer label for the order of credible sets from study-region'
    )
    credibleSetlog10BF: float | None = Field(
        ge=-0.8214126825332642, le=2142642079.466912, description='Log10 Bayes factor for the entire credible set'
    )
    purityMeanR2: float | None = Field(
        ge=7.7315139136e-08,
        le=1.0,
        description='Mean R-squared linkage disequilibrium for variants in the credible set',
    )
    purityMinR2: float | None = Field(
        ge=0.0, le=1.0, description=('Minimum R-squared linkage disequilibrium for variants in the credible set')
    )
    locusStart: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, le=248810446, description=('Start position of the region that was fine-mapped for this credible set')
    )
    locusEnd: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=57669, le=249687425, description='End position of the region that was fine-mapped for this credible set'
    )
    sampleSize: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=1, le=2915420, description='Sample size of the study from which this credible set is derived'
    )
    ldSet: list[LdSet] | None = Field(
        min_length=1,
        description=('Array of structs which denote the variants in LD with the credible set lead variant'),
    )
    locus: list[Locus] = Field(
        min_length=1, description='Array of structs which denote the variants within the credible set'
    )
    confidence: Literal[
        'PICS fine-mapped credible set based on reported top hit',
        'PICS fine-mapped credible set extracted from summary statistics',
        'SuSiE fine-mapped credible set with in-sample LD',
        'SuSiE fine-mapped credible set with out-of-sample LD',
    ] = Field(description=('Description of how this credible set was derived in terms of data and fine-mapping method'))
    studyType: Literal['eqtl', 'gwas', 'pqtl', 'sceqtl', 'sqtl', 'tuqtl'] = Field(
        description=('Descriptor for whether the credible set is derived from GWAS or molecular QTL')
    )
    isTransQtl: bool | None = Field(description='Boolean for whether this credible set is a trans-pQTL or not')
