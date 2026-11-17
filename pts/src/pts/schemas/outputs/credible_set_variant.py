"""Credible set variant: the `credible_set_variant` output.

Generated from the 26.12 release output (141,034,156 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from pts.schemas.dataset import DatasetModel, PolarsDtype, PrimaryKey, Reference
from pts.schemas.outputs.patterns import MD5_HEX, VARIANT_ID


class CredibleSetVariant(DatasetModel):
    """Credible set variant."""

    dataset_name: ClassVar[str] = 'credible_set_variant'
    dataset_title: ClassVar[str] = 'Credible set variant'
    dataset_description: ClassVar[str] = (
        'The variants of the 95% credible sets for GWAS and molQTL studies: '
        'one row per credible set and variant, with its association '
        'statistics, posterior probability, effect allele frequency and '
        'rescaled effect size'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Genetics',)
    example: ClassVar[dict] = {
        'credibleSetId': 'fef6593f44a085e9eb140939d50fb81e',
        'variantId': '2_132771885_G_T',
        'pValueMantissa': 6.0,
        'pValueExponent': -9,
        'effectAlleleFrequency': 0.1025327518582344,
        'sampleSize': 235850,
        'reportedBeta': 0.008,
        'reportedStandardError': 0.0013959829864472562,
        'posteriorProbability': 0.2966385781764984,
        'logBayesFactor': 12.62327766418457,
        'r2Overall': 1.0,
        'conditionalBeta': -0.00809260830283165,
        'conditionalStandardError': 0.076714426279068,
        'rescaledAbsBeta': 0.017276180908083916,
        'rescaledStandardError': 0.003014656715095043,
        'rescaledDirection': 1,
        'qualityControls': ['Allele frequency taken from gnomAD'],
    }

    credibleSetId: Annotated[str, Reference('credible_set', 'id'), PrimaryKey()] = Field(
        pattern=MD5_HEX, description='Identifier of the credible set of the variant'
    )
    variantId: Annotated[str, Reference('variant', 'variantId'), PrimaryKey()] = Field(
        pattern=VARIANT_ID, description='Identifier of the variant'
    )
    pValueMantissa: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        ge=0.0, description='Mantissa of the p-value of the variant'
    )
    pValueExponent: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        description='Exponent of the p-value of the variant'
    )
    effectAlleleFrequency: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        ge=0.0,
        le=1.0,
        description=(
            'Frequency of the effect allele, from the study or otherwise from gnomAD for the study populations'
        ),
    )
    sampleSize: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, description='Sample size of the association of the variant'
    )
    reportedBeta: float | None = Field(description='Effect size of the variant as reported by the study')
    reportedStandardError: float | None = Field(ge=0.0, description='Standard error of the reported effect size')
    posteriorProbability: Annotated[float, PolarsDtype(pl.Float32)] = Field(
        ge=0.0, le=1.0, description='Posterior inclusion probability of the variant in the credible set'
    )
    logBayesFactor: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        description='Log Bayes factor of the variant'
    )
    r2Overall: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        ge=0.0, le=1.0, description='LD r2 between the variant and the lead variant'
    )
    conditionalBeta: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        description=('Effect size of the variant conditional on the other signals of the locus')
    )
    conditionalStandardError: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        ge=0.0, description='Standard error of the conditional effect size'
    )
    rescaledAbsBeta: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        ge=0.0,
        description=(
            'Absolute effect size rescaled to the standard deviation of a '
            'quantitative trait or the log odds of a binary trait'
        ),
    )
    rescaledStandardError: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        ge=0.0, description='Standard error of the rescaled effect size'
    )
    rescaledDirection: Annotated[int, PolarsDtype(pl.Int8)] | None = Field(
        description=('Direction of the rescaled effect relative to the effect allele: 1 or -1')
    )
    qualityControls: list[
        Literal[
            'Allele frequency reported by the study',
            'Allele frequency taken from gnomAD',
            'Allele frequency is missing from the study and from gnomAD',
            'Allele frequency reported by the study is not in [0, 1]',
            'Rescaled effect derived from the conditional effect',
            'Rescaled effect derived from the marginal effect',
            'Variant in LD with the lead has no statistics in the locus',
            'Variant identifier is not a valid chromosome_position_ref_alt',
            'Variant appears more than once in its credible set',
            'Credible set of the variant is excluded',
            'Credible set identifier is not a 32-character hexadecimal md5',
            'Variant is outside the 95% coverage of the credible set',
            'Credible set PIPs do not sum to the expected coverage, or a PIP is not in (0, 1]',
            'Reported and conditional effects have opposite signs',
            'Only one of the reported beta and standard error is present',
            'Only one of the conditional beta and standard error is present',
            'P-value mantissa or exponent is missing',
            'Credible set of the variant is not among the credible sets of its source',
            'Chromosome of the variant differs from the credible set chromosome',
        ]
    ] = Field(
        description=(
            'Quality control flags of the variant, including the origin of its '
            'allele frequency and of its rescaled direction'
        )
    )
