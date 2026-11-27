"""Credible set: the `credible_set` output.

Generated from the 26.12 release output (5,981,052 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from ot_datamodels import ClickHouseType, DatasetModel, PolarsDtype, PrimaryKey, Reference
from ot_datamodels.outputs.patterns import MD5_HEX, VARIANT_ID
from ot_datamodels.vocabularies import Chromosome, CredibleSetQualityControl, FineMappingMethod, StudyType


class CredibleSet(DatasetModel):
    """Credible set."""

    dataset_name: ClassVar[str] = 'credible_set'
    dataset_title: ClassVar[str] = 'Credible set'
    dataset_description: ClassVar[str] = (
        '95% credible sets for GWAS and molQTL studies: one row per credible '
        'set, with its study, lead variant, fine-mapping method, confidence '
        'and quality control flags. The variants of each credible set are in '
        'the credible set variant dataset'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Genetics',)
    example: ClassVar[dict] = {
        'id': '00005ed1a798ede3eab60f4a30554bf4',
        'chromosome': '20',
        'locusStart': 44411929,
        'locusEnd': 44411929,
        'studyId': 'GCST90027091',
        'studyType': 'gwas',
        'leadVariantId': '20_44411929_C_T',
        'fineMappingMethod': 'PICS',
        'logBayesFactor': 3.037468040333606,
        'confidence': 1,
        'purityMeanR2': 0.934140222208083,
        'purityMinR2': 0.8653777318957422,
        'qualityControls': ['Lead variant is replicated in an independent study'],
    }

    id: Annotated[str, PrimaryKey()] = Field(
        pattern=MD5_HEX,
        description=(
            'Credible set identifier: the md5 hash of its study, chromosome, locus '
            'bounds, lead variant and fine-mapping method'
        ),
    )
    chromosome: Annotated[Chromosome, PolarsDtype(pl.Enum)] = Field(description='Chromosome of the credible set')
    locusStart: Annotated[int, PolarsDtype(pl.Int32), ClickHouseType('UInt32')] = Field(
        ge=0, description='Start of the fine-mapped region of the credible set'
    )
    locusEnd: Annotated[int, PolarsDtype(pl.Int32), ClickHouseType('UInt32')] = Field(
        ge=0, description='End of the fine-mapped region of the credible set'
    )
    studyId: Annotated[str, Reference('study', 'studyId')] = Field(
        min_length=1, description='Identifier of the study of the credible set'
    )
    studyType: Annotated[StudyType, PolarsDtype(pl.Enum)] = Field(
        description='Type of the study of the credible set (e.g., gwas, eqtl, pqtl)'
    )
    leadVariantId: Annotated[str, Reference('variant', 'variantId')] = Field(
        pattern=VARIANT_ID, description='Lead variant of the credible set'
    )
    fineMappingMethod: Annotated[FineMappingMethod, PolarsDtype(pl.Enum)] = Field(
        description='Method used for fine-mapping of the credible set (PICS, SuSiE or SuSiE-inf)'
    )
    logBayesFactor: float | None = Field(ge=0.0, description='Log Bayes factor of the entire credible set')
    confidence: Annotated[int, PolarsDtype(pl.Int8), ClickHouseType('UInt8')] = Field(
        ge=0,
        description=(
            'Confidence class of the credible set, from its fine-mapping method '
            'and data: 4 fine-mapped with in-sample LD, 3 fine-mapped with '
            'out-of-sample LD, 2 PICS from summary statistics, 1 PICS from a '
            'reported top hit, 0 unknown'
        ),
    )
    purityMeanR2: float | None = Field(
        ge=0.0, description='Mean pairwise LD r2 between the variants of the credible set'
    )
    purityMinR2: float | None = Field(
        ge=0.0, description='Minimum pairwise LD r2 between the variants of the credible set'
    )
    qualityControls: list[Annotated[CredibleSetQualityControl, PolarsDtype(pl.Enum)]] = Field(
        description=('Quality control flags of the credible set, including TRANS_QTL and REPLICATED')
    )
