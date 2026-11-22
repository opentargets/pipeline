"""Coding variants: the `coding_variant` view (pts `coding_variant`).

One row per amino acid changing transcript consequence of a variant with evidence, with the diseases,
therapeutic areas and datasources behind the variant.
"""

from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from ot_datamodels import Bioregistry, ClickHouseType, DatasetModel, PolarsDtype, Reference, StructModel, from_output
from ot_datamodels.outputs.patterns import ONTOLOGY_ID
from ot_datamodels.outputs.variant import TranscriptConsequences, Variant, VariantEffect
from ot_datamodels.vocabularies import CodingVariantDatasource

TargetId = from_output(TranscriptConsequences, 'targetId')
VariantId = from_output(Variant, 'variantId')
VariantEffectScore = from_output(VariantEffect, 'score', nullable=True)
VariantFunctionalConsequenceIds = from_output(TranscriptConsequences, 'variantFunctionalConsequenceIds')

_DiseaseId = Annotated[str, Reference('disease', 'id'), Field(pattern=ONTOLOGY_ID)]


class Datasource(StructModel):
    """`datasources[]`."""

    datasourceCount: Annotated[int, PolarsDtype(pl.Int32), ClickHouseType('UInt32')] = Field(
        ge=0, description='Number of distinct evidence of the datasource on the variant'
    )
    datasourceId: Annotated[CodingVariantDatasource, PolarsDtype(pl.Enum)] = Field(
        description='Identifier of the datasource'
    )
    datasourceNiceName: Annotated[str, ClickHouseType('LowCardinality(String)')] = Field(
        min_length=1, description='Display name of the datasource'
    )


class CodingVariant(DatasetModel):
    """Coding variants."""

    dataset_name: ClassVar[str] = 'coding_variant'
    dataset_title: ClassVar[str] = 'Coding variants'
    dataset_description: ClassVar[str] = (
        'Amino acid changing consequences of variants with evidence, with the diseases, therapeutic areas '
        'and datasources behind each variant'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Genetics',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000141510',
        'variantId': '17_7674220_C_T',
        'therapeuticAreas': ['MONDO_0045024'],
        'diseases': ['MONDO_0007903'],
        'datasources': [{'datasourceCount': 2, 'datasourceId': 'eva', 'datasourceNiceName': 'ClinVar'}],
        'variantEffect': 0.98,
        'variantFunctionalConsequenceIds': ['SO_0001583'],
        'uniprotAccessions': ['P04637'],
        'referenceAminoAcid': 'R',
        'aminoAcidPosition': 248,
        'alternateAminoAcid': 'Q',
    }

    targetId: TargetId
    variantId: VariantId
    therapeuticAreas: list[_DiseaseId] = Field(
        description='Therapeutic areas of the diseases with evidence on the variant, sorted'
    )
    diseases: list[_DiseaseId] = Field(
        description='Diseases with evidence on the variant, by descending harmonic sum of their evidence scores'
    )
    datasources: list[Datasource] = Field(description='Datasources with evidence on the variant, by datasourceId')
    variantEffect: VariantEffectScore
    variantFunctionalConsequenceIds: VariantFunctionalConsequenceIds
    uniprotAccessions: list[Annotated[str, Bioregistry('uniprot'), Field(min_length=1)]] = Field(
        description=(
            "UniProt accessions of the consequence, intersected with the target's SwissProt accessions when "
            'both are known; by length, then value'
        )
    )
    referenceAminoAcid: Annotated[str, ClickHouseType('LowCardinality(String)')] = Field(
        description='Reference amino acid of the change, empty when it has none'
    )
    aminoAcidPosition: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        description='Position of the amino acid change in the protein'
    )
    alternateAminoAcid: Annotated[str, ClickHouseType('LowCardinality(String)')] = Field(
        description='Alternate amino acid of the change, empty when it has none'
    )
