"""Protein coding coordinates by variant: the `coding_variant_by_variant_view` view (pts `coding_variant`).

One row per variant with evidence and an amino acid changing transcript consequence: its consequences, each with
the diseases, therapeutic areas and datasources behind the variant.
"""

from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from ot_datamodels import Bioregistry, DatasetModel, PolarsDtype, Reference, StructModel, from_output
from ot_datamodels.outputs.patterns import ONTOLOGY_ID
from ot_datamodels.outputs.variant import TranscriptConsequences, Variant, VariantEffect
from ot_datamodels.serving import Serving
from ot_datamodels.vocabularies import CodingVariantDatasource

TargetId = from_output(TranscriptConsequences, 'targetId')
VariantId = from_output(Variant, 'variantId')
VariantKey = from_output(Variant, 'variantId', primary_key=True)
VariantEffectScore = from_output(VariantEffect, 'score', nullable=True)
VariantFunctionalConsequenceIds = from_output(TranscriptConsequences, 'variantFunctionalConsequenceIds')

_DiseaseId = Annotated[str, Reference('disease', 'id'), Field(pattern=ONTOLOGY_ID)]


class ProteinCodingCoordDatasource(StructModel):
    """`proteinCodingCoords[].datasources[]`."""

    datasourceCount: Annotated[int, PolarsDtype(pl.Int32)] = Field(
        ge=0, description='Number of distinct evidence of the datasource on the variant'
    )
    datasourceId: CodingVariantDatasource = Field(description='Identifier of the datasource')
    datasourceNiceName: str = Field(min_length=1, description='Display name of the datasource')


class ProteinCodingCoord(StructModel):
    """`proteinCodingCoords[]`: one amino acid changing transcript consequence of a variant with evidence."""

    targetId: TargetId
    uniprotAccessions: list[Annotated[str, Bioregistry('uniprot'), Field(min_length=1)]] = Field(
        description=(
            "UniProt accessions of the consequence, intersected with the target's SwissProt accessions when "
            'both are known; by length, then value'
        )
    )
    aminoAcidPosition: Annotated[int, PolarsDtype(pl.Int32)] = Field(
        description='Position of the amino acid change in the protein'
    )
    alternateAminoAcid: str = Field(description='Alternate amino acid of the change, empty when it has none')
    referenceAminoAcid: str = Field(description='Reference amino acid of the change, empty when it has none')
    variantFunctionalConsequenceIds: VariantFunctionalConsequenceIds
    variantEffect: VariantEffectScore
    variantId: VariantId
    diseases: list[_DiseaseId] = Field(
        description='Diseases with evidence on the variant, by descending harmonic sum of their evidence scores'
    )
    datasources: list[ProteinCodingCoordDatasource] = Field(
        description='Datasources with evidence on the variant, by datasourceId'
    )
    therapeuticAreas: list[_DiseaseId] = Field(
        description='Therapeutic areas of the diseases with evidence on the variant, sorted'
    )


COORD_EXAMPLE = {
    'targetId': 'ENSG00000141510',
    'uniprotAccessions': ['P04637'],
    'aminoAcidPosition': 248,
    'alternateAminoAcid': 'Q',
    'referenceAminoAcid': 'R',
    'variantFunctionalConsequenceIds': ['SO_0001583'],
    'variantEffect': 0.98,
    'variantId': '17_7674220_C_T',
    'diseases': ['MONDO_0007903'],
    'datasources': [{'datasourceCount': 2, 'datasourceId': 'eva', 'datasourceNiceName': 'ClinVar'}],
    'therapeuticAreas': ['MONDO_0045024'],
}


class CodingVariantByVariantView(DatasetModel):
    """Protein coding coordinates by variant."""

    dataset_name: ClassVar[str] = 'coding_variant_by_variant_view'
    dataset_title: ClassVar[str] = 'Protein coding coordinates by variant'
    dataset_description: ClassVar[str] = (
        'Amino acid changing consequences of each variant with evidence, with the diseases, therapeutic areas '
        'and datasources behind the variant, as served by variant'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Genetics',)
    clickhouse: ClassVar[Serving] = Serving(
        engine='rocksdb',
        column_types={
            'proteinCodingCoords[]/variantEffect': 'Float64',
            'proteinCodingCoords[]/datasources[]/datasourceCount': 'UInt32',
        },
    )
    example: ClassVar[dict] = {'variantId': '17_7674220_C_T', 'proteinCodingCoords': [COORD_EXAMPLE]}

    variantId: VariantKey
    proteinCodingCoords: list[ProteinCodingCoord] = Field(
        description=(
            'Consequences of the variant, by targetId, aminoAcidPosition, referenceAminoAcid, alternateAminoAcid, '
            'variantEffect, variantFunctionalConsequenceIds, uniprotAccessions'
        )
    )
