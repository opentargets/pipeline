"""Protein coding coordinates by target: the `coding_variant_by_target_view` view (pts `coding_variant`).

One row per target with an amino acid changing transcript consequence of a variant with evidence: those
consequences, each with the diseases, therapeutic areas and datasources behind its variant.
"""

from typing import ClassVar

from pydantic import Field

from ot_datamodels import DatasetModel, from_output
from ot_datamodels.outputs.variant import TranscriptConsequences
from ot_datamodels.serving import Serving
from ot_datamodels.views.coding_variant_by_variant_view import COORD_EXAMPLE, ProteinCodingCoord

TargetKey = from_output(TranscriptConsequences, 'targetId', primary_key=True)


class CodingVariantByTargetView(DatasetModel):
    """Protein coding coordinates by target."""

    dataset_name: ClassVar[str] = 'coding_variant_by_target_view'
    dataset_title: ClassVar[str] = 'Protein coding coordinates by target'
    dataset_description: ClassVar[str] = (
        'Amino acid changing consequences on each target of variants with evidence, with the diseases, '
        'therapeutic areas and datasources behind each variant, as served by target'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Genetics',)
    clickhouse: ClassVar[Serving] = Serving(
        engine='rocksdb',
        column_types={
            'proteinCodingCoords[]/variantEffect': 'Float64',
            'proteinCodingCoords[]/datasources[]/datasourceCount': 'UInt32',
        },
    )
    example: ClassVar[dict] = {'targetId': 'ENSG00000141510', 'proteinCodingCoords': [COORD_EXAMPLE]}

    targetId: TargetKey
    proteinCodingCoords: list[ProteinCodingCoord] = Field(
        description=(
            'Consequences on the target, by variantId, aminoAcidPosition, referenceAminoAcid, alternateAminoAcid, '
            'variantEffect, variantFunctionalConsequenceIds, uniprotAccessions'
        )
    )
