"""Target by region: the `target_by_region` view (pts `target_view`), served as `targets_by_region`.

One row per target: its genomic location and its id, for region lookups.
"""

from typing import ClassVar

from pydantic import Field

from ot_datamodels import DatasetModel, StructModel, from_output
from ot_datamodels.outputs.target import GenomicLocation, Target

RegionChromosome = from_output(GenomicLocation, 'chromosome')
RegionStart = from_output(GenomicLocation, 'start')
RegionEnd = from_output(GenomicLocation, 'end')
RegionTargetId = from_output(Target, 'id')


class TargetByRegionTarget(StructModel):
    """`target`: the target located there."""

    id: RegionTargetId


class TargetByRegion(DatasetModel):
    """Target by region."""

    dataset_name: ClassVar[str] = 'target_by_region'
    dataset_title: ClassVar[str] = 'Target by region'
    dataset_description: ClassVar[str] = 'The genomic location of every target, with its id'
    example: ClassVar[dict] = {
        'chromosome': 'X',
        'start': 100627108,
        'end': 100639991,
        'target': {'id': 'ENSG00000000003'},
    }

    chromosome: RegionChromosome
    start: RegionStart
    end: RegionEnd
    target: TargetByRegionTarget = Field(description='The target at that location')
