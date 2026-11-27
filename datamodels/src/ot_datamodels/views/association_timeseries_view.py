"""Association timeseries: the `association_timeseries_view` view (pts `association_timeseries_view`).

The overall and by-datasource associations, direct and indirect, flattened to one row per association,
aggregation value and year of its timeseries.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from ot_datamodels import DatasetModel, from_output
from ot_datamodels.outputs.association_overall_direct import AssociationOverallDirect, Timeseries
from ot_datamodels.serving import Serving
from ot_datamodels.vocabularies import AggregationType, enum_of

# the four inputs agree on these, but by-datasource yearlyEvidenceCount is Int32, cast up (tests/test_views.py)
DiseaseId = from_output(AssociationOverallDirect, 'diseaseId')
TargetId = from_output(AssociationOverallDirect, 'targetId')
Year = from_output(Timeseries, 'year')
AssociationScore = from_output(Timeseries, 'associationScore')
Novelty = from_output(Timeseries, 'novelty')
YearlyEvidenceCount = from_output(Timeseries, 'yearlyEvidenceCount')


class AssociationTimeseriesView(DatasetModel):
    """Association timeseries."""

    dataset_name: ClassVar[str] = 'association_timeseries_view'
    dataset_title: ClassVar[str] = 'Association timeseries'
    dataset_description: ClassVar[str] = (
        'Overall and by data source association scores, direct and indirect, one row per year of their timeseries'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    products: ClassVar[tuple[str, ...]] = ('ppp',)
    clickhouse: ClassVar[Serving] = Serving(
        engine='mergetree',
        order_by=('diseaseId', 'targetId', 'isDirect', 'aggregationValue', 'year'),
        column_types={'year': 'UInt16', 'yearlyEvidenceCount': 'UInt32'},
    )
    example: ClassVar[dict] = {
        'diseaseId': 'EFO_0000180',
        'targetId': 'ENSG00000009413',
        'aggregationType': 'datasourceId',
        'aggregationValue': 'europepmc',
        'year': 2026,
        'associationScore': 0.0014800995840213117,
        'novelty': 0.00017643219529906104,
        'yearlyEvidenceCount': 1,
        'isDirect': True,
    }

    diseaseId: DiseaseId
    targetId: TargetId
    aggregationType: Annotated[Literal['overall', 'datasourceId'], enum_of(AggregationType)] = Field(
        description='How evidence data was aggregated: overall or datasourceId'
    )
    aggregationValue: str = Field(
        min_length=1, description="The datasource aggregated by; the string 'None' for the overall association"
    )
    year: Year
    associationScore: AssociationScore
    novelty: Novelty
    yearlyEvidenceCount: YearlyEvidenceCount
    isDirect: bool = Field(description='Whether the association is direct')
