"""Open Targets Projects: the `otar` output.

Generated from the 26.09 release output (365 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import ONTOLOGY_ID


class Projects(StructModel):
    """`projects[]`."""

    otar_code: str = Field(min_length=1, description='OTAR project identifier')
    status: Literal['Active', 'Closed'] = Field(description='Status of the project indicating completeness')
    project_name: str = Field(min_length=1, description='Name of the project')
    integrates_data_PPP: bool = Field(
        description=('Boolean flag indicating if the project is providing data to the Partner Preview Platform')
    )
    reference: str = Field(min_length=1, description='URL pointing to the project homepage')


class Otar(DatasetModel):
    """Open Targets Projects."""

    dataset_name: ClassVar[str] = 'otar'
    dataset_title: ClassVar[str] = 'Open Targets Projects'
    dataset_description: ClassVar[str] = 'Open Targets projects grouped by studied disease or phenotype'
    dataset_tags: ClassVar[tuple[str, ...]] = ()
    example: ClassVar[dict] = {
        'efo_id': 'EFO_0000651',
        'projects': [
            {
                'otar_code': 'OTAR015',
                'status': 'Closed',
                'project_name': 'CRISPR-Cas9 Target ID',
                'integrates_data_PPP': True,
                'reference': 'http://home.opentargets.org/OTAR015',
            }
        ],
    }

    efo_id: Annotated[str, Reference('disease', 'id'), PrimaryKey()] = Field(
        pattern=ONTOLOGY_ID, description='Disease identifier studied by a list OTAR projects'
    )
    projects: list[Projects] = Field(min_length=1, description='List of OTAR projects studying a given disease')
