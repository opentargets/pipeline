"""Disease phenotypes by disease: the `disease_phenotype_by_disease_view` view (pts `disease_phenotype_by_disease`).

One row per disease of `disease_phenotype`, as served in `disease_hpo`: its phenotypes ordered by `phenotype`, each
the output row.
"""

from typing import ClassVar

from pydantic import Field

from ot_datamodels import DatasetModel, from_output, row_of
from ot_datamodels.outputs.disease_phenotype import DiseasePhenotype

Disease = from_output(DiseasePhenotype, 'disease', primary_key=True)
DiseasePhenotypeItem = row_of(DiseasePhenotype, ['disease', 'phenotype', 'evidence'], name='DiseasePhenotypeItem')


class DiseasePhenotypeByDiseaseView(DatasetModel):
    """Disease phenotypes by disease."""

    dataset_name: ClassVar[str] = 'disease_phenotype_by_disease_view'
    dataset_title: ClassVar[str] = 'Disease phenotypes by disease'
    dataset_description: ClassVar[str] = 'Clinical signs and symptoms of each disease, as served by disease'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Disease',)
    example: ClassVar[dict] = {'disease': DiseasePhenotype.example['disease'], 'phenotypes': [DiseasePhenotype.example]}

    disease: Disease
    phenotypes: list[DiseasePhenotypeItem] = Field(min_length=1, description='Phenotypes of the disease, by phenotype')
