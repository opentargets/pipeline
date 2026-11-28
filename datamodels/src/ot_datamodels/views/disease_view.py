"""Disease view: the `disease_view` view (pts `disease_view`).

Every disease row in the served column order, with its synonyms as relation/terms pairs and the
studies mapped to it and to its descendants.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from ot_datamodels import DatasetModel, StructModel
from ot_datamodels.derive import from_output
from ot_datamodels.outputs.disease import Disease, Ontology
from ot_datamodels.serving import ClickHouseDictionary, Serving

# a list element takes no description, so not `from_output`; tests/test_views.py checks it equals studyId
_StudyId = Annotated[str, Field(min_length=1)]

Id = from_output(Disease, 'id', primary_key=True)
Name = from_output(Disease, 'name')
TherapeuticAreas = from_output(Disease, 'therapeuticAreas')
Description = from_output(Disease, 'description')
DbXRefs = from_output(Disease, 'dbXRefs')
ObsoleteTerms = from_output(Disease, 'obsoleteTerms')
Parents = from_output(Disease, 'parents')
Children = from_output(Disease, 'children')
Ancestors = from_output(Disease, 'ancestors')
Descendants = from_output(Disease, 'descendants')
IsTherapeuticArea = from_output(Ontology, 'isTherapeuticArea')


class DiseaseSynonym(StructModel):
    """`synonyms[]`."""

    relation: str = Field(
        description='Synonym relation: hasExactSynonym, hasRelatedSynonym, hasNarrowSynonym or hasBroadSynonym'
    )
    terms: list[str | None] = Field(description='The terms that are synonymous with the disease')


_EXAMPLE = Disease.example


class DiseaseView(DatasetModel):
    """Disease view."""

    dataset_name: ClassVar[str] = 'disease_view'
    dataset_title: ClassVar[str] = 'Disease view'
    dataset_description: ClassVar[str] = (
        'Diseases or phenotypes as served, with their synonyms by relation and the studies mapped to '
        'each disease and to its descendants'
    )
    clickhouse: ClassVar[Serving] = Serving(
        engine='rocksdb',
        dictionaries=(ClickHouseDictionary('disease_names', key='id', attributes=('name',)),),
        column_types={
            'therapeuticAreas[]': 'LowCardinality(String)',
            'parents[]': 'LowCardinality(String)',
            'ancestors[]': 'LowCardinality(String)',
        },
    )
    example: ClassVar[dict] = {
        **{
            c: _EXAMPLE[c]
            for c in (
                'id',
                'name',
                'therapeuticAreas',
                'description',
                'dbXRefs',
                'obsoleteTerms',
            )
        },
        'synonyms': [{'relation': r, 'terms': t} for r, t in _EXAMPLE['synonyms'].items()],
        **{c: _EXAMPLE[c] for c in ('parents', 'children', 'ancestors', 'descendants')},
        'isTherapeuticArea': _EXAMPLE['ontology']['isTherapeuticArea'],
        'studyIds': ['GCST90201095'],
        'indirectStudyIds': ['GCST90201095'],
    }

    id: Id
    name: Name
    therapeuticAreas: TherapeuticAreas
    description: Description
    dbXRefs: DbXRefs
    obsoleteTerms: ObsoleteTerms
    synonyms: list[DiseaseSynonym] = Field(description='List of synonymous disease labels, by relation')
    parents: Parents
    children: Children
    ancestors: Ancestors
    descendants: Descendants
    isTherapeuticArea: IsTherapeuticArea
    studyIds: list[_StudyId] = Field(description='Studies mapped to the disease')
    indirectStudyIds: list[_StudyId] = Field(description="Studies mapped to the disease's descendants")
