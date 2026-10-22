"""Disease/Phenotype: the `disease` output.

Generated from the 26.09 release output (45,896 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, StructModel
from pts.schemas.outputs.patterns import ONTOLOGY_ID


class Sources(StructModel):
    """`ontology/sources`."""

    url: str = Field(min_length=1, description='Disease URL from source')
    name: str = Field(pattern=ONTOLOGY_ID, description='Lable of the term in the ontology')


class Ontology(StructModel):
    """`ontology`."""

    isTherapeuticArea: bool = Field(
        description=('Boolean column indicating if a disease is root of the ontology tree, a therapeutic area')
    )
    leaf: bool = Field(description=('Boolean column indicating if a disease is a leaf entity in the ontology tree'))
    sources: Sources = Field(description='Source of the ontology')


class Synonyms(StructModel):
    """`synonyms`."""

    hasExactSynonym: list[Annotated[str, Field(min_length=1)]] = Field(description='List of exact synonyms')
    hasRelatedSynonym: list[Annotated[str, Field(min_length=1)]] = Field(description='List of related synonyms')
    hasNarrowSynonym: list[Annotated[str, Field(min_length=1)]] = Field(description='List of closely related sysnonyms')
    hasBroadSynonym: list[Annotated[str, Field(min_length=1)]] = Field(description='List of broadly related sysnonyms')


class Disease(DatasetModel):
    """Disease/Phenotype."""

    dataset_name: ClassVar[str] = 'disease'
    dataset_title: ClassVar[str] = 'Disease/Phenotype'
    dataset_description: ClassVar[str] = (
        'Core annotation for diseases or phenotypes. A disease or phenotype in '
        'the Platform is understood as any disease, phenotype, biological '
        'process or measurement that might have any type of causality '
        'relationship with a human target. The EMBL-EBI Experimental Factor '
        'Ontology (EFO) (slim version) is used as scaffold for the disease or '
        'phenotype entity'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Disease',)
    example: ClassVar[dict] = {
        'id': 'GO_0000050',
        'code': 'http://purl.obolibrary.org/obo/GO_0000050',
        'name': 'urea cycle',
        'description': (
            'The sequence of reactions by which arginine is synthesized from '
            'ornithine, then cleaved to yield urea and regenerate ornithine. The '
            'overall reaction equation is NH3 + CO2 + aspartate + 3 ATP + 2 H2O = '
            'urea + fumarate + 2 ADP + 2 phosphate + AMP + diphosphate.'
        ),
        'dbXRefs': ['Reactome:R-HSA-70635'],
        'parents': ['GO_0009058'],
        'exactSynonyms': ['ornithine cycle'],
        'relatedSynonyms': ['microtubule cytoskeleton organization and biogenesis'],
        'narrowSynonyms': ['inhibition of leukocyte activation'],
        'broadSynonyms': ['energy pathways'],
        'obsoleteTerms': ['GO_0033490'],
        'obsoleteXRefs': ['MESH:D007249'],
        'children': ['GO_0009087'],
        'ancestors': ['GO_0009058'],
        'therapeuticAreas': ['GO_0008150'],
        'descendants': ['GO_0009087'],
        'ontology': {
            'isTherapeuticArea': False,
            'leaf': False,
            'sources': {'url': 'http://purl.obolibrary.org/obo/GO_0000050', 'name': 'GO_0000050'},
        },
        'synonyms': {
            'hasExactSynonym': ['ornithine cycle'],
            'hasRelatedSynonym': ['microtubule cytoskeleton organization and biogenesis'],
            'hasNarrowSynonym': ['inhibition of leukocyte activation'],
            'hasBroadSynonym': ['energy pathways'],
        },
    }

    id: Annotated[str, PrimaryKey()] = Field(pattern=ONTOLOGY_ID, description='Open Targets disease identifier')
    code: str = Field(min_length=1, description='Disease reference URL')
    name: str = Field(min_length=1, description='Name of the disease')
    description: str | None = Field(min_length=1, description='Description of the disease')
    dbXRefs: list[Annotated[str, Field(min_length=1)]] = Field(
        description='Cross-references in other disease ontologies'
    )
    parents: list[Annotated[str, Field(pattern=ONTOLOGY_ID)]] = Field(description='List of parent disease terms')
    exactSynonyms: list[Annotated[str, Field(min_length=1)]] = Field(description='List of exact synonyms')
    relatedSynonyms: list[Annotated[str, Field(min_length=1)]] = Field(description='List of related synonyms')
    narrowSynonyms: list[Annotated[str, Field(min_length=1)]] = Field(description='List of closely related sysnonyms')
    broadSynonyms: list[Annotated[str, Field(min_length=1)]] = Field(description='List of broadly related sysnonyms')
    obsoleteTerms: list[Annotated[str, Field(pattern=ONTOLOGY_ID)]] = Field(description='List of obsoleted terms')
    obsoleteXRefs: list[Annotated[str, Field(min_length=1)]] = Field(description='List of obsoleted cross references')
    children: list[Annotated[str, Field(pattern=ONTOLOGY_ID)]] = Field(description='List of children disease terms')
    ancestors: list[Annotated[str, Field(pattern=ONTOLOGY_ID)]] = Field(
        description='List of all ancestral disease terms'
    )
    therapeuticAreas: list[Annotated[str, Field(pattern=ONTOLOGY_ID)]] = Field(
        min_length=1, description='List of major therapeutic areas the disease belongs to'
    )
    descendants: list[Annotated[str, Field(pattern=ONTOLOGY_ID)]] = Field(description='List of all descendant terms')
    ontology: Ontology = Field(description="Description of the disease's place in the ontology tree")
    synonyms: Synonyms = Field(description='List of synonymous disease labels')
