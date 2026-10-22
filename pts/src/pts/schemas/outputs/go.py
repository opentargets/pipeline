"""Target - gene ontology: the `go` output.

Generated from the 26.09 release output (48,340 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey


class Go(DatasetModel):
    """Target - gene ontology."""

    dataset_name: ClassVar[str] = 'go'
    dataset_title: ClassVar[str] = 'Target - gene ontology'
    dataset_description: ClassVar[str] = (
        'This dataset contains the mapping between Gene Ontology (GO) '
        'identifiers and their terms, including hierarchy, relationships, and '
        'metadata'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Ontology',)
    example: ClassVar[dict] = {
        'id': 'GO:0000001',
        'label': 'mitochondrion inheritance',
        'namespace': 'biological_process',
        'altIds': ['GO:0000013'],
        'isA': ['GO:0048308'],
        'partOf': ['GO:0005829'],
        'regulates': ['GO:0006310'],
        'negativelyRegulates': ['GO:0006366'],
        'positivelyRegulates': ['GO:0000747'],
        'isObsolete': False,
        'ancestors': ['GO:0006996'],
        'descendants': ['GO:0033164'],
    }

    id: Annotated[str, Bioregistry('go'), PrimaryKey()] = Field(
        min_length=1, description='Primary Gene Ontology term identifier'
    )
    label: str = Field(min_length=1, description='Human-readable name of the GO term')
    namespace: Literal['biological_process', 'cellular_component', 'molecular_function'] = Field(
        description=(
            'Ontology branch the term belongs to: biological_process, molecular_function, or cellular_component'
        )
    )
    altIds: list[Annotated[str, Field(min_length=1)]] = Field(
        description='List of alternative/secondary GO IDs that are synonyms for this term'
    )
    isA: list[Annotated[str, Field(min_length=1)]] = Field(
        description=("List of parent GO term IDs representing direct 'is a' hierarchical relationships")
    )
    partOf: list[Annotated[str, Field(min_length=1)]] = Field(
        description="List of GO term IDs to which this term has a 'part of' relationship"
    )
    regulates: list[Annotated[str, Field(min_length=1)]] = Field(
        description=('List of GO term IDs that this term regulates (generic regulatory relationship)')
    )
    negativelyRegulates: list[Annotated[str, Field(min_length=1)]] = Field(
        description=('List of GO term IDs that this term negatively regulates (inhibitory relationship)')
    )
    positivelyRegulates: list[Annotated[str, Field(min_length=1)]] = Field(
        description=('List of GO term IDs that this term positively regulates (activating relationship)')
    )
    isObsolete: bool = Field(
        description=('Boolean flag indicating whether the GO term has been deprecated and is no longer in active use')
    )
    ancestors: list[Annotated[str, Field(min_length=1)]] = Field(description='List of ancestor GO terms')
    descendants: list[Annotated[str, Field(min_length=1)]] = Field(description='List of descendant GO terms')
