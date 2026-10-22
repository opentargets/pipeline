"""Literature vector: the `literature_vector` output.

Generated from the 26.09 release output (56,295 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey


class LiteratureVector(DatasetModel):
    """Literature vector."""

    dataset_name: ClassVar[str] = 'literature_vector'
    dataset_title: ClassVar[str] = 'Literature vector'
    dataset_description: ClassVar[str] = 'Word to Vector representation of entity matches in literature'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Literature',)
    example: ClassVar[dict] = {
        'category': 'drug',
        'word': 'CHEMBL3976807',
        'norm': 0.40075366301369325,
        'vector': [-0.052375517785549164],
    }

    category: Literal['disease', 'drug', 'target'] = Field(
        description=('Category of the entity match in the literature. Either target, drug or disease')
    )
    word: Annotated[str, PrimaryKey()] = Field(min_length=1, description='Normalised identifier of the matched entity')
    norm: float = Field(
        ge=0.026338259093323026, le=5.249216492433747, description='Normalised form of the extracted word'
    )
    vector: list[Annotated[float, Field(ge=-1.5422346591949463, le=1.7609695196151733)]] = Field(
        min_length=1, description='Vector representation of the word based on word embedding'
    )
