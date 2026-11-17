"""Literature vector: the `literature_vector` output.

Generated from the 26.12 release output (57,568 rows) and its croissant metadata.
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
        'category': 'disease',
        'word': 'MONDO_0017258',
        'norm': 0.6879864381101375,
        'vector': [-0.011121146380901337],
    }

    category: Literal['disease', 'drug', 'target'] = Field(
        description=('Category of the entity match in the literature. Either target, drug or disease')
    )
    word: Annotated[str, PrimaryKey()] = Field(min_length=1, description='Normalised identifier of the matched entity')
    norm: float = Field(ge=0.0, description='Normalised form of the extracted word')
    vector: list[float] = Field(description='Vector representation of the word based on word embedding')
