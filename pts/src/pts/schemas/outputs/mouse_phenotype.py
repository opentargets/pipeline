"""Mouse phenotype: the `mouse_phenotype` output.

Generated from the 26.12 release output (210,729 rows) and its croissant metadata by
`scripts/output_schemas/generate.py`, whose docstring lists which observations become rules.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import DIGITS, ENSEMBL_GENE_ID


class BiologicalModels(StructModel):
    """`biologicalModels[]`."""

    allelicComposition: str = Field(min_length=1, description='The specific allelic composition of the mouse model')
    geneticBackground: str = Field(min_length=1, description='The genetic background strain of the mouse model')
    id: Annotated[str, Bioregistry('mgi')] | None = Field(
        min_length=1, description='Unique identifier for the biological model'
    )
    literature: list[Annotated[Annotated[str, Bioregistry('pubmed')], Field(pattern=DIGITS)]] | None = Field(
        description='References related to the mouse model'
    )


class ModelPhenotypeClasses(StructModel):
    """`modelPhenotypeClasses[]`."""

    id: Annotated[str, Bioregistry('mp')] = Field(min_length=1, description='Unique identifier for the phenotype class')
    label: str = Field(min_length=1, description='Descriptive label for the phenotype class')


class MousePhenotype(DatasetModel):
    """Mouse phenotype."""

    dataset_name: ClassVar[str] = 'mouse_phenotype'
    dataset_title: ClassVar[str] = 'Mouse phenotype'
    dataset_description: ClassVar[str] = 'Phenotypes of mouse models mapped to human targets'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target',)
    example: ClassVar[dict] = {
        'targetInModel': '1110059G10Rik',
        'targetInModelMgiId': 'MGI:1913452',
        'targetInModelEnsemblId': 'ENSMUSG00000032551',
        'targetFromSourceId': 'ENSG00000163807',
        'modelPhenotypeId': 'MP:0003036',
        'modelPhenotypeLabel': 'vertebral transformation',
        'biologicalModels': [
            {
                'allelicComposition': '1110059G10Rik<tm1a(KOMP)Wtsi> hom early',
                'geneticBackground': 'C57BL/6N',
                'id': 'MGI:4419470',
                'literature': ['41239414'],
            }
        ],
        'modelPhenotypeClasses': [{'id': 'MP:0005390', 'label': 'skeleton phenotype'}],
    }

    targetInModel: str = Field(min_length=1, description='Name of the target gene as represented in the mouse model')
    targetInModelMgiId: Annotated[str, Bioregistry('mgi'), PrimaryKey()] = Field(
        min_length=1, description=('Mouse Genome Informatics (MGI) identifier for the target gene in the model')
    )
    targetInModelEnsemblId: Annotated[str, Bioregistry('ensembl')] = Field(
        min_length=1, description='Ensembl identifier for the target gene in the mouse model'
    )
    targetFromSourceId: Annotated[str, Reference('target', 'id'), PrimaryKey()] = Field(
        pattern=ENSEMBL_GENE_ID, description='Identifier for the human target as provided by the data source'
    )
    modelPhenotypeId: Annotated[str, Bioregistry('mp'), PrimaryKey()] = Field(
        min_length=1, description='Identifier for the specific phenotype observed in the model'
    )
    modelPhenotypeLabel: str = Field(min_length=1, description='Human-readable label describing the observed phenotype')
    biologicalModels: list[BiologicalModels] = Field(
        description='Container for all biological model-related attributes.'
    )
    modelPhenotypeClasses: list[ModelPhenotypeClasses] = Field(
        description='Container for phenotype class-related attributes'
    )
