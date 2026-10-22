"""Mouse phenotype: the `mouse_phenotype` output.

Generated from the 26.09 release output (203,583 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
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
        min_length=1, description='References related to the mouse model'
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
        'targetInModel': '1500009L16Rik',
        'targetInModelMgiId': 'MGI:1917034',
        'targetInModelEnsemblId': 'ENSMUSG00000087651',
        'targetFromSourceId': 'ENSG00000235162',
        'modelPhenotypeId': 'MP:0002968',
        'modelPhenotypeLabel': 'increased circulating alkaline phosphatase level',
        'biologicalModels': [
            {
                'allelicComposition': '1500009L16Rik<em1(IMPC)Mbp> hom early',
                'geneticBackground': 'C57BL/6NCrl',
                'id': 'MGI:6276858',
                'literature': ['41239414'],
            }
        ],
        'modelPhenotypeClasses': [{'id': 'MP:0005376', 'label': 'homeostasis/metabolism phenotype'}],
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
        min_length=1, description='Container for all biological model-related attributes.'
    )
    modelPhenotypeClasses: list[ModelPhenotypeClasses] = Field(
        min_length=1, description='Container for phenotype class-related attributes'
    )
