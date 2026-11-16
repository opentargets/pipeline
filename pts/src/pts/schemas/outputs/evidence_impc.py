"""IMPC evidence: the `evidence_impc` output.

Generated from the 26.12 release output (4,659,348 rows) and its croissant metadata by
`scripts/output_schemas/generate.py`, whose docstring lists which observations become rules.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class DiseaseModelAssociatedModelPhenotypes(StructModel):
    """`diseaseModelAssociatedModelPhenotypes[]`."""

    id: Annotated[str, Bioregistry('mp'), Reference('mouse_phenotype', 'modelPhenotypeId')] = Field(
        min_length=1, description='Mouse phenotype ontology identifier'
    )
    label: str = Field(min_length=1, description='Mouse phenotype ontology label')


class DiseaseModelAssociatedHumanPhenotypes(StructModel):
    """`diseaseModelAssociatedHumanPhenotypes[]`."""

    id: Annotated[str, Bioregistry('hp'), Reference('disease_hpo', 'id')] = Field(
        min_length=1, description='Human phenotype ontology identifier'
    )
    label: str = Field(min_length=1, description='Human phenotype ontology label')


class EvidenceImpc(DatasetModel):
    """IMPC evidence."""

    dataset_name: ClassVar[str] = 'evidence_impc'
    dataset_title: ClassVar[str] = 'IMPC evidence'
    dataset_description: ClassVar[str] = (
        'Disease-target evidence based on mouse knock-out models, where mouse '
        'phenotypes are mapped to human diseases by PhenoDigm algorithm. Data '
        'provided by IMPC'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000008710',
        'id': '28f52b6dd09a02a2224b6796de00d85583c26562',
        'targetFromSourceId': 'ENSG00000008710',
        'diseaseFromSourceMappedId': 'MONDO_0054743',
        'targetInModelMgiId': 'MGI:97603',
        'diseaseFromSourceId': 'OMIM:617874',
        'biologicalModelGeneticBackground': 'involves: 129S4/SvJaeSor',
        'biologicalModelAllelicComposition': 'Pkd1<tm1Rsa>/Pkd1<+>',
        'diseaseFromSource': 'Polycystic Liver Disease 3 With Or Without Kidney Cysts',
        'resourceScore': 100.0,
        'targetInModel': 'Pkd1',
        'targetInModelEnsemblId': 'ENSMUSG00000032855',
        'diseaseModelAssociatedModelPhenotypes': [{'id': 'MP:0003327', 'label': 'liver cyst'}],
        'diseaseModelAssociatedHumanPhenotypes': [{'id': 'HP:0000107', 'label': 'Renal cyst'}],
        'literature': ['10677526'],
        'biologicalModelId': 'MGI:2173244',
        'datasourceId': 'impc',
        'datatypeId': 'animal_model',
        'qualityControls': [],
        'diseaseId': 'MONDO_0054743',
        'publicationDate': '1997-10-01',
        'evidenceDate': '1997-10-01',
        'score': 1.0,
        'directionOnTrait': 'risk',
        'directionOnTarget': 'LoF',
    }

    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        pattern=ENSEMBL_GENE_ID, description='Ensembl gene identifier of the human ortholog of the mouse gene'
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    targetInModelMgiId: str = Field(min_length=1, description='MGI identifier of the modified gene in the mouse model')
    diseaseFromSourceId: str = Field(
        min_length=1,
        description=('OMIM or Orphanet identifier of the human disease whose phenotypes the mouse model resembles'),
    )
    biologicalModelGeneticBackground: str = Field(
        min_length=1, description='Genetic background (strain) of the mouse model'
    )
    biologicalModelAllelicComposition: str = Field(min_length=1, description='Allelic composition of the mouse model')
    diseaseFromSource: str = Field(
        min_length=1, description='Name of the human disease whose phenotypes the mouse model resembles'
    )
    resourceScore: float = Field(
        ge=0.0,
        description=(
            'PhenoDigm score of the phenotypic similarity between the mouse model and the human disease (0-100)'
        ),
    )
    targetInModel: str = Field(min_length=1, description='Symbol of the modified gene in the mouse model')
    targetInModelEnsemblId: str = Field(
        min_length=1, description='Ensembl gene identifier of the modified gene in the mouse model'
    )
    diseaseModelAssociatedModelPhenotypes: list[DiseaseModelAssociatedModelPhenotypes] = Field(
        description=(
            'Phenotypes observed in the mouse model (all annotated model '
            'phenotypes, not only those matched to the human disease)'
        )
    )
    diseaseModelAssociatedHumanPhenotypes: list[DiseaseModelAssociatedHumanPhenotypes] | None = Field(
        description=('Phenotypes of the human disease matched to the mouse model phenotypes by PhenoDigm')
    )
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        description='List of PubMed or preprint reference identifiers'
    )
    biologicalModelId: str | None = Field(min_length=1, description='MGI identifier of the mouse model')
    datasourceId: Literal['impc'] = Field(description='Identifer of the evidence source')
    datatypeId: str = Field(pattern=ONTOLOGY_ID, description='Type of the evidence')
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    publicationDate: str | None = Field(
        pattern=ISO_DATE, description='Date of the earliest publication supporting the evidence'
    )
    evidenceDate: str | None = Field(pattern=ISO_DATE, description='Earliest data for the evidence')
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            'Score of the evidence reflecting the strength of the disease/target '
            'relationship; the PhenoDigm score divided by 100'
        ),
    )
    directionOnTrait: Literal['risk', 'protect'] = Field(description='Predicted direction of effect on the trait')
    directionOnTarget: Literal['GoF', 'LoF'] = Field(
        description=(
            'Gain or loss of function effect of the evidence on the target '
            'resulting from genetic variants, pharmacological modulation, or other '
            'perturbations'
        )
    )
