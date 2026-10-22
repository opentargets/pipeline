"""IMPC evidence: the `evidence_impc` output.

Generated from the 26.09 release output (6,911,070 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
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
        'targetId': 'ENSG00000125965',
        'id': '000515759568498d1e5370d217149960aa85cd84',
        'targetFromSourceId': 'ENSG00000125965',
        'diseaseFromSourceMappedId': 'MONDO_0015229',
        'targetInModelMgiId': 'MGI:95688',
        'diseaseFromSourceId': 'OMIM:615986',
        'biologicalModelGeneticBackground': 'involves: BALB/cJ',
        'biologicalModelAllelicComposition': 'Gdf5<bp-3J>/Gdf5<bp-3J>',
        'diseaseFromSource': 'Bardet-Biedl Syndrome 9',
        'resourceScore': 46.29,
        'targetInModel': 'Gdf5',
        'targetInModelEnsemblId': 'ENSMUSG00000038259',
        'diseaseModelAssociatedModelPhenotypes': [{'id': 'MP:0004686', 'label': 'decreased length of long bones'}],
        'diseaseModelAssociatedHumanPhenotypes': [{'id': 'HP:0100259', 'label': 'Postaxial polydactyly'}],
        'literature': ['10631181'],
        'biologicalModelId': 'MGI:2168885',
        'datasourceId': 'impc',
        'datatypeId': 'animal_model',
        'qualityControls': [],
        'diseaseId': 'MONDO_0015229',
        'publicationDate': '1977-07-01',
        'evidenceDate': '1977-07-01',
        'score': 0.4629,
        'directionOnTrait': 'risk',
        'directionOnTarget': 'LoF',
    }

    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        pattern=ENSEMBL_GENE_ID,
        description=(
            'Target ID in resource of origin (accepted sources include Ensembl '
            'gene ID, Uniprot ID, gene symbol), only capital letters are accepted'
        ),
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    targetInModelMgiId: Annotated[str, Bioregistry('mgi')] = Field(
        min_length=1, description='Target MGI ID in animal model'
    )
    diseaseFromSourceId: str = Field(min_length=1, description='Disease identifier from the original source')
    biologicalModelGeneticBackground: str = Field(min_length=1, description='Genetic background of the model organism')
    biologicalModelAllelicComposition: str = Field(
        min_length=1, description='Allelic composition of the model organism'
    )
    diseaseFromSource: str = Field(min_length=1, description='Disease label from the original source')
    resourceScore: float = Field(
        ge=41.0,
        le=100.0,
        description=('Score provided by datasource indicating strength of target-disease association'),
    )
    targetInModel: str = Field(min_length=1, description='Target name/synonym in animal model')
    targetInModelEnsemblId: Annotated[str, Bioregistry('ensembl')] = Field(
        min_length=1, description='Target Ensembl ID in animal model'
    )
    diseaseModelAssociatedModelPhenotypes: list[DiseaseModelAssociatedModelPhenotypes] = Field(
        min_length=1, description='Phenotypes observed in genetically modified animal models'
    )
    diseaseModelAssociatedHumanPhenotypes: list[DiseaseModelAssociatedHumanPhenotypes] | None = Field(
        min_length=1, description='Human phenotypes equivalent to those observed in animal models'
    )
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        min_length=1, description='List of PubMed or preprint reference identifiers'
    )
    biologicalModelId: str | None = Field(min_length=1, description='Identifier of the biological model (eg. in MGI)')
    datasourceId: Literal['impc'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['animal_model'] = Field(description='Type of the evidence')
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    publicationDate: str | None = Field(
        pattern=ISO_DATE, description='Date of the earliest publication supporting the evidence'
    )
    evidenceDate: str | None = Field(pattern=ISO_DATE, description='Earliest data for the evidence')
    score: float = Field(
        ge=0.41,
        le=1.0,
        description=('Score of the evidence reflecting the strength of the disease/target relationship'),
    )
    directionOnTrait: Literal['risk'] = Field(description='Predicted direction of effect on the trait')
    directionOnTarget: Literal['LoF'] = Field(
        description=(
            'Gain or loss of function effect of the evidence on the target '
            'resulting from genetic variants, pharmacological modulation, or other '
            'perturbations'
        )
    )
