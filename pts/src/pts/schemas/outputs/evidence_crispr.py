"""ProjectScore evidence: the `evidence_crispr` output.

Generated from the 26.09 release output (517 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, ONTOLOGY_ID, SHA1_HEX


class DiseaseCellLines(StructModel):
    """`diseaseCellLines[]`."""

    id: str = Field(min_length=1, description='Cell type identifier in cell ontology or in cell model database')
    name: str = Field(min_length=1, description='Name of the cell model')
    tissue: str = Field(min_length=1, description='Name of the tissue from which the cells were sampled')
    tissueId: Annotated[str, Bioregistry('uberon'), Reference('biosample', 'biosampleId')] = Field(
        pattern=ONTOLOGY_ID, description='Anatomical identifier of the sampled organ/tissue'
    )


class EvidenceCrispr(DatasetModel):
    """ProjectScore evidence."""

    dataset_name: ClassVar[str] = 'evidence_crispr'
    dataset_title: ClassVar[str] = 'ProjectScore evidence'
    dataset_description: ClassVar[str] = (
        'Disease-target evidence from unified analysis of whole-genome CRISPR/Cas9 screens from ProjectScore project'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'id': '57240fdbe4c74449abe1b5ff3cf56974cf3fd4fb',
        'targetFromSourceId': 'ENSG00000092531',
        'diseaseFromSourceMappedId': 'EFO_0000182',
        'targetFromSource': 'SNAP23',
        'diseaseFromSource': 'hepatocellular carcinoma',
        'resourceScore': 44.79999923706055,
        'literature': ['38215750'],
        'datasourceId': 'crispr',
        'datatypeId': 'affected_pathway',
        'diseaseCellLines': [{'id': 'SIDM01439', 'name': 'SNU-761', 'tissue': 'liver', 'tissueId': 'UBERON_0002107'}],
        'qualityControls': [],
        'diseaseId': 'MONDO_0007256',
        'targetId': 'ENSG00000092531',
        'publicationDate': '2024-01-11',
        'evidenceDate': '2024-01-11',
        'score': 0.44799999237060545,
    }

    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        pattern=ENSEMBL_GENE_ID,
        description=(
            'Target ID in resource of origin (accepted sources include Ensembl '
            'gene ID, Uniprot ID, gene symbol), only capital letters are accepted'
        ),
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    targetFromSource: str = Field(
        min_length=1, description='Target name/synonym or non HGNC symbol in resource of origin'
    )
    diseaseFromSource: str = Field(min_length=1, description='Disease label from the original source')
    resourceScore: float = Field(
        ge=36.0,
        le=83.80000305175781,
        description=('Score provided by datasource indicating strength of target-disease association'),
    )
    literature: list[Literal['38215750']] = Field(
        min_length=1, description='List of PubMed or preprint reference identifiers'
    )
    datasourceId: Literal['crispr'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['affected_pathway'] = Field(description='Type of the evidence')
    diseaseCellLines: list[DiseaseCellLines] = Field(
        min_length=1, description='Cancer cell lines used to generate evidence'
    )
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    publicationDate: Literal['2024-01-11'] = Field(
        description='Date of the earliest publication supporting the evidence'
    )
    evidenceDate: Literal['2024-01-11'] = Field(description='Earliest data for the evidence')
    score: float = Field(
        ge=0.415,
        le=0.8380000305175781,
        description=('Score of the evidence reflecting the strength of the disease/target relationship'),
    )
