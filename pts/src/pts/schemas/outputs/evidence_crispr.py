"""ProjectScore evidence: the `evidence_crispr` output.

Generated from the 26.12 release output (517 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class DiseaseCellLines(StructModel):
    """`diseaseCellLines[]`."""

    id: str = Field(min_length=1, description='Cell Model Passports identifier of the cell line (e.g. SIDM00616)')
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
        'id': '77f3392e64d64406aa67f85ae199bf79ade71c1b',
        'targetFromSourceId': 'ENSG00000213281',
        'diseaseFromSourceMappedId': 'EFO_0000756',
        'targetFromSource': 'NRAS',
        'diseaseFromSource': 'melanoma',
        'resourceScore': 83.80000305175781,
        'literature': ['38215750'],
        'datasourceId': 'crispr',
        'datatypeId': 'affected_pathway',
        'diseaseCellLines': [{'id': 'SIDM01793', 'name': 'Mel-285', 'tissue': 'eye', 'tissueId': 'UBERON_0000970'}],
        'qualityControls': [],
        'diseaseId': 'MONDO_0005105',
        'targetId': 'ENSG00000213281',
        'publicationDate': '2024-01-11',
        'evidenceDate': '2024-01-11',
        'score': 0.8380000305175781,
    }

    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        pattern=ENSEMBL_GENE_ID, description='Ensembl gene identifier of the target as reported by Project Score'
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    targetFromSource: str = Field(min_length=1, description='Gene symbol of the target as reported by Project Score')
    diseaseFromSource: str = Field(min_length=1, description='Cancer type as reported by Project Score')
    resourceScore: float = Field(ge=0.0, description='Project Score target priority score (0-100)')
    literature: list[Annotated[str, Field(pattern=DIGITS)]] = Field(
        description='PubMed identifier of the Project Score publication'
    )
    datasourceId: Literal['crispr'] = Field(description='Identifer of the evidence source')
    datatypeId: str = Field(pattern=ONTOLOGY_ID, description='Type of the evidence')
    diseaseCellLines: list[DiseaseCellLines] = Field(
        description='Cancer cell lines in which the target was prioritised by Project Score'
    )
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    publicationDate: str = Field(
        pattern=ISO_DATE, description='Date of the earliest publication supporting the evidence'
    )
    evidenceDate: str = Field(pattern=ISO_DATE, description='Earliest data for the evidence')
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            'Score of the evidence reflecting the strength of the disease/target '
            'relationship; the Project Score priority score divided by 100, with a '
            'floor of 0.415'
        ),
    )
