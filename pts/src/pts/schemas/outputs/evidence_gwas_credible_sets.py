"""GWAS credible-set evidence: the `evidence_gwas_credible_sets` output.

Generated from the 26.09 release output (3,467,408 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, Reference
from pts.schemas.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, MD5_HEX, ONTOLOGY_ID, SHA1_HEX


class EvidenceGwasCredibleSets(DatasetModel):
    """GWAS credible-set evidence."""

    dataset_name: ClassVar[str] = 'evidence_gwas_credible_sets'
    dataset_title: ClassVar[str] = 'GWAS credible-set evidence'
    dataset_description: ClassVar[str] = (
        'Disease-target evidence based on Locus-to-Gene prediction on GWAS credible sets'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'id': '000105495d2dc078f1ecf8341a3baceccf6db2ea',
        'targetFromSourceId': 'ENSG00000077498',
        'diseaseFromSourceMappedId': 'MONDO_0021454',
        'datatypeId': 'genetic_association',
        'datasourceId': 'gwas_credible_sets',
        'resourceScore': 0.1055402159690857,
        'curationDate': '2024-07-19',
        'studyLocusId': '925bb486bd43aaf8f289510335454395',
        'literature': ['39024449'],
        'qualityControls': [],
        'diseaseId': 'MONDO_0021454',
        'targetId': 'ENSG00000077498',
        'publicationDate': '2024-07-19',
        'evidenceDate': '2024-07-19',
        'score': 0.1055402159690857,
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
    datatypeId: Literal['genetic_association'] = Field(description='Type of the evidence')
    datasourceId: Literal['gwas_credible_sets'] = Field(description='Identifer of the evidence source')
    resourceScore: float = Field(
        ge=0.050000015646219254,
        le=0.9978274703025818,
        description=('Score provided by datasource indicating strength of target-disease association'),
    )
    curationDate: str | None = Field(pattern=ISO_DATE, description='Date of the GWAS study curation')
    studyLocusId: Annotated[str, Reference('credible_set', 'studyLocusId')] = Field(
        pattern=MD5_HEX, description='Identifier of the Open Targets Study Locus'
    )
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        min_length=1, description='List of PubMed or preprint reference identifiers'
    )
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    publicationDate: str | None = Field(
        pattern=ISO_DATE, description='Date of the earliest publication supporting the evidence'
    )
    evidenceDate: str = Field(pattern=ISO_DATE, description='Earliest data for the evidence')
    score: float = Field(
        ge=0.050000015646219254,
        le=0.9978274703025818,
        description=('Score of the evidence reflecting the strength of the disease/target relationship'),
    )
