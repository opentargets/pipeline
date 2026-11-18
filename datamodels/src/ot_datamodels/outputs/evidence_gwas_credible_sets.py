"""GWAS credible-set evidence: the `evidence_gwas_credible_sets` output.

Generated from the 26.12 release output (3,462,426 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from ot_datamodels import DatasetModel, PrimaryKey, Reference
from ot_datamodels.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, MD5_HEX, ONTOLOGY_ID, SHA1_HEX


class EvidenceGwasCredibleSets(DatasetModel):
    """GWAS credible-set evidence."""

    dataset_name: ClassVar[str] = 'evidence_gwas_credible_sets'
    dataset_title: ClassVar[str] = 'GWAS credible-set evidence'
    dataset_description: ClassVar[str] = (
        'Disease-target evidence based on Locus-to-Gene prediction on GWAS credible sets'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'id': '185289a7dbdeb11045056e499b4b7c8421922d3c',
        'targetFromSourceId': 'ENSG00000092470',
        'diseaseFromSourceMappedId': 'OBA_VT0010046',
        'datatypeId': 'genetic_association',
        'datasourceId': 'gwas_credible_sets',
        'resourceScore': 0.05137481540441513,
        'curationDate': '2026-05-20',
        'studyLocusId': 'fda139b06ba3b449f9e71f296a99d715',
        'literature': ['42162431'],
        'qualityControls': [],
        'diseaseId': 'OBA_VT0010046',
        'targetId': 'ENSG00000092470',
        'publicationDate': '2026-05-20',
        'evidenceDate': '2026-05-20',
        'score': 0.05137481540441513,
    }

    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        pattern=ENSEMBL_GENE_ID,
        description=('Ensembl gene identifier of the target prioritised by the locus-to-gene model'),
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    datatypeId: str = Field(pattern=ONTOLOGY_ID, description='Type of the evidence')
    datasourceId: Literal['gwas_credible_sets'] = Field(description='Identifer of the evidence source')
    resourceScore: float = Field(
        ge=0.0, le=1.0, description='Locus-to-gene (L2G) score of the target for the GWAS credible set'
    )
    curationDate: str | None = Field(pattern=ISO_DATE, description='Date the GWAS study was curated')
    studyLocusId: Annotated[str, Reference('credible_set', 'id')] = Field(
        pattern=MD5_HEX, description=('Identifier of the GWAS credible set the locus-to-gene prediction was made for')
    )
    literature: list[Annotated[str, Field(pattern=DIGITS)]] = Field(
        description='PubMed identifier of the publication of the GWAS study'
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
        ge=0.0,
        le=1.0,
        description=(
            'Score of the evidence reflecting the strength of the disease/target '
            'relationship; equal to the locus-to-gene (L2G) score'
        ),
    )
