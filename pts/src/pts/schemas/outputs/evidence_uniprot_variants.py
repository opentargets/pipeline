"""Uniprot variants evidence: the `evidence_uniprot_variants` output.

Generated from the 26.09 release output (34,698 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, Reference
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, RS_ID, SHA1_HEX


class EvidenceUniprotVariants(DatasetModel):
    """Uniprot variants evidence."""

    dataset_name: ClassVar[str] = 'evidence_uniprot_variants'
    dataset_title: ClassVar[str] = 'Uniprot variants evidence'
    dataset_description: ClassVar[str] = (
        'Disease-target evidence based on disease causing protein sequence alteration curated by the Uniprot team'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'id': '05930d2d126e84858d8b300e5ccd19efa5227fe8',
        'targetFromSourceId': 'P82279',
        'diseaseFromSourceMappedId': 'MONDO_0010818',
        'confidence': 'high',
        'datasourceId': 'uniprot_variants',
        'datatypeId': 'genetic_association',
        'diseaseFromSource': 'Retinitis pigmentosa 12',
        'diseaseFromSourceId': 'OMIM:600105',
        'literature': ['22065545'],
        'targetModulation': 'up_or_down',
        'variantId': '1_197427544_C_T',
        'variantRsId': 'rs1664649015',
        'qualityControls': [],
        'diseaseId': 'MONDO_0010818',
        'targetId': 'ENSG00000134376',
        'publicationDate': '2011-12-27',
        'evidenceDate': '2011-12-27',
        'score': 1.0,
    }

    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        min_length=1,
        description=(
            'Target ID in resource of origin (accepted sources include Ensembl '
            'gene ID, Uniprot ID, gene symbol), only capital letters are accepted'
        ),
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    confidence: Literal['high', 'medium'] = Field(description='Confidence qualifier on the reported evidence')
    datasourceId: Literal['uniprot_variants'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['genetic_association'] = Field(description='Type of the evidence')
    diseaseFromSource: str = Field(min_length=1, description='Disease label from the original source')
    diseaseFromSourceId: str = Field(min_length=1, description='Disease identifier from the original source')
    literature: list[str] = Field(min_length=1, description='List of PubMed or preprint reference identifiers')
    targetModulation: Literal['up_or_down'] = Field(description='Description of target modulation event')
    variantId: Annotated[str, Reference('variant', 'variantId')] | None = Field(
        min_length=1, description=('Identifier in CHROM_POS_REF_ALT notation of the disease-causing variant')
    )
    variantRsId: str | None = Field(pattern=RS_ID, description='Variant reference SNP cluster ID (Rsid)')
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
    evidenceDate: str | None = Field(pattern=ISO_DATE, description='Earliest data for the evidence')
    score: float = Field(
        ge=0.5, le=1.0, description=('Score of the evidence reflecting the strength of the disease/target relationship')
    )
