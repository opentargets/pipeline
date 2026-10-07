"""Fields every evidence datasource shares, and the identifier shapes they use.

Descriptions are copied verbatim from the croissant `evidence_*.json` recordsets. Fields whose
definition differs between datasources (`datasourceId`, `datatypeId`, `targetFromSourceId`,
`evidenceDate`) are declared by each dataset instead of overridden here, so no dataset silently
inherits a rule meant for another.
"""

from typing import Annotated, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference

#: The quality-control flag vocabulary, mirroring `pts.transformers.evidence.utils.flags` (kept as
#: a literal here so schemas do not import transformers; a test keeps the two in sync).
QualityFlag = Literal[
    'No valid disease',
    'No valid target',
    'Duplicated',
    'No valid score',
    'Invalid biotype',
]

#: Ensembl gene ID.
ENSEMBL_GENE_ID = r'^ENSG\d{11}$'

#: Ontology-style ID: a namespace prefix, underscore, code -- EFO_/MONDO_/Orphanet_/HP_ and so on.
ONTOLOGY_ID = r'^[A-Za-z]+_[A-Za-z0-9]+$'

#: sha1 hex digest, as produced by `pts.transformers.evidence.utils.identifiers`.
SHA1_HEX = r'^[0-9a-f]{40}$'

#: An ISO `YYYY-MM-DD` date, matching the whole value.
ISO_DATE = r'^\d{4}-\d{2}-\d{2}$'


class EvidenceBase(DatasetModel):
    """Shared base of every evidence dataset. Not a dataset itself: it sets no `dataset_name`."""

    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID,
        description='Open Targets disease identifier',
    )
    targetId: Annotated[str, Reference('target', 'id'), Bioregistry('ensembl')] = Field(
        pattern=ENSEMBL_GENE_ID,
        description='Open Targets target identifier',
    )
    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    qualityControls: list[QualityFlag] = Field(description='Evidence quality flags')
    score: float = Field(
        ge=0.0,
        le=1.0,
        description='Score of the evidence reflecting the strength of the disease/target relationship',
    )
