"""Orphanet evidence: the `evidence_orphanet` output.

Generated from the 26.09 release output (6,771 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class EvidenceOrphanet(DatasetModel):
    """Orphanet evidence."""

    dataset_name: ClassVar[str] = 'evidence_orphanet'
    dataset_title: ClassVar[str] = 'Orphanet evidence'
    dataset_description: ClassVar[str] = 'Curated collection of rare disase-target evidence from Orphanet'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000103222',
        'id': '24cebabc9b71ab7b86feb0ddc4cbbdf3a629c8c9',
        'targetFromSourceId': 'ENSG00000103222',
        'diseaseFromSourceMappedId': 'MONDO_0019587',
        'datasourceId': 'orphanet',
        'datatypeId': 'genetic_association',
        'alleleOrigins': ['germline'],
        'confidence': 'Assessed',
        'diseaseFromSource': 'Autosomal dominant non-syndromic genetic deafness',
        'diseaseFromSourceId': 'Orphanet_90635',
        'literature': ['31273342'],
        'targetFromSource': 'ATP binding cassette subfamily C member 1 (ABCC1 blood group)',
        'variantFunctionalConsequenceId': 'SO_0002054',
        'qualityControls': [],
        'diseaseId': 'MONDO_0019587',
        'publicationDate': '2019-07-05',
        'evidenceDate': '2019-07-05',
        'score': 1.0,
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
    datasourceId: Literal['orphanet'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['genetic_association'] = Field(description='Type of the evidence')
    alleleOrigins: list[Literal['germline', 'somatic']] = Field(
        min_length=1, description='Origin of the variant allele'
    )
    confidence: Literal['Assessed', 'Not yet assessed'] = Field(
        description='Confidence qualifier on the reported evidence'
    )
    diseaseFromSource: str = Field(min_length=1, description='Disease label from the original source')
    diseaseFromSourceId: str = Field(pattern=ONTOLOGY_ID, description='Disease identifier from the original source')
    literature: list[Annotated[str, Field(min_length=1)]] | None = Field(
        description='List of PubMed or preprint reference identifiers'
    )
    targetFromSource: str = Field(
        min_length=1, description='Target name/synonym or non HGNC symbol in resource of origin'
    )
    variantFunctionalConsequenceId: Annotated[str, Bioregistry('so'), Reference('so', 'id')] | None = Field(
        pattern=ONTOLOGY_ID,
        description=('Sequence ontology (SO) identifier of the functional consequence of the variant'),
    )
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    publicationDate: str | None = Field(
        pattern=ISO_DATE, description='Date of the earliest publication supporting the evidence'
    )
    evidenceDate: str | None = Field(pattern=ISO_DATE, description='Earliest data for the evidence')
    score: float = Field(
        ge=0.5, le=1.0, description=('Score of the evidence reflecting the strength of the disease/target relationship')
    )
    directionOnTrait: Literal['risk'] = Field(description='Predicted direction of effect on the trait')
    directionOnTarget: Literal['GoF', 'LoF'] | None = Field(
        description=(
            'Gain or loss of function effect of the evidence on the target '
            'resulting from genetic variants, pharmacological modulation, or other '
            'perturbations'
        )
    )
