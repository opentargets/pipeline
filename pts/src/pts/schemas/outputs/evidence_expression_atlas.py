"""Differential expression evidence: the `evidence_expression_atlas` output.

Generated from the 26.09 release output (238,857 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, Reference
from pts.schemas.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class EvidenceExpressionAtlas(DatasetModel):
    """Differential expression evidence."""

    dataset_name: ClassVar[str] = 'evidence_expression_atlas'
    dataset_title: ClassVar[str] = 'Differential expression evidence'
    dataset_description: ClassVar[str] = (
        'Disease-target evidence based on differential expression data provided by Expression Atlas'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'id': '004d367fcd96b78e7f6aee4fe97931783d57e8c4',
        'targetFromSourceId': 'ENSG00000144354',
        'diseaseFromSourceMappedId': 'EFO_0003060',
        'biosamplesFromSource': ['UBERON_0002048'],
        'confidence': 'low',
        'contrast': ("'squamous cell carcinoma' vs 'normal tissue adjacent to squamous cell carcinoma'"),
        'datasourceId': 'expression_atlas',
        'datatypeId': 'rna_expression',
        'literature': ['29031501'],
        'log2FoldChangePercentileRank': 99,
        'log2FoldChangeValue': 2.3,
        'resourceScore': 0.025,
        'studyId': 'E-MTAB-5231',
        'studyOverview': 'Gene expression analysis of lung cancer samples and healthy controls',
        'qualityControls': [],
        'diseaseId': 'MONDO_0005233',
        'targetId': 'ENSG00000144354',
        'publicationDate': '2017-10-12',
        'evidenceDate': '2017-10-12',
        'score': 0.0364789060025377,
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
    biosamplesFromSource: (
        list[Annotated[Annotated[str, Reference('biosample', 'biosampleId')], Field(pattern=ONTOLOGY_ID)]] | None
    ) = Field(min_length=1, description='Identifier of the referenced biological material')
    confidence: Literal['high', 'low', 'medium'] = Field(description='Confidence qualifier on the reported evidence')
    contrast: str = Field(min_length=1, description='Experiment contrast')
    datasourceId: Literal['expression_atlas'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['rna_expression'] = Field(description='Type of the evidence')
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        min_length=1, description='List of PubMed or preprint reference identifiers'
    )
    log2FoldChangePercentileRank: int = Field(
        ge=62, le=100, description=('Percentile of top differentially regulated genes (transcripts) within experiment')
    )
    log2FoldChangeValue: float = Field(
        ge=-25.0, le=23.2, description='Log 2 fold expression change in contrast experiment'
    )
    resourceScore: float = Field(
        ge=1e-234,
        le=0.05,
        description=('Score provided by datasource indicating strength of target-disease association'),
    )
    studyId: str = Field(min_length=1, description='Identifier of the study generating the data')
    studyOverview: str = Field(min_length=1, description='Description of the study')
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
        ge=0.010496439238203844,
        le=1.0,
        description=('Score of the evidence reflecting the strength of the disease/target relationship'),
    )
