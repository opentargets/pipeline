"""Differential expression evidence: the `evidence_expression_atlas` output.

Generated from the 26.12 release output (238,857 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from ot_datamodels import DatasetModel, PrimaryKey, Reference
from ot_datamodels.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX
from ot_datamodels.vocabularies import Datasource, Datatype, enum_of


class EvidenceExpressionAtlas(DatasetModel):
    """Differential expression evidence."""

    dataset_name: ClassVar[str] = 'evidence_expression_atlas'
    dataset_title: ClassVar[str] = 'Differential expression evidence'
    dataset_description: ClassVar[str] = (
        'Disease-target evidence based on differential expression data provided by Expression Atlas'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'id': '3f1cac313db12ecff8e3da5d643c9c022c5d7ec9',
        'targetFromSourceId': 'ENSG00000118113',
        'diseaseFromSourceMappedId': 'MONDO_0005113',
        'biosamplesFromSource': ['UBERON_0000178'],
        'confidence': 'high',
        'contrast': "'bacterial disease' vs 'normal'",
        'datasourceId': 'expression_atlas',
        'datatypeId': 'rna_expression',
        'literature': ['37255317'],
        'log2FoldChangePercentileRank': 99,
        'log2FoldChangeValue': 10.0,
        'resourceScore': 8.38e-08,
        'studyId': 'E-CURD-149',
        'studyOverview': (
            'Diagnosis of multisystem inflammatory syndrome in children (MIS-C) by '
            'a whole-blood transcriptional signature - differential analysis'
        ),
        'qualityControls': [],
        'diseaseId': 'MONDO_0005113',
        'targetId': 'ENSG00000118113',
        'publicationDate': '2023-06-01',
        'evidenceDate': '2023-06-01',
        'score': 0.7005988421556026,
    }

    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        pattern=ENSEMBL_GENE_ID, description='Ensembl gene identifier of the target as reported by Expression Atlas'
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    biosamplesFromSource: (
        list[Annotated[Annotated[str, Reference('biosample', 'biosampleId')], Field(pattern=ONTOLOGY_ID)]] | None
    ) = Field(description=('Identifiers of the tissues or cell types profiled in the Expression Atlas experiment'))
    confidence: str = Field(
        min_length=1,
        description=(
            "Confidence of the differential expression assigned by Expression Atlas ('low', 'medium', 'high')"
        ),
    )
    contrast: str = Field(
        min_length=1,
        description=(
            "Expression Atlas comparison between disease and reference samples (e.g. \"'sporadic ALS' vs 'normal'\")"
        ),
    )
    datasourceId: Annotated[Literal['expression_atlas'], enum_of(Datasource)] = Field(
        description='Identifer of the evidence source'
    )
    datatypeId: Annotated[Literal['rna_expression'], enum_of(Datatype)] = Field(description='Type of the evidence')
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        description=('PubMed identifiers of the publications describing the Expression Atlas experiment')
    )
    log2FoldChangePercentileRank: int = Field(
        ge=0,
        le=2147483647,
        description=(
            'Percentile rank of the absolute log 2 fold change of the target among '
            'the differentially expressed genes of the contrast'
        ),
    )
    log2FoldChangeValue: float = Field(
        description=('Log 2 fold change of the target expression between the disease and the reference samples')
    )
    resourceScore: float = Field(
        ge=0.0, le=1.0, description='P-value of the differential expression of the target in the contrast'
    )
    studyId: str = Field(min_length=1, description='Expression Atlas experiment accession (e.g. E-MTAB-1238)')
    studyOverview: str = Field(min_length=1, description='Title of the Expression Atlas experiment')
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
        ge=0.0,
        le=1.0,
        description=(
            'Score of the evidence reflecting the strength of the disease/target '
            'relationship, combining the p-value, the log 2 fold change and its '
            'percentile rank'
        ),
    )
