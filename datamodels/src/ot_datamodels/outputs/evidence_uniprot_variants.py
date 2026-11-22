"""Uniprot variants evidence: the `evidence_uniprot_variants` output.

Generated from the 26.12 release output (34,765 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from ot_datamodels import ClickHouseType, DatasetModel, PrimaryKey, Reference
from ot_datamodels.outputs.patterns import ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, RS_ID, SHA1_HEX
from ot_datamodels.vocabularies import Datasource, Datatype, enum_of


class EvidenceUniprotVariants(DatasetModel):
    """Uniprot variants evidence."""

    dataset_name: ClassVar[str] = 'evidence_uniprot_variants'
    dataset_title: ClassVar[str] = 'Uniprot variants evidence'
    dataset_description: ClassVar[str] = (
        'Disease-target evidence based on disease causing protein sequence alteration curated by the Uniprot team'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'id': 'e3ec11e6fd2913cfc2f06e48a4a55d4c50cfe253',
        'targetFromSourceId': 'P35914',
        'diseaseFromSourceMappedId': 'MONDO_0009520',
        'confidence': 'high',
        'datasourceId': 'uniprot_variants',
        'datatypeId': 'genetic_association',
        'diseaseFromSource': '3-hydroxy-3-methylglutaryl-CoA lyase deficiency',
        'diseaseFromSourceId': 'OMIM:246450',
        'literature': ['17173698'],
        'targetModulation': 'up_or_down',
        'variantId': '1_23820532_C_T',
        'variantRsId': 'rs121964997',
        'qualityControls': [],
        'diseaseId': 'MONDO_0009520',
        'targetId': 'ENSG00000117305',
        'publicationDate': '1998-02-01',
        'evidenceDate': '1998-02-01',
        'score': 1.0,
    }

    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(min_length=1, description='UniProt accession of the target')
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    confidence: Literal['high', 'medium'] = Field(
        description=("Confidence of the UniProt curated gene-disease association ('high', 'medium')")
    )
    datasourceId: Annotated[Literal['uniprot_variants'], enum_of(Datasource)] = Field(
        description='Identifer of the evidence source'
    )
    datatypeId: Annotated[Literal['genetic_association'], enum_of(Datatype)] = Field(description='Type of the evidence')
    diseaseFromSource: str = Field(min_length=1, description='Disease name as curated by UniProt')
    diseaseFromSourceId: str = Field(min_length=1, description='OMIM identifier of the disease as curated by UniProt')
    literature: list[str] = Field(description='PubMed identifiers cited by UniProt for the disease-causing variant')
    targetModulation: str = Field(
        min_length=1, description=("Effect of the disease on the target; 'up_or_down' as UniProt does not specify it")
    )
    variantId: Annotated[str, Reference('variant', 'variantId')] | None = Field(
        min_length=1,
        description=('Identifier in CHROM_POS_REF_ALT notation of the disease-causing variant curated by UniProt'),
    )
    variantRsId: str | None = Field(
        pattern=RS_ID, description=('Reference SNP cluster ID (rsID) of the disease-causing variant curated by UniProt')
    )
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id'), ClickHouseType('LowCardinality(String)')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    targetId: Annotated[str, Reference('target', 'id'), ClickHouseType('LowCardinality(String)')] = Field(
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
            'relationship, derived from the UniProt confidence'
        ),
    )
