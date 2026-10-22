"""Genomics England - PanelApp evidence: the `evidence_genomics_england` output.

Generated from the 26.09 release output (42,315 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, Reference
from pts.schemas.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class EvidenceGenomicsEngland(DatasetModel):
    """Genomics England - PanelApp evidence."""

    dataset_name: ClassVar[str] = 'evidence_genomics_england'
    dataset_title: ClassVar[str] = 'Genomics England - PanelApp evidence'
    dataset_description: ClassVar[str] = 'Curated disease-target evidence provided by Genomics England PanelApp'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'id': '0497a29e801130f0762b47f4f9ce000f6778b777',
        'targetFromSourceId': 'WFS1',
        'diseaseFromSourceMappedId': 'MONDO_0005148',
        'studyId': '26',
        'studyOverview': ('Diabetes with additional phenotypes suggestive of a monogenic aetiology'),
        'confidence': 'green',
        'cohortPhenotypes': ['Wolfram syndrome, OMIM:222300'],
        'diseaseFromSource': 'Diabetes mellitus, noninsulin-dependent, association with',
        'diseaseFromSourceId': 'OMIM:125853',
        'literature': ['27185633'],
        'allelicRequirements': ['BOTH monoallelic and biallelic, autosomal or pseudoautosomal'],
        'datasourceId': 'genomics_england',
        'datatypeId': 'genetic_literature',
        'qualityControls': [],
        'diseaseId': 'MONDO_0005148',
        'targetId': 'ENSG00000109501',
        'publicationDate': '2016-05-14',
        'evidenceDate': '2016-05-14',
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
    studyId: str = Field(pattern=DIGITS, description='Identifier of the study generating the data')
    studyOverview: str = Field(min_length=1, description='Description of the study')
    confidence: Literal['amber', 'green'] = Field(description='Confidence qualifier on the reported evidence')
    cohortPhenotypes: list[str] = Field(
        min_length=1, description='Clinical features/phenotypes observed in studied individuals'
    )
    diseaseFromSource: str | None = Field(min_length=1, description='Disease label from the original source')
    diseaseFromSourceId: str | None = Field(min_length=1, description='Disease identifier from the original source')
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        min_length=1, description='List of PubMed or preprint reference identifiers'
    )
    allelicRequirements: (
        list[
            Literal[
                'BIALLELIC, autosomal or pseudoautosomal',
                (
                    'BOTH monoallelic and biallelic (but BIALLELIC mutations cause a more '
                    'SEVERE disease form), autosomal or pseudoautosomal'
                ),
                'BOTH monoallelic and biallelic, autosomal or pseudoautosomal',
                'MITOCHONDRIAL',
                'MONOALLELIC, autosomal or pseudoautosomal, NOT imprinted',
                'MONOALLELIC, autosomal or pseudoautosomal, imprinted status unknown',
                ('MONOALLELIC, autosomal or pseudoautosomal, maternally imprinted (paternal allele expressed)'),
                ('MONOALLELIC, autosomal or pseudoautosomal, paternally imprinted (maternal allele expressed)'),
                'Other',
                'Other - please specifiy in evaluation comments',
                'Unknown',
                'X-LINKED: hemizygous mutation in males, biallelic mutations in females',
                (
                    'X-LINKED: hemizygous mutation in males, monoallelic mutations in '
                    'females may cause disease (may be less severe, later onset than males)'
                ),
            ]
        ]
        | None
    ) = Field(min_length=1, description='Inheritance patterns')
    datasourceId: Literal['genomics_england'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['genetic_literature'] = Field(description='Type of the evidence')
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
