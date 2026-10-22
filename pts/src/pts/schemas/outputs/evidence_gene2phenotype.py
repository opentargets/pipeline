"""Gene2Phenotype evidence: the `evidence_gene2phenotype` output.

Generated from the 26.09 release output (4,865 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference
from pts.schemas.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class EvidenceGene2phenotype(DatasetModel):
    """Gene2Phenotype evidence."""

    dataset_name: ClassVar[str] = 'evidence_gene2phenotype'
    dataset_title: ClassVar[str] = 'Gene2Phenotype evidence'
    dataset_description: ClassVar[str] = 'Disease-target evidence curated by the Gene2Phenotype consortium'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000119888',
        'id': '6a8fea8a7fba511483d0297b6dcdd3cc31e5f654',
        'targetFromSourceId': 'EPCAM',
        'diseaseFromSourceMappedId': 'MONDO_0013184',
        'literature': ['18572020'],
        'studyId': 'DD',
        'confidence': 'definitive',
        'allelicRequirements': ['biallelic_autosomal'],
        'diseaseFromSourceId': 'MONDO:0013184',
        'diseaseFromSource': 'diarrhea with tufting enteropathy, congenital',
        'variantFunctionalConsequenceId': 'SO_0002317',
        'datasourceId': 'gene2phenotype',
        'datatypeId': 'genetic_literature',
        'qualityControls': [],
        'diseaseId': 'MONDO_0013184',
        'publicationDate': '2008-05-15',
        'evidenceDate': '2008-05-15',
        'score': 1.0,
        'directionOnTrait': 'risk',
        'directionOnTarget': 'LoF',
    }

    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        min_length=1,
        description=(
            'Target ID in resource of origin (accepted sources include Ensembl '
            'gene ID, Uniprot ID, gene symbol), only capital letters are accepted'
        ),
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        min_length=1, description='List of PubMed or preprint reference identifiers'
    )
    studyId: str = Field(min_length=1, description='Identifier of the study generating the data')
    confidence: Literal['definitive', 'limited', 'moderate', 'strong'] = Field(
        description='Confidence qualifier on the reported evidence'
    )
    allelicRequirements: list[
        Literal[
            'biallelic_autosomal',
            'mitochondrial',
            'monoallelic_X',
            'monoallelic_X_hemizygous',
            'monoallelic_X_heterozygous',
            'monoallelic_Y_hemizygous',
            'monoallelic_autosomal',
        ]
    ] = Field(min_length=1, description='Inheritance patterns')
    diseaseFromSourceId: str | None = Field(min_length=1, description='Disease identifier from the original source')
    diseaseFromSource: str = Field(min_length=1, description='Disease label from the original source')
    variantFunctionalConsequenceId: Annotated[str, Bioregistry('so'), Reference('so', 'id')] | None = Field(
        pattern=ONTOLOGY_ID,
        description=('Sequence ontology (SO) identifier of the functional consequence of the variant'),
    )
    datasourceId: Literal['gene2phenotype'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['genetic_literature'] = Field(description='Type of the evidence')
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    publicationDate: str | None = Field(
        pattern=ISO_DATE, description='Date of the earliest publication supporting the evidence'
    )
    evidenceDate: str | None = Field(pattern=ISO_DATE, description='Earliest data for the evidence')
    score: float = Field(
        ge=0.01,
        le=1.0,
        description=('Score of the evidence reflecting the strength of the disease/target relationship'),
    )
    directionOnTrait: Literal['risk'] = Field(description='Predicted direction of effect on the trait')
    directionOnTarget: Literal['GoF', 'LoF'] | None = Field(description='Effect of direction on target')
