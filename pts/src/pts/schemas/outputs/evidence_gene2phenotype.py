"""Gene2Phenotype evidence: the `evidence_gene2phenotype` output.

Generated from the 26.12 release output (4,884 rows) and its croissant metadata by
`scripts/output_schemas/generate.py`, whose docstring lists which observations become rules.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, Reference
from pts.schemas.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class EvidenceGene2phenotype(DatasetModel):
    """Gene2Phenotype evidence."""

    dataset_name: ClassVar[str] = 'evidence_gene2phenotype'
    dataset_title: ClassVar[str] = 'Gene2Phenotype evidence'
    dataset_description: ClassVar[str] = 'Disease-target evidence curated by the Gene2Phenotype consortium'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000136492',
        'id': '5706290c829a1e823b38b397a54f8382d353646d',
        'targetFromSourceId': 'BRIP1',
        'diseaseFromSourceMappedId': 'MONDO_0012187',
        'literature': ['16116424'],
        'studyId': 'DD; Cancer',
        'confidence': 'definitive',
        'allelicRequirements': ['biallelic_autosomal'],
        'diseaseFromSourceId': 'MONDO:0012187',
        'diseaseFromSource': 'Fanconi anemia',
        'variantFunctionalConsequenceId': 'SO_0002317',
        'datasourceId': 'gene2phenotype',
        'datatypeId': 'genetic_literature',
        'qualityControls': [],
        'diseaseId': 'MONDO_0012187',
        'publicationDate': '2005-08-21',
        'evidenceDate': '2005-08-21',
        'score': 1.0,
        'directionOnTrait': 'risk',
        'directionOnTarget': 'LoF',
    }

    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(min_length=1, description='Gene symbol of the target as curated by Gene2Phenotype')
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        description='PubMed identifiers cited by Gene2Phenotype for the gene-disease pair'
    )
    studyId: str = Field(
        min_length=1, description=("Gene2Phenotype panel(s) the gene-disease pair belongs to (e.g. 'DD; Eye')")
    )
    confidence: Literal['definitive', 'limited', 'moderate', 'strong'] = Field(
        description=(
            'Gene2Phenotype confidence category of the gene-disease pair '
            "('definitive', 'strong', 'moderate', 'limited')"
        )
    )
    allelicRequirements: list[Annotated[str, Field(min_length=1)]] = Field(
        description=("Allelic requirement as curated by Gene2Phenotype (e.g. 'biallelic_autosomal')")
    )
    diseaseFromSourceId: str | None = Field(
        min_length=1, description='Disease identifier (e.g. MONDO) as curated by Gene2Phenotype'
    )
    diseaseFromSource: str = Field(min_length=1, description='Disease name as curated by Gene2Phenotype')
    variantFunctionalConsequenceId: Annotated[str, Reference('so', 'id')] | None = Field(
        pattern=ONTOLOGY_ID,
        description=(
            'Sequence ontology (SO) identifier of the molecular mechanism curated '
            "by Gene2Phenotype (e.g. 'absent gene product')"
        ),
    )
    datasourceId: Literal['gene2phenotype'] = Field(description='Identifer of the evidence source')
    datatypeId: str = Field(pattern=ONTOLOGY_ID, description='Type of the evidence')
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
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
            'relationship, derived from the Gene2Phenotype confidence category'
        ),
    )
    directionOnTrait: Literal['risk', 'protect'] = Field(description='Predicted direction of effect on the trait')
    directionOnTarget: Literal['GoF', 'LoF'] | None = Field(
        description=(
            'Gain or loss of function effect of the evidence on the target '
            'resulting from genetic variants, pharmacological modulation, or other '
            'perturbations'
        )
    )
