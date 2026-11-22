"""Drug/Target - pharmacogenetics: the `pharmacogenomics` output.

Generated from the 26.12 release output (33,485 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from ot_datamodels import Bioregistry, ClickHouseType, DatasetModel, PolarsDtype, Reference, StructModel
from ot_datamodels.outputs.patterns import CHEMBL_ID, DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, RS_ID, VARIANT_ID
from ot_datamodels.vocabularies import (
    PgxCategory,
    PgxEffectType,
    PgxEvidenceLevel,
    PharmacogenomicsDatasource,
    PharmacogenomicsDatatype,
)


class VariantAnnotation(StructModel):
    """`variantAnnotation[]`."""

    baseAlleleOrGenotype: str = Field(min_length=1, description='Reference allele or genotype in comparison')
    comparisonAlleleOrGenotype: str | None = Field(
        min_length=1, description='Allele or genotype the variant effect is interpreted on'
    )
    directionality: str | None = Field(min_length=1, description='Direction of effect of the allele or genotype')
    effect: str | None = Field(min_length=1, description='Pharmacogenomic effect of the variation')
    effectDescription: str = Field(min_length=1, description='Summary of the impact of the allele on the drug response')
    effectType: Annotated[PgxEffectType, PolarsDtype(pl.Enum)] = Field(
        description='Type of effect (eg. functional, drug or phenotype)'
    )
    entity: str | None = Field(min_length=1, description='Entity affected by the effect')
    id: str = Field(pattern=DIGITS, description='Identifier of the variant annotation at source')
    literature: str = Field(
        pattern=DIGITS, description='Literature reference of the variant annotation supporting the evidence'
    )


class Drugs(StructModel):
    """`drugs[]`."""

    drugFromSource: Annotated[str, ClickHouseType('LowCardinality(String)')] = Field(
        min_length=1, description='List of drug identifiers from the original data source'
    )
    drugId: (
        Annotated[
            str, Bioregistry('chembl'), Reference('drug_molecule', 'id'), ClickHouseType('LowCardinality(String)')
        ]
        | None
    ) = Field(pattern=CHEMBL_ID, description='List of Open Targets drug identifiers')


class Pharmacogenomics(DatasetModel):
    """Drug/Target - pharmacogenetics."""

    dataset_name: ClassVar[str] = 'pharmacogenomics'
    dataset_title: ClassVar[str] = 'Drug/Target - pharmacogenetics'
    dataset_description: ClassVar[str] = (
        'Genetic variants influencing individual drug responses. '
        'Pharmacogenetics data is integrated from sources including ClinPGx'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target', 'Genetics')
    example: ClassVar[dict] = {
        'datasourceId': 'clinpgx',
        'datasourceVersion': '2026-07-05',
        'datatypeId': 'clinical_annotation',
        'directionality': 'Normal function',
        'evidenceLevel': '3',
        'genotype': 'CC',
        'genotypeAnnotationText': (
            'Patients with the CC genotype may have decreased chance of response '
            'to risperidone as compared to patients with the TT genotype. Other '
            'genetic and clinical factors may also influence a patients chance of '
            'response to risperidone.'
        ),
        'genotypeId': '1_163063901_C_C,C',
        'haplotypeFromSourceId': 'PA165816576',
        'haplotypeId': 'CYP2D6*1',
        'literature': ['17588543'],
        'pgxCategory': 'efficacy',
        'phenotypeFromSourceId': 'HP_0005184',
        'phenotypeText': 'decreased chance of response to risperidone',
        'variantAnnotation': [
            {
                'baseAlleleOrGenotype': 'TT',
                'comparisonAlleleOrGenotype': 'CC + CT',
                'directionality': 'increased',
                'effect': 'likelihood of',
                'effectDescription': (
                    'Genotype TT is associated with increased likelihood of response when '
                    'treated with risperidone in people with Schizophrenia as compared to '
                    'genotypes CC + CT.'
                ),
                'effectType': 'phenotype',
                'entity': 'response',
                'id': '655387128',
                'literature': '17588543',
            }
        ],
        'studyId': '655384602',
        'targetFromSourceId': 'ENSG00000117152',
        'variantFunctionalConsequenceId': 'SO_0002073',
        'variantRsId': 'rs951439',
        'variantId': '1_163063901_C_T',
        'isDirectTarget': False,
        'drugs': [{'drugFromSource': 'risperidone', 'drugId': 'CHEMBL85'}],
    }

    datasourceId: Annotated[PharmacogenomicsDatasource, PolarsDtype(pl.Enum)] = Field(
        description='Identifier for the data provider'
    )
    datasourceVersion: str = Field(pattern=ISO_DATE, description='Data provider version number')
    datatypeId: Annotated[PharmacogenomicsDatatype, PolarsDtype(pl.Enum)] = Field(
        description='Classification of the type of pharmacogenomic data (e.g. clinical_annotation)'
    )
    directionality: str | None = Field(
        min_length=1, description=('Indicates wether the genetic variant increases or decreases drug response')
    )
    evidenceLevel: Annotated[PgxEvidenceLevel, PolarsDtype(pl.Enum)] = Field(
        description='Strength of the scientific support for the variant/drug response'
    )
    genotype: str = Field(min_length=1, description='Genetic variant configuration')
    genotypeAnnotationText: str = Field(min_length=1, description="Explanation of the genotype's clinical significance")
    genotypeId: str | None = Field(
        min_length=1, description=('Identifier for the specific genetic variant combination (e.g. 1_1500_A_A,T)')
    )
    haplotypeFromSourceId: str | None = Field(min_length=1, description='Haplotype ID in the ClinPGx dataset')
    haplotypeId: str | None = Field(
        min_length=1,
        description=('Combination of genetic variants that constitute a particular allele of a gene (e.g. CYP2C9*3)'),
    )
    literature: list[Annotated[str, Field(pattern=DIGITS)]] = Field(
        description='List of PubMed or preprint reference identifiers'
    )
    pgxCategory: Annotated[PgxCategory, PolarsDtype(pl.Enum)] = Field(
        description='Classification of the drug response type (e.g. Toxicity)'
    )
    phenotypeFromSourceId: Annotated[str, Reference('disease', 'id')] | None = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier result of mapping phenotypeText'
    )
    phenotypeText: str | None = Field(min_length=1, description='Drug response influenced by the variation')
    variantAnnotation: list[VariantAnnotation] | None = Field(description='Metadata on the underlying variation')
    studyId: str = Field(pattern=DIGITS, description='Clinical Annotation ID in the ClinPGx dataset')
    targetFromSourceId: Annotated[str, Reference('target', 'id')] | None = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier as provided by the data source'
    )
    variantFunctionalConsequenceId: Annotated[str, Bioregistry('so'), Reference('so', 'id')] | None = Field(
        pattern=ONTOLOGY_ID,
        description=(
            'The sequence ontology identifier of the consequence of the variant '
            'based on Ensembl VEP in the context of the transcript'
        ),
    )
    variantRsId: str | None = Field(pattern=RS_ID, description='Variant reference SNP cluster ID (Rsid)')
    variantId: Annotated[str, Reference('variant', 'variantId')] | None = Field(
        pattern=VARIANT_ID,
        description=(
            'The unique identifier for the variant following schema: '
            '{chromosome}-{position}-{referenceAllele}-{alternateAllele}'
        ),
    )
    isDirectTarget: bool = Field(
        description=('Flag indicating whether the target overlapping the variation is directly targeted by the drug')
    )
    drugs: list[Drugs] = Field(description='List of drug references')
