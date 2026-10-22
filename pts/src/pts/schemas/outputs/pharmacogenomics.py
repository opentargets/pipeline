"""Drug/Target - pharmacogenetics: the `pharmacogenomics` output.

Generated from the 26.09 release output (33,479 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, Reference, StructModel
from pts.schemas.outputs.patterns import CHEMBL_ID, DIGITS, ENSEMBL_GENE_ID, ONTOLOGY_ID, RS_ID, VARIANT_ID


class VariantAnnotation(StructModel):
    """`variantAnnotation[]`."""

    baseAlleleOrGenotype: str = Field(min_length=1, description='Reference allele or genotype in comparison')
    comparisonAlleleOrGenotype: str | None = Field(
        min_length=1, description='Allele or genotype the variant effect is interpreted on'
    )
    directionality: Literal['decreased', 'increased'] | None = Field(
        description='Direction of effect of the allele or genotype'
    )
    effect: str | None = Field(min_length=1, description='Pharmacogenomic effect of the variation')
    effectDescription: str = Field(min_length=1, description='Summary of the impact of the allele on the drug response')
    effectType: Literal['drug', 'functional', 'phenotype'] = Field(
        description='Type of effect (eg. functional, drug or phenotype)'
    )
    entity: str | None = Field(min_length=1, description='Entity affected by the effect')
    id: str = Field(pattern=DIGITS, description='Identifier of the variant annotation at source')
    literature: str = Field(
        pattern=DIGITS, description='Literature reference of the variant annotation supporting the evidence'
    )


class Drugs(StructModel):
    """`drugs[]`."""

    drugFromSource: str = Field(min_length=1, description='List of drug identifiers from the original data source')
    drugId: Annotated[str, Bioregistry('chembl'), Reference('drug_molecule', 'id')] | None = Field(
        pattern=CHEMBL_ID, description='List of Open Targets drug identifiers'
    )


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
        'directionality': 'Decreased function',
        'evidenceLevel': '3',
        'genotype': '*10',
        'genotypeAnnotationText': (
            'Patients carrying the *10 allele in combination with a no function or '
            'another decreased function allele may have an increased risk of '
            'experiencing sedation when treated with postoperative tramadol and '
            'ketoprofen as compared to patients with two normal function alleles '
            'or a normal function and a decreased function allele. Other genetic '
            "and clinical factors may also affect a patient's risk of sedation "
            'when treated with tramadol and ketoprofen.'
        ),
        'genotypeId': '10_133537858_C_T,T',
        'haplotypeFromSourceId': 'PA165816582',
        'haplotypeId': 'CYP2D6*10',
        'literature': ['25989235'],
        'pgxCategory': 'toxicity',
        'phenotypeFromSourceId': 'MONDO_0018493',
        'phenotypeText': 'decreased metabolism of dihydrocodeine',
        'variantAnnotation': [
            {
                'baseAlleleOrGenotype': '*1/*3 + *1/*4 + *1/*5 + *1/*6 + *4/*41 + *6/*10 + *10/*10 + *41/*41',
                'comparisonAlleleOrGenotype': '*1/*1 + *1/*2 + *1/*9 + *1/*10 + *1/*17 + *1/*41 + *2/*2',
                'directionality': 'increased',
                'effect': 'risk of',
                'effectDescription': (
                    'CYP2D6 *1/*3 + *1/*4 + *1/*5 + *1/*6 + *4/*41 + *6/*10 + *10/*10 + '
                    '*41/*41 (assigned as intermediate metabolizer phenotype) are '
                    'associated with increased risk of sedation when treated with '
                    'ketoprofen and tramadol in people with Pain, Postoperative as '
                    'compared to CYP2D6 *1/*1 + *1/*2 + *1/*9 + *1/*10 + *1/*17 + *1/*41 + '
                    '*2/*2 (assigned as normal metabolizer phenotype) .'
                ),
                'effectType': 'phenotype',
                'entity': 'sedation',
                'id': '1451208260',
                'literature': '25989235',
            }
        ],
        'studyId': '1451208340',
        'targetFromSourceId': 'ENSG00000100197',
        'variantFunctionalConsequenceId': 'SO_0001819',
        'variantRsId': 'rs28358571',
        'variantId': '10_133537858_C_T',
        'isDirectTarget': False,
        'drugs': [{'drugFromSource': 'ketoprofen', 'drugId': 'CHEMBL571'}],
    }

    datasourceId: Literal['clinpgx'] = Field(description='Identifier for the data provider')
    datasourceVersion: Literal['2026-07-05'] = Field(description='Data provider version number')
    datatypeId: Literal['clinical_annotation'] = Field(
        description=('Classification of the type of pharmacogenomic data (e.g. clinical_annotation)')
    )
    directionality: (
        Literal[
            'Decreased function',
            'II/Deficient',
            'III/Deficient',
            'IV/Normal',
            'Increased function',
            'No function',
            'Normal function',
            'Presence',
            'Uncertain function',
            'Unknown function',
        ]
        | None
    ) = Field(description=('Indicates wether the genetic variant increases or decreases drug response'))
    evidenceLevel: Literal['1A', '1B', '2A', '2B', '3', '4'] = Field(
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
        min_length=1, description='List of PubMed or preprint reference identifiers'
    )
    pgxCategory: Literal['dosage', 'efficacy', 'metabolism/pk', 'other', 'pd', 'toxicity'] = Field(
        description='Classification of the drug response type (e.g. Toxicity)'
    )
    phenotypeFromSourceId: Annotated[str, Reference('disease', 'id')] | None = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier result of mapping phenotypeText'
    )
    phenotypeText: str | None = Field(min_length=1, description='Drug response influenced by the variation')
    variantAnnotation: list[VariantAnnotation] | None = Field(
        min_length=1, description='Metadata on the underlying variation'
    )
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
    drugs: list[Drugs] = Field(min_length=1, description='List of drug references')
