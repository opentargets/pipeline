"""Cancer biomarker evidence: the `evidence_cancer_biomarkers` output.

Generated from the 26.09 release output (1,301 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import CHEMBL_ID, DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX, VARIANT_ID


class Urls(StructModel):
    """`urls[]`."""

    niceName: str = Field(min_length=1, description='Human readable label of the remote reference')
    url: str = Field(min_length=1, description='URL to remote resource')


class GeneExpression(StructModel):
    """`biomarkers/geneExpression[]`."""

    id: Annotated[Literal['GO_0010467', 'GO_0010628', 'GO_0010629'], Bioregistry('go')] = Field(
        description=('Gene Ontology (GO) identifiers of regulation or background expression processes')
    )
    name: str = Field(min_length=1, description='Raw gene expression annotation from the source')


class GeneticVariation(StructModel):
    """`biomarkers/geneticVariation[]`."""

    functionalConsequenceId: Annotated[str, Bioregistry('so'), Reference('so', 'id')] | None = Field(
        pattern=ONTOLOGY_ID, description='Functional consequene identifier of the variant biomarker'
    )
    id: Annotated[str, Reference('variant', 'variantId')] | None = Field(
        pattern=VARIANT_ID, description='Variant identifier'
    )
    name: str = Field(min_length=1, description='Name of the variant biomarker')


class Biomarkers(StructModel):
    """`biomarkers`."""

    geneExpression: list[GeneExpression] | None = Field(
        min_length=1, description='List of gene expression altering biomarkers'
    )
    geneticVariation: list[GeneticVariation] | None = Field(
        min_length=1, description='List of gene expression altering biomarkers'
    )


class EvidenceCancerBiomarkers(DatasetModel):
    """Cancer biomarker evidence."""

    dataset_name: ClassVar[str] = 'evidence_cancer_biomarkers'
    dataset_title: ClassVar[str] = 'Cancer biomarker evidence'
    dataset_description: ClassVar[str] = (
        'Disease-target evidence based on cancer biomarkers from the Cancer Genome Interpreter'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'id': '8f4586585642f889be1e83b21c64f9133fd9ea1b',
        'targetFromSourceId': 'MET',
        'diseaseFromSourceMappedId': 'EFO_0000571',
        'datasourceId': 'cancer_biomarkers',
        'datatypeId': 'affected_pathway',
        'drugFromSource': 'Rociletinib',
        'drugId': 'CHEMBL3545308',
        'drugResponse': 'EFO_0020001',
        'diseaseFromSource': 'Lung adenocarcinoma',
        'confidence': 'Case report',
        'biomarkerName': 'MET amplification',
        'literature': ['27252416'],
        'urls': [
            {
                'niceName': 'Food and Drug Administration guidelines',
                'url': 'https://www.fda.gov/RegulatoryInformation/Guidances/default.htm',
            }
        ],
        'biomarkers': {
            'geneExpression': [{'id': 'GO_0010629', 'name': 'CRBN:under'}],
            'geneticVariation': [{'functionalConsequenceId': 'SO_0001563', 'id': '6_18139228_C_T', 'name': 'MET:amp'}],
        },
        'qualityControls': [],
        'diseaseId': 'MONDO_0005061',
        'targetId': 'ENSG00000105976',
        'publicationDate': '2016-06-01',
        'evidenceDate': '2016-06-01',
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
    datasourceId: Literal['cancer_biomarkers'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['affected_pathway'] = Field(description='Type of the evidence')
    drugFromSource: str = Field(min_length=1, description='Drug name/family in resource of origin')
    drugId: Annotated[str, Bioregistry('chembl'), Reference('drug_molecule', 'id')] | None = Field(
        pattern=CHEMBL_ID, description='Open Targets drug identifier'
    )
    drugResponse: Literal[
        'EFO_0005558',
        'EFO_0006951',
        'EFO_0007053',
        'EFO_0020001',
        'EFO_0020002',
        'EFO_0020003',
        'GO_0042493',
        'HP_0002904',
    ] = Field(description='Observed patterns of drug response')
    diseaseFromSource: str = Field(min_length=1, description='Disease label from the original source')
    confidence: Literal[
        'CPIC guidelines',
        'Case Report',
        'Case report',
        'Clinical trials',
        'Early Trials',
        'Early trials',
        'European LeukemiaNet guidelines',
        'FDA guidelines',
        'Late trials',
        'NCCN guidelines',
        'NCCN/CAP guidelines',
        'Pre-clinical',
    ] = Field(description='Confidence qualifier on the reported evidence')
    biomarkerName: str = Field(min_length=1, description='Altered characteristics that influences the disease process')
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        min_length=1, description='List of PubMed or preprint reference identifiers'
    )
    urls: list[Urls] | None = Field(
        min_length=1,
        description=(
            'Reference to linked external resource (e.g. clinical trials, studies, package inserts, reports, etc.)'
        ),
    )
    biomarkers: Biomarkers = Field(description='List of biomarkers')
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
        ge=1.0, le=1.0, description=('Score of the evidence reflecting the strength of the disease/target relationship')
    )
