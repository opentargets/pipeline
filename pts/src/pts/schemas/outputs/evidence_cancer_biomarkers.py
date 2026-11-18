"""Cancer biomarker evidence: the `evidence_cancer_biomarkers` output.

Generated from the 26.12 release output (1,302 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import CHEMBL_ID, DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX, VARIANT_ID


class Urls(StructModel):
    """`urls[]`."""

    niceName: str = Field(min_length=1, description='Human readable label of the remote reference')
    url: str = Field(min_length=1, description='URL to remote resource')


class GeneExpression(StructModel):
    """`biomarkers/geneExpression[]`."""

    id: Annotated[str, Bioregistry('go')] = Field(
        pattern=ONTOLOGY_ID,
        description=('Gene Ontology (GO) identifiers of regulation or background expression processes'),
    )
    name: str = Field(
        min_length=1, description=('Gene expression alteration as reported by the Cancer Genome Interpreter')
    )


class GeneticVariation(StructModel):
    """`biomarkers/geneticVariation[]`."""

    functionalConsequenceId: Annotated[str, Reference('so', 'id')] | None = Field(
        pattern=ONTOLOGY_ID, description=('Sequence ontology (SO) identifier of the alteration type of the biomarker')
    )
    id: Annotated[str, Reference('variant', 'variantId')] | None = Field(
        pattern=VARIANT_ID,
        description=(
            'Identifier in CHROM_POS_REF_ALT notation of the biomarker variant, '
            'when the alteration is a specific variant'
        ),
    )
    name: str = Field(
        min_length=1,
        description=("Alteration as reported by the Cancer Genome Interpreter (e.g. 'MET:amp', 'PTEN:del')"),
    )


class Biomarkers(StructModel):
    """`biomarkers`."""

    geneExpression: list[GeneExpression] | None = Field(
        description=('Gene expression changes (over- or under-expression) that define the biomarker')
    )
    geneticVariation: list[GeneticVariation] | None = Field(
        description=(
            'Genetic alterations (e.g. mutations, amplifications, deletions, fusions) that define the biomarker'
        )
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
        'id': '7a4a028f7933a61f5e466a561f3f02c7a18346d7',
        'targetFromSourceId': 'AKT1',
        'diseaseFromSourceMappedId': 'EFO_0000389',
        'datasourceId': 'cancer_biomarkers',
        'datatypeId': 'affected_pathway',
        'drugFromSource': 'Braf Inhibitors',
        'drugId': 'CHEMBL1289926',
        'drugResponse': 'EFO_0020001',
        'diseaseFromSource': 'Cutaneous melanoma',
        'confidence': 'Case report',
        'biomarkerName': 'AKT1 (Q79K,E17K)',
        'literature': ['24265152'],
        'urls': [
            {
                'niceName': ('EORTC-NCI-AACR Symposium on Molecular Targets and Cancer Therapeutics 2014'),
                'url': 'https://www.ecco-org.eu/Events/Past-conferences/EORTC_NCI_AACR_2014',
            }
        ],
        'biomarkers': {
            'geneExpression': [{'id': 'GO_0010628', 'name': 'AR:over'}],
            'geneticVariation': [
                {'functionalConsequenceId': 'SO_0001777', 'id': '14_105243048_G_T', 'name': 'AKT1:Q79K'}
            ],
        },
        'qualityControls': [],
        'diseaseId': 'MONDO_0005012',
        'targetId': 'ENSG00000142208',
        'publicationDate': '2013-11-21',
        'evidenceDate': '2013-11-21',
        'score': 1.0,
    }

    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        min_length=1, description='Gene symbol of the target as reported by the Cancer Genome Interpreter'
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    datasourceId: str = Field(pattern=ONTOLOGY_ID, description='Identifer of the evidence source')
    datatypeId: str = Field(pattern=ONTOLOGY_ID, description='Type of the evidence')
    drugFromSource: str = Field(
        min_length=1,
        description=('Drug or drug family associated with the biomarker, as reported by the Cancer Genome Interpreter'),
    )
    drugId: Annotated[str, Reference('drug_molecule', 'id')] | None = Field(
        pattern=CHEMBL_ID,
        description=(
            'Open Targets drug identifier, when the drug reported by the Cancer Genome Interpreter could be mapped'
        ),
    )
    drugResponse: str = Field(
        pattern=ONTOLOGY_ID,
        description=(
            'Ontology term of the drug response associated with the biomarker '
            '(e.g. response to drug, drug resistance, drug toxicity)'
        ),
    )
    diseaseFromSource: str = Field(min_length=1, description='Tumour type as reported by the Cancer Genome Interpreter')
    confidence: str = Field(
        min_length=1,
        description=(
            'Evidence level of the biomarker-drug association in the Cancer Genome '
            "Interpreter (e.g. 'FDA guidelines', 'NCCN guidelines', 'Early "
            "trials', 'Pre-clinical')"
        ),
    )
    biomarkerName: str = Field(
        min_length=1,
        description=(
            'Name of the biomarker as reported by the Cancer Genome Interpreter '
            "(e.g. 'EGFR exon 20 insertions', 'KIT wildtype')"
        ),
    )
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        description='PubMed identifiers supporting the biomarker-drug association'
    )
    urls: list[Urls] | None = Field(
        description=('Links to the guidelines, clinical trials or reports supporting the biomarker-drug association')
    )
    biomarkers: Biomarkers = Field(description='Genetic alterations and expression changes that define the biomarker')
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
            'relationship; 1.0 for all Cancer Genome Interpreter biomarkers'
        ),
    )
