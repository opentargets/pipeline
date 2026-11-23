"""CRISPR screen evidence: the `evidence_crispr_screen` output.

Generated from the 26.12 release output (21,631 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from ot_datamodels import ClickHouseType, DatasetModel, PrimaryKey, Reference
from ot_datamodels.outputs.patterns import ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX
from ot_datamodels.vocabularies import Datasource, Datatype, enum_of


class EvidenceCrisprScreen(DatasetModel):
    """CRISPR screen evidence."""

    dataset_name: ClassVar[str] = 'evidence_crispr_screen'
    dataset_title: ClassVar[str] = 'CRISPR screen evidence'
    dataset_description: ClassVar[str] = (
        'Disease-target evidence derived from systematic gene perturation assays of CRISPR knock-out screens'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'id': '05abce17ec5da912af9fff24205573f371417026',
        'targetFromSourceId': 'LTV1',
        'diseaseFromSourceMappedId': 'MONDO_0005559',
        'studyId': 'Glutamatergic Neuron-CellRox-CRISPRi',
        'datatypeId': 'affected_pathway',
        'datasourceId': 'crispr_screen',
        'projectId': 'crispr_brain',
        'crisprScreenLibrary': 'Genome-wide CRISPRi-v2',
        'studyOverview': (
            'CRISPRi FACS screen for reactive oxygen species in human iPSC-derived glutamatergic neurons.'
        ),
        'cellType': 'Glutamatergic Neuron',
        'geneticBackground': 'PSAP KO',
        'literature': ['34031600'],
        'diseaseFromSource': 'neurodegenerative disease, ageing',
        'contrast': 'Reactive Oxygen Species (CellRox Intensity)',
        'resourceScore': 0.003281126031652093,
        'statisticalTestTail': 'lower tail',
        'phenotypicConsequenceScore': -10.69931943057236,
        'qualityControls': [],
        'diseaseId': 'MONDO_0005559',
        'targetId': 'ENSG00000135521',
        'publicationDate': '2021-05-24',
        'evidenceDate': '2021-05-24',
        'score': 1.0,
    }

    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(min_length=1, description='Gene symbol of the target as reported by CRISPRbrain')
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    studyId: str = Field(min_length=1, description='Name of the CRISPRbrain screen')
    datatypeId: Annotated[Literal['affected_pathway'], enum_of(Datatype)] = Field(description='Type of the evidence')
    datasourceId: Annotated[Literal['crispr_screen'], enum_of(Datasource)] = Field(
        description='Identifer of the evidence source'
    )
    projectId: str = Field(
        pattern=ONTOLOGY_ID,
        description=("Identifier of the project that generated the data ('crispr_brain' for CRISPRbrain)"),
    )
    crisprScreenLibrary: str = Field(
        min_length=1, description=('CRISPR library screened in the CRISPRbrain screen (e.g. Genome-wide CRISPRi-v2)')
    )
    studyOverview: str = Field(min_length=1, description='Description of the CRISPRbrain screen')
    cellType: str = Field(min_length=1, description='Cell type in which the CRISPRbrain screen was performed')
    geneticBackground: str | None = Field(
        min_length=1, description="Genetic background of the screened cells (e.g. 'PSAP KO')"
    )
    literature: list[Literal['30297964', '34031600', 'PPR387167', 'ppr393667']] | None = Field(
        description=('PubMed or Europe PMC preprint identifier of the publication describing the CRISPRbrain screen')
    )
    diseaseFromSource: str = Field(
        min_length=1, description=('Disease area associated with the CRISPRbrain screen, as curated by Open Targets')
    )
    contrast: str = Field(
        min_length=1,
        description=(
            "Phenotype read out in the CRISPRbrain screen (e.g. 'Phagocytosis (pHrodo-labeled synaptosomes)')"
        ),
    )
    resourceScore: float = Field(ge=0.0, le=1.0, description='P-value of the target in the CRISPRbrain screen')
    statisticalTestTail: Literal['lower tail', 'upper tail'] = Field(
        description=("'upper tail' for positive hits and 'lower tail' for negative hits of the CRISPRbrain screen")
    )
    phenotypicConsequenceScore: float = Field(
        description=(
            'Phenotype score of the target in the CRISPRbrain screen, computed '
            'with the MAGeCK-iNC pipeline and, as described for most screens, '
            'standardised against negative-control quasi-genes. Positive for '
            'positive hits, negative for negative hits'
        )
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
            'relationship, derived from the log10 of the CRISPRbrain p-value'
        ),
    )
