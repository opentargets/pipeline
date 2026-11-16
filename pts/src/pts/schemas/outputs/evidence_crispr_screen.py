"""CRISPR screen evidence: the `evidence_crispr_screen` output.

Generated from the 26.12 release output (21,631 rows) and its croissant metadata by
`scripts/output_schemas/generate.py`, whose docstring lists which observations become rules.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, Reference
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


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
    datatypeId: str = Field(pattern=ONTOLOGY_ID, description='Type of the evidence')
    datasourceId: str = Field(pattern=ONTOLOGY_ID, description='Identifer of the evidence source')
    projectId: str = Field(
        pattern=ONTOLOGY_ID,
        description=("Identifier of the project that generated the data ('crispr_brain' for CRISPRbrain)"),
    )
    crisprScreenLibrary: Literal[
        'Custom',
        'Genome-wide',
        'Genome-wide CRISPRa-v2',
        'Genome-wide CRISPRi-v2',
        'H1 Sublibrary CRISPRa-v2',
        'H1 Sublibrary CRISPRi-v2',
        'hCRISPRi_v2 H1 and H4',
        'hCRISPRi_v2 H1 and custom human TFs library',
    ] = Field(description=('CRISPR library screened in the CRISPRbrain screen (e.g. Genome-wide CRISPRi-v2)'))
    studyOverview: str = Field(min_length=1, description='Description of the CRISPRbrain screen')
    cellType: Literal['Astrocyte', 'Glutamatergic Neuron', 'Hematopoietic Stem and Progenitor Cell', 'Microglia'] = (
        Field(description='Cell type in which the CRISPRbrain screen was performed')
    )
    geneticBackground: Literal['PSAP KO'] | None = Field(
        description="Genetic background of the screened cells (e.g. 'PSAP KO')"
    )
    literature: list[Literal['30297964', '34031600', 'PPR387167', 'ppr393667']] | None = Field(
        description=('PubMed or Europe PMC preprint identifier of the publication describing the CRISPRbrain screen')
    )
    diseaseFromSource: Literal[
        'Leukemia ',
        'essential genes / neurodegenerative disease',
        'neurodegenerative disease',
        'neurodegenerative disease, ageing',
        'neurodegenerative disease, lysosomal storage disease, AD, PD, MS',
        'neurodegenerative disease, lysosomal storage disease, ageing',
        'neuroinflammatory disorders',
    ] = Field(description=('Disease area associated with the CRISPRbrain screen, as curated by Open Targets'))
    contrast: Literal[
        'Expansion (CD34 Staining)',
        'Immune Activation (CD38 levels)',
        'Inflammatory activation (cell-surface VCAM1 levels)',
        'Labile Iron (FeRhoNox Intensity)',
        'Lysosome (Lysotracker Intensity)',
        'Lysosome exocytosis (cell-surface LAMP1 levels)',
        'Lysosome mass or pH (LysoTracker staining)',
        'Peroxidized Lipids (Liperfluo Intensity)',
        'Phagocytosis (pHRodo-rat synaptosomes)',
        'Phagocytosis (pHrodo-labeled synaptosomes)',
        'Reactive Oxygen Species (CellRox Intensity)',
        'Survival',
    ] = Field(
        description=("Phenotype read out in the CRISPRbrain screen (e.g. 'Phagocytosis (pHrodo-labeled synaptosomes)')")
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
            'relationship, derived from the log10 of the CRISPRbrain p-value'
        ),
    )
