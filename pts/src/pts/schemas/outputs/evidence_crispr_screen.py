"""CRISPR screen evidence: the `evidence_crispr_screen` output.

Generated from the 26.09 release output (21,635 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, Reference
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, ONTOLOGY_ID, SHA1_HEX


class EvidenceCrisprScreen(DatasetModel):
    """CRISPR screen evidence."""

    dataset_name: ClassVar[str] = 'evidence_crispr_screen'
    dataset_title: ClassVar[str] = 'CRISPR screen evidence'
    dataset_description: ClassVar[str] = (
        'Disease-target evidence derived from systematic gene perturation assays of CRISPR knock-out screens'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'id': '11641a5c309e45c8bf14286035a07fb770b10630',
        'targetFromSourceId': 'ABHD11',
        'diseaseFromSourceMappedId': 'MONDO_0005559',
        'studyId': 'Glutamatergic Neuron-Liperfluo-CRISPRi',
        'datatypeId': 'affected_pathway',
        'datasourceId': 'crispr_screen',
        'projectId': 'crispr_brain',
        'crisprScreenLibrary': 'Genome-wide CRISPRi-v2',
        'studyOverview': 'Description',
        'cellType': 'Glutamatergic Neuron',
        'geneticBackground': 'PSAP KO',
        'literature': ['34031600'],
        'diseaseFromSource': 'neurodegenerative disease, ageing',
        'contrast': 'Peroxidized Lipids (Liperfluo Intensity)',
        'resourceScore': 0.004910633899271488,
        'statisticalTestTail': 'upper tail',
        'log2FoldChangeValue': 4.532442256417316,
        'qualityControls': [],
        'diseaseId': 'MONDO_0005559',
        'targetId': 'ENSG00000106077',
        'publicationDate': '2021-05-24',
        'evidenceDate': '2021-05-24',
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
    diseaseFromSourceMappedId: Literal[
        'MONDO_0000568',
        'MONDO_0002561',
        'MONDO_0004975',
        'MONDO_0005059',
        'MONDO_0005180',
        'MONDO_0005301',
        'MONDO_0005559',
    ] = Field(description='Mapped Open Targets disease identifier')
    studyId: str = Field(min_length=1, description='Identifier of the study generating the data')
    datatypeId: Literal['affected_pathway'] = Field(description='Type of the evidence')
    datasourceId: Literal['crispr_screen'] = Field(description='Identifer of the evidence source')
    projectId: Literal['crispr_brain'] = Field(description='The identifer of the project that generated the data')
    crisprScreenLibrary: Literal[
        'Custom',
        'Genome-wide',
        'Genome-wide CRISPRa-v2',
        'Genome-wide CRISPRi-v2',
        'H1 Sublibrary CRISPRa-v2',
        'H1 Sublibrary CRISPRi-v2',
        'hCRISPRi_v2 H1 and H4',
        'hCRISPRi_v2 H1 and custom human TFs library',
    ] = Field(description='Applied screening library in the CRISPR/Cas9 project')
    studyOverview: Literal['Description'] = Field(description='Description of the study')
    cellType: Literal['Astrocyte', 'Glutamatergic Neuron', 'Hematopoietic Stem and Progenitor Cell', 'Microglia'] = (
        Field(description='The studied cell type. Preferably the cell line ontology label')
    )
    geneticBackground: Literal['PSAP KO'] | None = Field(
        description=('Additional genetic background of the applied cell line in the CRISPR screen')
    )
    literature: list[Literal['30297964', '34031600', 'PPR387167', 'ppr393667']] | None = Field(
        min_length=1, description='List of PubMed or preprint reference identifiers'
    )
    diseaseFromSource: Literal[
        'Leukemia ',
        'essential genes / neurodegenerative disease',
        'neurodegenerative disease',
        'neurodegenerative disease, ageing',
        'neurodegenerative disease, lysosomal storage disease, AD, PD, MS',
        'neurodegenerative disease, lysosomal storage disease, ageing',
        'neuroinflammatory disorders',
    ] = Field(description='Disease label from the original source')
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
    ] = Field(description='Experiment contrast')
    resourceScore: float = Field(
        ge=0.002895219950005412,
        le=0.49739590287208557,
        description=('Score provided by datasource indicating strength of target-disease association'),
    )
    statisticalTestTail: Literal['lower tail', 'upper tail'] = Field(
        description='End of the distribution the target was picked from'
    )
    log2FoldChangeValue: float = Field(
        ge=-37.164215444007375, le=33.61077926742325, description='Log 2 fold expression change in contrast experiment'
    )
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    publicationDate: Literal['2018-10-08', '2021-05-24', '2021-08-24', '2021-09-12'] | None = Field(
        description='Date of the earliest publication supporting the evidence'
    )
    evidenceDate: Literal['2018-10-08', '2021-05-24', '2021-08-24', '2021-09-12'] | None = Field(
        description='Earliest data for the evidence'
    )
    score: float = Field(
        ge=0.001133900369500035,
        le=1.0,
        description=('Score of the evidence reflecting the strength of the disease/target relationship'),
    )
