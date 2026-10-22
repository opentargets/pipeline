"""Baseline expression: the `baseline_expression` output.

Generated from the 26.09 release output (83,290,030 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, Reference
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, ONTOLOGY_ID


class BaselineExpression(DatasetModel):
    """Baseline expression."""

    dataset_name: ClassVar[str] = 'baseline_expression'
    dataset_title: ClassVar[str] = 'Baseline expression'
    dataset_description: ClassVar[str] = (
        'Aggregated expression data generated from RNA-seq and mass spectrometry proteomic data'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target',)
    example: ClassVar[dict] = {
        'celltypeBiosampleId': 'CL_0000775',
        'tissueBiosampleId': 'UBERON_0002118',
        'targetId': 'ENSG00000000419',
        'datasourceId': 'tabula_sapiens',
        'datatypeId': 'scrna-seq',
        'unit': 'CPM(pseudobulk sum[counts])',
        'celltypeBiosampleFromSource': 'neutrophil',
        'min': 134.9971923828125,
        'q1': 134.9971923828125,
        'median': 134.9971923828125,
        'q3': 134.9971923828125,
        'max': 134.9971923828125,
        'distribution_score': 0.9485559566787004,
        'specificity_score': 0.08673607355327204,
        'tissueBiosampleFromSource': 'right ovary',
        'targetFromSourceId': 'A0A0J9YY54',
        'tissueBiosampleParentId': 'UBERON_0003134',
        'celltypeBiosampleParentId': 'CL_0000094',
        'qualityControls': [],
    }

    celltypeBiosampleId: Annotated[str, Bioregistry('cell ontology'), Reference('biosample', 'biosampleId')] | None = (
        Field(pattern=ONTOLOGY_ID, description='Cell type identifier that maps to the cell type reported by study')
    )
    tissueBiosampleId: Annotated[str, Bioregistry('uberon'), Reference('biosample', 'biosampleId')] | None = Field(
        pattern=ONTOLOGY_ID, description='Tissue identifier that maps to the tissue reported by study'
    )
    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Unique identifier for the target'
    )
    datasourceId: Literal['DICE', 'PRIDE', 'gtex', 'tabula_sapiens'] = Field(description='Origin of expression data')
    datatypeId: Literal['bulk rna-seq', 'mass-spectrometry proteomics', 'scrna-seq'] = Field(
        description='Type of expression data'
    )
    unit: Literal['CPM(pseudobulk sum[counts])', 'PPB (iBAQ)', 'TPM'] = Field(
        description='Unit for the target expression'
    )
    celltypeBiosampleFromSource: str | None = Field(min_length=1, description='Tissue or cell type reported by study')
    min: float | None = Field(ge=0.0, le=232099581.86360273, description='Minimum value of expression')
    q1: float | None = Field(ge=0.0, le=232099581.86360273, description='First quantile of expression')
    median: float | None = Field(ge=0.0, le=232099581.86360273, description='Median value of expression')
    q3: float | None = Field(ge=0.0, le=261457592.2468968, description='Third quantile of expression')
    max: float | None = Field(ge=0.0, le=290572862.4642683, description='Maximum value of expression')
    distribution_score: float = Field(
        ge=0.0,
        le=1.0,
        description=('Proportion of biosamples, within the dataset, that display median expression above a threshold'),
    )
    specificity_score: float | None = Field(
        ge=0.0, le=1.0, description='Measure of how specific expression is to a biosample'
    )
    tissueBiosampleFromSource: str | None = Field(min_length=1, description='Cell type reported by study')
    targetFromSourceId: str | None = Field(min_length=1, description='Target identifier from the source')
    tissueBiosampleParentId: Annotated[str, Bioregistry('uberon'), Reference('biosample', 'biosampleId')] | None = (
        Field(pattern=ONTOLOGY_ID, description='Broad tissue identifier that maps to the tissue reported by study')
    )
    celltypeBiosampleParentId: (
        Annotated[str, Bioregistry('cell ontology'), Reference('biosample', 'biosampleId')] | None
    ) = Field(
        pattern=ONTOLOGY_ID, description=('Broad cell type identifier that maps to the cell type reported by study')
    )
    qualityControls: list[str] = Field(
        max_length=0, description='Quality control flags or notes for baseline expression'
    )
