"""Biosample enrichment of L2G-prioritised genes: the `biosample_enrichment` output.

Written by hand, as no release with this output was available to generate it from: dtypes and column order
from `pts.schemas.biosample_enrichment`, descriptions and references from croissant's
`biosample_enrichment.json`, and rules only where `pts.transformers.biosample_enrichment` guarantees them:

* each row is one study on its own (`studyId`) or one disease's studies pooled (`diseaseId`), never both
  nor neither -- a row check, as it spans two columns;
* `tissueBiosampleId` is set only for a tissue x cell type panel;
* `enrichmentApproach` is the transformer's constant; `biosampleDataset` comes from the step's settings, so
  it is not a closed vocabulary;
* undefined statistics are written as null, never NaN; a p-value lies in [0, 1] and a standard error is
  not negative.

croissant declares no primary key, and the natural one (unit, dataset, biosample, tissue) has nullable
columns, so none is declared.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from ot_datamodels import DatasetModel, Reference, RowCheck
from ot_datamodels.outputs.patterns import ONTOLOGY_ID


class BiosampleEnrichment(DatasetModel):
    """Biosample enrichment of L2G-prioritised genes."""

    dataset_name: ClassVar[str] = 'biosample_enrichment'
    dataset_title: ClassVar[str] = 'Biosample enrichment'
    dataset_description: ClassVar[str] = (
        'Enrichment of biosample-specific expression among the genes prioritised by locus-to-gene for a GWAS study, '
        'or for the studies of one disease pooled together'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Genetics',)
    example: ClassVar[dict] = {
        'diseaseId': 'EFO_0000270',
        'studyId': None,
        'biosampleId': 'CL_0000236',
        'tissueBiosampleId': 'UBERON_0002048',
        'biosampleDataset': 'tabula_sapiens',
        'biosampleFromSource': 'lung B cell',
        'enrichmentApproach': 'l2g_regression',
        'beta': 0.12,
        'betaError': 0.03,
        'pValue': 0.0004,
    }
    row_checks: ClassVar[tuple[RowCheck, ...]] = (
        RowCheck(
            'one_study_or_one_pooled_disease',
            "Exactly one of `studyId` and `diseaseId` is set: a study on its own, or a disease's studies pooled.",
            pl.col('studyId').is_null() != pl.col('diseaseId').is_null(),
        ),
    )

    diseaseId: Annotated[str | None, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID,
        description=(
            'Identifier of the disease or phenotype tested for biosample enrichment, with its studies pooled; null '
            'for a single study, which sets studyId instead'
        ),
    )
    studyId: Annotated[str | None, Reference('study', 'studyId')] = Field(
        min_length=1,
        description=(
            'Identifier of the GWAS tested for biosample enrichment on its own; null for a pooled disease, which '
            'sets diseaseId instead'
        ),
    )
    biosampleId: Annotated[str, Reference('biosample', 'biosampleId')] = Field(
        min_length=1, description='Ontology identifier of the biosample tested for enrichment'
    )
    tissueBiosampleId: Annotated[str | None, Reference('biosample', 'biosampleId')] = Field(
        min_length=1,
        description=(
            'Ontology identifier of the tissue the cell type was profiled in, for tissue x cell type biosamples; '
            'null otherwise'
        ),
    )
    biosampleDataset: str = Field(
        min_length=1, description='Source dataset containing the biosample annotations used for enrichment'
    )
    biosampleFromSource: str | None = Field(min_length=1, description='Biosample as reported by the source dataset')
    enrichmentApproach: Literal['l2g_regression'] = Field(description='Method used to estimate biosample enrichment')
    beta: float | None = Field(description='Estimated effect size of the biosample enrichment')
    betaError: float | None = Field(ge=0.0, description='Standard error of the biosample enrichment effect size')
    pValue: float | None = Field(ge=0.0, le=1.0, description='P-value for the biosample enrichment')
