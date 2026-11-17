"""Structs shared by several dataset models.

A struct is shared only where it means the same thing in every dataset that uses it and has the same rules
there. Structs that merely have the same field names, or that each dataset documents differently, stay in
their own model. Nullability of a struct column or of its list elements is declared by each model's field,
so models sharing a struct can still differ there.
"""

from typing import Annotated

from pydantic import Field

from pts.schemas.dataset import Reference, StructModel
from pts.schemas.outputs.patterns import ONTOLOGY_ID


class Biomarker(StructModel):
    """A biomarker of a cell model: `biomarkerList[]` of evidence_encore and evidence_ot_crispr_validation."""

    description: str = Field(min_length=1, description='One biomarker for the model')
    name: str = Field(min_length=1, description='Short name of the biomarker')


class MappedDisease(StructModel):
    """A disease as the source reports it, with its mapped id: `diseases[]` of clinical_report and clinical_target."""

    diseaseFromSource: str = Field(min_length=1, description='Disease name as reported in the source')
    diseaseId: Annotated[str, Reference('disease', 'id')] | None = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
