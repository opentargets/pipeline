"""ENCORE genetic-interaction evidence: the `evidence` output of `evidence_encore`.

Descriptions -- of nested struct fields too -- are copied verbatim from croissant's
`evidence_encore.json`, except `geneticInteractionType`, whose croissant entry is misspelled
`geneInteractionType`. Croissant has no entries for `validationReadouts`' fields, so those are
undescribed.

Allowed values, patterns and nullability are inferred from a real output (112,134 rows), which has
no nulls at any depth. That snapshot shows a single value for `datatypeId`, `datasourceId` and
`statisticalMethod`, and only `'antagonistic'` for `geneticInteractionType` -- whose description
names `'cooperative'` too, so both are allowed. The single-valued ones may need loosening if a
future ENCORE release adds a second `statisticalMethod`.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, Note, Reference, StructModel
from pts.schemas.outputs.evidence_base import ISO_DATE, EvidenceBase

TargetRole = Literal['Paralog', 'Library', 'GIControl', 'Anchor']


class Biomarker(StructModel):
    """A biomarker of the biological model."""

    description: str = Field(description='One biomarker for the model')
    name: str = Field(description='Short name of the biomarker')


class DiseaseCellLine(StructModel):
    """A cancer cell line used to generate the evidence."""

    id: str = Field(description='Cell type identifier in cell ontology or in cell model database')
    name: str = Field(description='Name of the cell model')
    tissue: str = Field(description='Name of the tissue from which the cells were sampled')
    tissueId: Annotated[str, Reference('biosample', 'biosampleId'), Bioregistry('uberon')] = Field(
        description='Anatomical identifier of the sampled organ/tissue',
    )


class ValidationReadout(StructModel):
    """An experimental validation readout."""

    hsaValue: float
    isValidated: bool
    readoutMethodName: str
    screen: str


class EncoreEvidence(EvidenceBase):
    """ENCORE genetic-interaction evidence."""

    dataset_name: ClassVar[str] = 'evidence_encore'
    dataset_description: ClassVar[str] = 'ENCORE genetic-interaction evidence'
    example: ClassVar[dict] = {
        'diseaseFromSourceMappedId': 'EFO_0000365',
        'diseaseId': 'EFO_0000365',
        'targetId': 'ENSG00000149554',
        'id': '9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0d',
        # valid evidence never carries a flag; one is here so the element rules get exercised
        'qualityControls': ['Duplicated'],
        'score': 0.5,
        'datasourceId': 'encore',
        'datatypeId': 'ot_partner',
        'targetFromSourceId': 'CHEK1',
        'biomarkerList': [{'description': 'KRAS mutant', 'name': 'KRAS'}],
        'diseaseCellLines': [
            {'id': 'SIDM00650', 'name': 'HCT-116', 'tissue': 'Large Intestine', 'tissueId': 'UBERON_0000059'},
        ],
        'diseaseFromSource': 'Colorectal Carcinoma',
        'geneticInteractionScore': -8.0,
        'geneticInteractionType': 'antagonistic',
        'interactingTargetFromSourceId': 'PARK7',
        'interactingTargetRole': 'Library',
        'phenotypicConsequenceLogFoldChange': -1.2,
        'phenotypicConsequencePValue': 0.001,
        'projectId': 'OTAR2062',
        'releaseDate': '2026-03-31',
        'releaseVersion': '26.06',
        'statisticalMethod': 'HSA_Z',
        'targetRole': 'Anchor',
        'validationReadouts': [
            {'hsaValue': 0.42, 'isValidated': True, 'readoutMethodName': 'CellTiter-Glo', 'screen': 'combinatorial'},
        ],
        'evidenceDate': '2026-03-31',
    }

    datasourceId: Literal['encore'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['ot_partner'] = Field(description='Type of the evidence')
    targetFromSourceId: str = Field(
        description=(
            'Target ID in resource of origin (accepted sources include Ensembl gene ID, Uniprot '
            'ID, gene symbol), only capital letters are accepted'
        ),
    )
    biomarkerList: list[Biomarker] = Field(description='List of biomarkers associated with the biological model')
    diseaseCellLines: list[DiseaseCellLine] = Field(description='Cancer cell lines used to generate evidence')
    diseaseFromSource: str = Field(description='Disease label from the original source')
    geneticInteractionScore: float = Field(
        description=(
            'The strength of the genetic interaction. Directionality is captured as well: '
            'antagonistics < 0 < cooperative'
        ),
    )
    geneticInteractionType: Annotated[
        Literal['antagonistic', 'cooperative'],
        Note("croissant's own entry for this field is misspelled 'geneInteractionType'"),
    ] = Field(description='Description of the interaction between the two genes')
    interactingTargetFromSourceId: str = Field(description='Identifer of the interacting target')
    interactingTargetRole: TargetRole = Field(description='Role of a target in the genetic interaction test')
    phenotypicConsequenceLogFoldChange: float = Field(description='Log 2 fold change of the cell survival')
    phenotypicConsequencePValue: float = Field(ge=0.0, le=1.0, description='P-value of the the cell survival test')
    projectId: str = Field(description='The identifer of the project that generated the data')
    releaseDate: str = Field(pattern=ISO_DATE, description="Date of the release of the data in a 'YYYY-MM-DD' format")
    releaseVersion: str = Field(description='Open Targets data release version')
    statisticalMethod: Annotated[
        Literal['HSA_Z'],
        Note('only one method observed in the reference snapshot; loosen if ENCORE adds another'),
    ] = Field(description='Statistical method used to calculate the association')
    targetRole: TargetRole = Field(description='Role of a target in the genetic interaction test')
    validationReadouts: list[ValidationReadout] = Field(description='List of experimental validation readouts')
    evidenceDate: str = Field(pattern=ISO_DATE, description='Earliest data for the evidence')
