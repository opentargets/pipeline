"""Pandera schema validating PTS ENCORE genetic-interaction evidence.

For the frame produced by `pts.transformers.encore_evidence.encore_evidence`, right before
`write_dataset` writes it to parquet. Field descriptions are copied verbatim from
`croissant/src/ot_croissant/assets/recordset/evidence_encore.json` -- except `geneticInteractionType`,
whose croissant entry is spelled `geneInteractionType` (missing "tic"; a real mismatch in that
file, not this schema) -- and that file's `isPrimaryKey`/`foreign_key`/`[bioregistry:...]` tags are
mirrored into `pa.Field(metadata=...)`, matching `pts.schemas.evidence`'s convention.

Enum values, patterns and nullability below are inferred from the real reference output at
`work/output/evidence_encore` (112,134 rows) -- confirmed, not merely plausible, but from one
snapshot: `targetRole`/`interactingTargetRole` show all 4 documented role combinations, but
`datatypeId`/`datasourceId`/`statisticalMethod` show only a single constant value each in this
snapshot, and `geneticInteractionType` only `'antagonistic'` even though its own description names
`'cooperative'` too (`score < 0 < score` in the source statistic) -- so that one enum includes both
values by description, not just by observation, while the single-valued ones are asserted as
observed and may need loosening if a future ENCORE release adds a second `statisticalMethod`.

Three struct-list columns (`biomarkerList`, `diseaseCellLines`, `validationReadouts`) are typed
using the actual `pl.List(pl.Struct(...))` dtype object directly as the field's Python type
annotation -- pandera.polars accepts a real Polars dtype instance in that position exactly like it
accepts `str`/`list[str]` elsewhere, which is how nested struct columns get real dtype validation
here instead of being left as opaque `object`.
"""

import pandera.polars as pa
import polars as pl
from pandera.polars import PolarsData

#: `id` as sha1 hex, matching `identifiers.py::_sha1` -- same as `pts.schemas.evidence`.
_SHA1_HEX = r'^[0-9a-f]{40}$'

#: Ensembl gene ID shape, for `targetId` (always Ensembl for Open Targets' own identifier, unlike
#: `targetFromSourceId`, which for ENCORE is a bare gene symbol -- e.g. `CHEK1` -- not constrained
#: to a single pattern here, matching `evidence_encore.json`'s own "accepted sources include
#: Ensembl gene ID, Uniprot ID, gene symbol" description).
_ENSEMBL_GENE_ID = r'^ENSG\d{11}$'

#: Ontology-style ID: a namespace prefix, underscore, code -- `diseaseFromSourceMappedId`/`diseaseId`
#: observed as EFO_/MONDO_ in the reference output.
_ONTOLOGY_ID = r'^[A-Za-z]+_[A-Za-z0-9]+$'

_QUALITY_FLAGS = (
    'No valid disease',
    'No valid target',
    'Duplicated',
    'No valid score',
    'Invalid biotype',
)

_BIOMARKER_STRUCT = pl.List(pl.Struct({'description': pl.String, 'name': pl.String}))
_DISEASE_CELL_LINE_STRUCT = pl.List(
    pl.Struct({'id': pl.String, 'name': pl.String, 'tissue': pl.String, 'tissueId': pl.String})
)
_VALIDATION_READOUT_STRUCT = pl.List(
    pl.Struct({
        'hsaValue': pl.Float64,
        'isValidated': pl.Boolean,
        'readoutMethodName': pl.String,
        'screen': pl.String,
    })
)


class EncoreEvidenceSchema(pa.DataFrameModel):
    """Schema for the `evidence` output of `pts.transformers.encore_evidence.encore_evidence`."""

    targetFromSourceId: str = pa.Field(
        description=(
            'Target ID in resource of origin (accepted sources include Ensembl gene ID, Uniprot '
            'ID, gene symbol), only capital letters are accepted'
        ),
    )
    diseaseFromSourceMappedId: str = pa.Field(
        str_matches=_ONTOLOGY_ID,
        description='Mapped Open Targets disease identifier',
    )
    biomarkerList: _BIOMARKER_STRUCT = pa.Field(  # ty:ignore[invalid-type-form]
        description='List of biomarkers associated with the biological model'
    )
    datasourceId: str = pa.Field(isin=['encore'], description='Identifer of the evidence source')
    datatypeId: str = pa.Field(isin=['ot_partner'], description='Type of the evidence')
    diseaseCellLines: _DISEASE_CELL_LINE_STRUCT = pa.Field(  # ty:ignore[invalid-type-form]
        description=(
            'Cancer cell lines used to generate evidence -- nested `tissueId` is an anatomical '
            'identifier [bioregistry:uberon] and a foreign key to biosample/biosampleId, but '
            'pandera field metadata cannot target a field nested inside a struct-list, so it is '
            'noted here instead of as structured metadata'
        ),
    )
    diseaseFromSource: str = pa.Field(description='Disease label from the original source')
    geneticInteractionScore: float = pa.Field(
        description=(
            'The strength of the genetic interaction. Directionality is captured as well: '
            'antagonistics < 0 < cooperative'
        ),
    )
    geneticInteractionType: str = pa.Field(
        isin=['antagonistic', 'cooperative'],
        description='Description of the interaction between the two genes',
        metadata={'note': "croissant's own entry for this field is misspelled 'geneInteractionType'"},
    )
    interactingTargetFromSourceId: str = pa.Field(description='Identifer of the interacting target')
    interactingTargetRole: str = pa.Field(
        isin=['Paralog', 'Library', 'GIControl', 'Anchor'],
        description='Role of a target in the genetic interaction test',
    )
    phenotypicConsequenceLogFoldChange: float = pa.Field(description='Log 2 fold change of the cell survival')
    phenotypicConsequencePValue: float = pa.Field(
        ge=0.0,
        le=1.0,
        description='P-value of the the cell survival test',
    )
    projectId: str = pa.Field(description='The identifer of the project that generated the data')
    releaseDate: str = pa.Field(
        str_matches=r'^\d{4}-\d{2}-\d{2}$',
        description="Date of the release of the data in a 'YYYY-MM-DD' format",
    )
    releaseVersion: str = pa.Field(description='Open Targets data release version')
    statisticalMethod: str = pa.Field(
        isin=['HSA_Z'],
        description='Statistical method used to calculate the association',
        metadata={'note': 'only one method observed in the reference snapshot; loosen if ENCORE adds another'},
    )
    targetRole: str = pa.Field(
        isin=['Paralog', 'Library', 'GIControl', 'Anchor'],
        description='Role of a target in the genetic interaction test',
    )
    validationReadouts: _VALIDATION_READOUT_STRUCT = pa.Field(  # ty:ignore[invalid-type-form]
        description='List of experimental validation readouts'
    )
    qualityControls: list[str] = pa.Field(
        description='Evidence quality flags',
        metadata={'enum': _QUALITY_FLAGS},
    )
    diseaseId: str = pa.Field(
        str_matches=_ONTOLOGY_ID,
        description='Open Targets disease identifier',
        metadata={'foreign_key': 'disease/id'},
    )
    targetId: str = pa.Field(
        str_matches=_ENSEMBL_GENE_ID,
        description='Open Targets target identifier',
        metadata={'foreign_key': 'target/id'},
    )
    id: str = pa.Field(
        str_matches=_SHA1_HEX,
        unique=True,
        description='Identifer of the disease/target evidence',
        metadata={'primary_key': True},
    )
    evidenceDate: str = pa.Field(
        str_matches=r'^\d{4}-\d{2}-\d{2}$',
        description='Earliest data for the evidence',
    )
    score: float = pa.Field(
        ge=0.0,
        le=1.0,
        description='Score of the evidence reflecting the strength of the disease/target relationship',
    )

    class Config:
        """Schema-level metadata, also lifted from the croissant asset."""

        name = 'evidence_encore'
        description = 'ENCORE genetic-interaction evidence (pts_evidence_postprocess_encore step output)'
        metadata = {'source_step': 'evidence_postprocess_encore', 'datasourceId': 'encore'}
        strict = True

    @pa.check('qualityControls')
    def quality_controls_values_are_known(cls, data: PolarsData) -> pl.LazyFrame:
        """Every `qualityControls` element must be one of `_QUALITY_FLAGS`."""
        return data.lazyframe.select(pl.col(data.key).list.eval(pl.element().is_in(_QUALITY_FLAGS)).list.all())
