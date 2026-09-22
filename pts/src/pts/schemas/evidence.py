"""Pandera schemas validating PTS evidence datasets.

Prototype, currently holding one schema (`GwasCredibleSetEvidenceSchema`) for the frame produced
by `pts.transformers.gwas_evidence.gwas_evidence`, right before `write_dataset` writes it to
parquet. Unrelated to the legacy `evidence.json` in this same package, which is a PySpark
`StructType` dump used by `pts.pyspark.evidence_postprocess`'s other datasources -- that one pins
raw column types for reading, this one validates values/constraints on an already-built dataset.

Field descriptions are copied verbatim from
`croissant/src/ot_croissant/assets/recordset/evidence_gwas_credible_sets.json`, which is also
where `isPrimaryKey`/`foreign_key` already exist as first-class metadata -- this schema mirrors
that structure into pandera's own `Field(metadata=...)` instead of inventing new spellings for the
same facts, so the two stay easy to cross-check.

Enum values and numeric bounds below are *inferred* from the transformer code
(`pts.transformers.evidence.*`), not independently confirmed against production data:

* `datatypeId`/`datasourceId` are literal constants for this one datasource
  (`gwas_evidence.py:156-157`).
* `qualityControls` elements are restricted to the flag vocabulary in `evidence/flags.py`.
* `score`/`resourceScore` are bounded to `[0, 1]` by `evidence/scoring.py`'s own validity check.
* `curationDate` uses the exact same loose (non-anchored, non-real-date) regex the transformer
  checks against (`gwas_evidence.py`'s `_CURATION_DATE_REGEX`), including its "9999-99-99 passes"
  quirk -- tightening it here would make the schema reject data the transformer itself accepts.
* `publicationDate` dates come from the literature LUT's `firstPublicationDate` and are asserted
  to be real ISO dates, since that field is not exercised by the same loose-regex code path.
* ID patterns (`id` as sha1 hex, `targetId`/`targetFromSourceId` as Ensembl gene IDs, disease IDs
  as `PREFIX_number`) are inferred from `identifiers.py` and observed fixture/production shapes
  (`cancer_biomarkers.py`'s `EFO_...` constants) -- worth confirming against real output before
  relying on them, and flagged individually below.
"""

import pandera.polars as pa
import polars as pl
from pandera.polars import PolarsData

#: Mirrors `pts.transformers.evidence.flags` -- kept as a plain tuple here (rather than importing
#: the module) so this schema documents the vocabulary it checks against without a hard import
#: dependency; keep in sync with `flags.py` if that vocabulary grows.
_QUALITY_FLAGS = (
    'No valid disease',
    'No valid target',
    'Duplicated',
    'No valid score',
    'Invalid biotype',
)

#: Ensembl gene ID shape. Both `targetId` and `targetFromSourceId` are Ensembl-only for THIS
#: datasource (`geneId` comes straight from gentropy L2G predictions) even though the general
#: cross-datasource evidence spec allows UniProt IDs/gene symbols too -- see
#: `evidence_gwas_credible_sets.json`'s `targetFromSourceId` description.
_ENSEMBL_GENE_ID = r'^ENSG\d{11}$'

#: Ontology-style ID: a namespace prefix, underscore, code. Covers EFO/MONDO/Orphanet/HP as
#: observed elsewhere in the codebase (e.g. `cancer_biomarkers.py`'s `EFO_0020002`), but not
#: verified against every disease ID actually flowing through this specific step.
_ONTOLOGY_ID = r'^[A-Za-z]+_[A-Za-z0-9]+$'

#: sha1 hex digest, as produced by `identifiers.py::_sha1`.
_SHA1_HEX = r'^[0-9a-f]{40}$'

#: Loose date-shaped check, copied as-is from `gwas_evidence.py::_CURATION_DATE_REGEX` -- NOT
#: anchored, NOT a real date parse. Matches the transformer's own (documented) behaviour rather
#: than a stricter ideal.
_LOOSE_DATE = r'\d{4}-\d{2}-\d{2}'

#: A real ISO date, anchored. Used for `publicationDate`, which is sourced from the literature
#: LUT's `firstPublicationDate` rather than the loose-regex-checked `curationDate`.
_ISO_DATE = r'^\d{4}-\d{2}-\d{2}$'


class GwasCredibleSetEvidenceSchema(pa.DataFrameModel):
    """Schema for the `evidence` output of `pts.transformers.gwas_evidence.gwas_evidence`.

    Column order below matches the order columns are actually added in `gwas_evidence.py`'s
    pipeline, not the croissant asset's alphabetical listing.
    """

    datatypeId: str = pa.Field(
        isin=['genetic_association'],
        description='Type of the evidence',
        metadata={'note': 'constant for this datasource; not an enum in the shared evidence spec'},
    )
    datasourceId: str = pa.Field(
        isin=['gwas_credible_sets'],
        description='Identifer of the evidence source',
    )
    targetFromSourceId: str = pa.Field(
        str_matches=_ENSEMBL_GENE_ID,
        description=(
            'Target ID in resource of origin (accepted sources include Ensembl gene ID, Uniprot '
            'ID, gene symbol), only capital letters are accepted'
        ),
    )
    diseaseFromSourceMappedId: str = pa.Field(
        str_matches=_ONTOLOGY_ID,
        description='Mapped Open Targets disease identifier',
    )
    resourceScore: float = pa.Field(
        ge=0.0,
        le=1.0,
        description='Score provided by datasource indicating strength of target-disease association',
    )
    curationDate: str | None = pa.Field(
        nullable=True,
        str_matches=_LOOSE_DATE,
        description='Date of the GWAS study curation',
    )
    studyLocusId: str = pa.Field(
        description='Identifier of the Open Targets Study Locus',
        metadata={'foreign_key': 'credible_set/studyLocusId'},
    )
    literature: list[str] | None = pa.Field(
        nullable=True,
        description='List of PubMed or preprint reference identifiers',
        metadata={'note': 'digits-only (PMID) for this datasource; other sources may carry PPR/AGR ids'},
    )
    diseaseId: str = pa.Field(
        str_matches=_ONTOLOGY_ID,
        description='Open Targets disease identifier',
        metadata={'foreign_key': 'disease/id'},
    )
    qualityControls: list[str] = pa.Field(
        description='Evidence quality flags',
        metadata={'enum': _QUALITY_FLAGS},
    )
    targetId: str = pa.Field(
        str_matches=_ENSEMBL_GENE_ID,
        description='Open Targets target identifier',
        metadata={'foreign_key': 'target/id'},
    )
    id: str = pa.Field(
        str_matches=_SHA1_HEX,
        description='Identifer of the disease/target evidence',
        metadata={'primary_key': True},
    )
    publicationDate: str | None = pa.Field(
        nullable=True,
        str_matches=_ISO_DATE,
        description='Date of the earliest publication supporting the evidence',
    )
    evidenceDate: str | None = pa.Field(
        nullable=True,
        description='Earliest data for the evidence',
    )
    score: float = pa.Field(
        ge=0.0,
        le=1.0,
        description='Score of the evidence reflecting the strength of the disease/target relationship',
    )

    class Config:
        """Schema-level metadata, also lifted from the croissant asset."""

        name = 'evidence_gwas_credible_sets'
        description = 'GWAS credible-set locus-to-gene evidence (pts_gwas_evidence step output)'
        metadata = {'source_step': 'pts_gwas_evidence', 'datasourceId': 'gwas_credible_sets'}
        strict = True

    @pa.check('qualityControls')
    def quality_controls_values_are_known(cls, data: PolarsData) -> pl.LazyFrame:
        """Every `qualityControls` element must be one of `_QUALITY_FLAGS`."""
        return data.lazyframe.select(pl.col(data.key).list.eval(pl.element().is_in(_QUALITY_FLAGS)).list.all())

    @pa.check('literature')
    def literature_values_are_numeric_pmids(cls, data: PolarsData) -> pl.LazyFrame:
        """Every `literature` element must be a bare digit-string PMID, for this datasource."""
        return data.lazyframe.select(
            pl
            .col(data.key)
            .list.eval(pl.element().str.contains(r'^\d+$'))
            .list.all()
            .fill_null(True)  # a null list (no literature) trivially satisfies the check
        )
