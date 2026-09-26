"""Pandera schema validating the PTS biosample index dataset.

For the frame produced by `pts.transformers.biosample.biosample`, right before `write_dataset`
writes it to parquet. Field descriptions are copied verbatim from
`croissant/src/ot_croissant/assets/recordset/biosample.json`, which also has `biosampleId`'s
`isPrimaryKey` as first-class metadata -- mirrored here into `pa.Field(metadata=...)` the same way
as `pts.schemas.evidence.GwasCredibleSetEvidenceSchema`, so the two stay easy to cross-check.
Unlike that schema, croissant carries no `foreign_key`/`[bioregistry:...]` tag for any field here
(checked), so there's no PK/FK cross-referencing or bioregistry metadata to add.

`biosampleId` is intentionally left without a `str_matches` pattern: verified against a real
production `output/biosample` (35,477 rows) that the id space spans dozens of ontology prefixes
(`UBERON`, `CL`, `EFO`, `GO`, `PR`, plus ~35 rarer ones such as `BFO`/`RO`/`BSPO` pulled in as
imported or referenced terms) -- there is no single regex that's actually true of production data,
so only non-null + `unique=True` (this is the dataset's primary key) are asserted.

All 6 relationship columns (`xrefs`, `synonyms`, `parents`, `ancestors`, `descendants`, `children`)
are marked `nullable=True`. Real reference data shows `description`/`parents`/`children` can
genuinely be null while `xrefs`/`synonyms`/`ancestors`/`descendants` empirically never are -- but I
could not fully reconcile that split against gentropy's own merge logic (which nominally collapses
every list column to `[]`, never null, once source ontologies are merged) without the historical
code that actually produced that reference, so nullable is asserted uniformly rather than a
guarantee I can't verify.
"""

import pandera.polars as pa


class BiosampleIndexSchema(pa.DataFrameModel):
    """Schema for the output of `pts.transformers.biosample.biosample`."""

    biosampleId: str = pa.Field(
        unique=True,
        description='Unique identifier for the biosample',
        metadata={'primary_key': True},
    )
    biosampleName: str = pa.Field(description='Name of the biosample')
    description: str | None = pa.Field(nullable=True, description='Description of the biosample')
    xrefs: list[str] | None = pa.Field(
        nullable=True,
        description='Cross-reference IDs from other ontologies',
    )
    synonyms: list[str] | None = pa.Field(
        nullable=True,
        description='List of synonymous names for the term',
    )
    parents: list[str] | None = pa.Field(
        nullable=True,
        description='Direct parent biosample IDs in the ontology',
    )
    ancestors: list[str] | None = pa.Field(
        nullable=True,
        description='List of ancestor biosample IDs in the ontology',
    )
    children: list[str] | None = pa.Field(
        nullable=True,
        description='Direct child biosample IDs in the ontology',
    )
    descendants: list[str] | None = pa.Field(
        nullable=True,
        description='List of descendant biosample IDs in the ontology',
    )

    class Config:
        """Schema-level metadata, also lifted from the croissant asset."""

        name = 'biosample'
        description = 'Biosample index merged from Cell Ontology, Uberon and EFO (pts_biosample step output)'
        metadata = {'source_step': 'pts_biosample'}
        strict = True
