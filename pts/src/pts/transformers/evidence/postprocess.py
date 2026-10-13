"""The evidence post-processing recipe, parametrised and detached from any one step.

Harmonising, validating, dating and scoring evidence is the same sequence for every datasource;
only a handful of per-datasource values differ. This module holds that sequence once, as an object
you configure and then run. Directions of effect and variant id hashing are not part of it: the
datasources that need them apply them to the `Evidence` it returns.

It reads nothing and writes nothing: `run` takes a `LazyFrame` and returns a lazy `Evidence`. Any
caller that can produce a frame can use it, whether the evidence came off storage or was generated
in memory.

It also imports nothing from `expressions`. Each expression arrives as its own parameter, so
whoever constructs the postprocessor decides where they come from -- a central registry or a
datasource's own definition -- and the recipe depends on neither.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from pts.transformers.evidence.core import Evidence


@dataclass(frozen=True)
class ValidationLuts:
    """The lookup tables every datasource is validated against.

    Passed in already materialised: they are shared across datasources, so building them belongs to
    the caller. `pl.DataFrame` rather than `LazyFrame` because each is joined against repeatedly.
    `publication` is only needed for evidence carrying `literature`.
    """

    disease: pl.DataFrame
    target: pl.DataFrame
    publication: pl.DataFrame | None = None


@dataclass(frozen=True)
class EvidencePostprocessor:
    """One datasource's post-processing, configured but not yet run.

    Args:
        datasource_id: the `datasourceId` to keep. Rows carrying any other value are DROPPED, not
            flagged, so they appear in neither output.
        unique_fields: the fields whose contents identify an evidence row, used to derive `id`.
        score: expression producing the evidence score.
        excluded_biotypes: target biotypes to flag as invalid, for datasources that restrict them.
    """

    datasource_id: str
    unique_fields: list[str]
    score: pl.Expr
    excluded_biotypes: list[str] | None = None

    def run(self, lf: pl.LazyFrame, luts: ValidationLuts) -> Evidence:
        """Apply the full post-processing chain to one datasource's raw evidence.

        The call order is load-bearing: later steps read columns earlier ones add, and
        `validate_uniqueness` in particular depends on the `id` that `assign_evidence_identifier`
        derives.

        Args:
            lf: raw evidence, in its source columns and dtypes; harmonisation to `evidence_schema`
                happens inside `Evidence`.
            luts: the lookup tables to validate against.

        Returns:
            The scored evidence, still lazy and not yet split into valid and invalid.

        Raises:
            ValueError: if `lf` carries `literature` and `luts.publication` is not given.
        """
        if luts.publication is None and 'literature' in lf.collect_schema().names():
            msg = f'{self.datasource_id} evidence carries literature but no publication look-up table was given'
            raise ValueError(msg)
        deduplicated = (
            Evidence(lf)
            .validate_diseases(luts.disease)
            .validate_target(luts.target, self.excluded_biotypes)
            .validate_datasource(self.datasource_id)
            .assign_evidence_identifier(self.unique_fields)
            .validate_uniqueness()
        )
        if luts.publication is not None:
            deduplicated = deduplicated.resolve_publication_date(luts.publication)
        return deduplicated.resolve_evidence_date().calculate_evidence_score(self.score)
