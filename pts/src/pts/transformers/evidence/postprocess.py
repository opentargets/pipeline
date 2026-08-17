"""The evidence post-processing recipe, parametrised and detached from any one step.

Harmonising, validating, dating and scoring evidence is the same sequence for every datasource;
only a handful of per-datasource values differ. This module holds that sequence once, as an object
you configure and then run.

It reads nothing and writes nothing: `run` takes a `LazyFrame` and returns `LazyFrame`s. Any caller
that can produce a frame can use it, whether the evidence came off storage or was generated in
memory.

It also does not import `EXPRESSIONS`. Expressions arrive as a parameter, so whoever constructs the
postprocessor decides where they come from -- a central registry or a datasource's own definition.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from pts.transformers.evidence.core import Evidence
from pts.transformers.evidence.expressions import DatasourceExpressions


@dataclass(frozen=True)
class ValidationLuts:
    """The lookup tables every datasource is validated against.

    Passed in already materialised: they are shared across datasources, so building them belongs to
    the caller. `pl.DataFrame` rather than `LazyFrame` because each is joined against repeatedly.
    """

    disease: pl.DataFrame
    target: pl.DataFrame
    publication: pl.DataFrame


@dataclass(frozen=True)
class PostprocessedEvidence:
    """The two halves of a post-processed datasource, split on quality control.

    Both stay lazy. Each collects the upstream chain independently, so whether that work is done
    once or twice is the caller's decision.
    """

    valid: pl.LazyFrame
    invalid: pl.LazyFrame


@dataclass(frozen=True)
class EvidencePostprocessor:
    """One datasource's post-processing, configured but not yet run.

    Args:
        datasource_id: the `datasourceId` to keep. Rows carrying any other value are DROPPED, not
            flagged, so they appear in neither output.
        unique_fields: the fields whose contents identify an evidence row, used to derive `id`.
        expressions: the score and direction-of-effect expressions for this datasource.
        excluded_biotypes: target biotypes to flag as invalid, for datasources that restrict them.
    """

    datasource_id: str
    unique_fields: list[str]
    expressions: DatasourceExpressions
    excluded_biotypes: list[str] | None = None

    def run(self, lf: pl.LazyFrame, luts: ValidationLuts) -> PostprocessedEvidence:
        """Apply the full post-processing chain to one datasource's raw evidence.

        The call order is load-bearing: later steps read columns earlier ones add, and
        `validate_uniqueness` in particular depends on the `id` that `assign_evidence_identifier`
        derives.

        Args:
            lf: raw evidence, in its source columns and dtypes; harmonisation to `evidence_schema`
                happens inside `Evidence`.
            luts: the lookup tables to validate against.

        Returns:
            The valid and invalid halves, both still lazy.
        """
        processed = (
            Evidence(lf)
            .validate_diseases(luts.disease)
            .validate_target(luts.target, self.excluded_biotypes)
            .validate_datasource(self.datasource_id)
            .assign_evidence_identifier(self.unique_fields)
            .validate_uniqueness()
            .resolve_publication_date(luts.publication)
            .resolve_evidence_date()
            .calculate_evidence_score(self.expressions.score)
            .assign_direction_on_trait(self.expressions.direction_on_trait)
            .assign_direction_on_target(self.expressions.direction_on_target, luts.target)
            .hash_long_variant_identifiers()
        )
        return PostprocessedEvidence(valid=processed.valid(), invalid=processed.invalid())
