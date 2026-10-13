"""One datasource's evidence settings and the shared post-processing recipe."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from loguru import logger

from pts.schemas.evidence import evidence_schema
from pts.transformers.evidence.utils.evidence import Evidence
from pts.transformers.evidence.utils.luts import ValidationLuts

#: Columns every raw evidence frame must carry.
REQUIRED_COLUMNS = ['datasourceId', 'targetFromSourceId', 'diseaseFromSourceMappedId']

#: Columns the recipe adds, so `unique_fields` may name them without the raw frame having them.
DERIVED_COLUMNS = {'targetId', 'diseaseId'}


@dataclass(frozen=True)
class EvidenceDatasource:
    """One datasource's post-processing settings.

    Args:
        datasource_id: the `datasourceId` to keep; rows with any other value are dropped.
        unique_fields: the fields that identify an evidence row, used to derive `id`.
        score: expression producing the evidence score.
        excluded_biotypes: target biotypes to flag as invalid.
    """

    datasource_id: str
    unique_fields: list[str]
    score: pl.Expr
    excluded_biotypes: list[str] | None = None

    def check_raw(self, raw: pl.LazyFrame) -> None:
        """Check raw evidence columns; columns outside `evidence_schema` only warn.

        Raises:
            ValueError: if a required or `unique_fields` column is missing.
        """
        names = raw.collect_schema().names()
        missing = [c for c in REQUIRED_COLUMNS if c not in names]
        missing += [c for c in dict.fromkeys(self.unique_fields) if c not in names and c not in DERIVED_COLUMNS]
        if missing:
            msg = f'{self.datasource_id} raw evidence is missing {missing}'
            raise ValueError(msg)
        unknown = [c for c in names if c not in evidence_schema]
        if unknown:
            logger.warning(f'{self.datasource_id} raw evidence has columns outside evidence_schema: {unknown}')

    def postprocess(self, raw: pl.LazyFrame, luts: ValidationLuts) -> Evidence:
        """Validate, identify, deduplicate, date and score raw evidence.

        Returns:
            Scored evidence, lazy and not yet split into valid and invalid.

        Raises:
            ValueError: if `raw` carries `literature` and `luts.publication` is not given.
        """
        if luts.publication is None and 'literature' in raw.collect_schema().names():
            msg = f'{self.datasource_id} evidence carries literature but no publication look-up table was given'
            raise ValueError(msg)
        evidence = (
            Evidence(raw)
            .validate_diseases(luts.disease)
            .validate_target(luts.target, self.excluded_biotypes)
            .validate_datasource(self.datasource_id)
            .assign_evidence_identifier(self.unique_fields)
            .validate_uniqueness()
        )
        if luts.publication is not None:
            evidence = evidence.resolve_publication_date(luts.publication)
        return evidence.resolve_evidence_date().calculate_evidence_score(self.score)
