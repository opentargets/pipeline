"""Disease, target and datasource validation for evidence.

Polars port of the `validate_diseases`/`validate_target`/`validate_datasource` methods of
`pts.pyspark.evidence_utils.evidence.Evidence`.
"""

import polars as pl

from pts.transformers.evidence import flags
from pts.transformers.utils.quality_flags import update_quality_flag


def validate_diseases(df: pl.DataFrame, disease_lut: pl.DataFrame) -> pl.DataFrame:
    """Resolve `diseaseFromSourceMappedId` to `diseaseId`, flagging unresolved evidence.

    Args:
        df: evidence with a `diseaseFromSourceMappedId` column.
        disease_lut: output of `luts.build_disease_lut`.

    Returns:
        `df` with a new `diseaseId` column, and `qualityControls` flagged with
        `flags.INVALID_DISEASE` where no disease could be resolved.
    """
    joined = df.join(disease_lut, on='diseaseFromSourceMappedId', how='left')
    return update_quality_flag(joined, pl.col('diseaseId').is_null(), flags.INVALID_DISEASE)


def validate_target(
    df: pl.DataFrame,
    target_lut: pl.DataFrame,
    excluded_biotypes: list[str] | None = None,
) -> pl.DataFrame:
    """Resolve `targetFromSourceId` to `targetId`, flagging unresolved or excluded-biotype evidence.

    Args:
        df: evidence with a `targetFromSourceId` column.
        target_lut: output of `luts.build_target_lut`.
        excluded_biotypes: target biotypes that should be flagged (but not dropped) as invalid.

    Returns:
        `df` with a new `targetId` column, and `qualityControls` flagged with
        `flags.INVALID_TARGET` where no target could be resolved, and `flags.INVALID_BIOTYPE`
        where the resolved target's biotype is in `excluded_biotypes`.
    """
    joined = df.join(
        target_lut.select('targetId', 'biotype', 'targetFromSourceId'),
        on='targetFromSourceId',
        how='left',
    )
    joined = update_quality_flag(joined, pl.col('targetId').is_null(), flags.INVALID_TARGET)
    joined = update_quality_flag(joined, pl.col('biotype').is_in(excluded_biotypes or []), flags.INVALID_BIOTYPE)
    return joined.drop('biotype')


def validate_datasource(df: pl.DataFrame, datasource_id: str) -> pl.DataFrame:
    """Keep only rows whose `datasourceId` matches `datasource_id`.

    Unlike disease/target validation, this is a hard filter, not a quality flag: rows from another
    datasource are dropped outright, matching `Evidence.validate_datasource`.

    Args:
        df: evidence with a `datasourceId` column.
        datasource_id: the only datasource identifier to keep.

    Returns:
        `df` filtered to `datasourceId == datasource_id`.
    """
    return df.filter(pl.col('datasourceId') == datasource_id)
