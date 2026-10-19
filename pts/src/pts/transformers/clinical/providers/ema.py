"""Extraction of clinical reports from the EMA list of human medicines."""

from collections.abc import Callable

import polars as pl
from loguru import logger

from pts.transformers.clinical.datasets import ClinicalReport
from pts.transformers.clinical.helpers import extract_per_distinct_text
from pts.transformers.clinical.schemas import (
    ClinicalProvider,
    ClinicalReportOrigin,
    ClinicalReportType,
    ClinicalSource,
)


def extract_marketing_year(marketing_date: str | None) -> int | None:
    """Extract the year from an EMA marketing authorisation date (``dd/mm/yyyy``)."""
    if not marketing_date:
        return None
    try:
        return pl.Series([marketing_date]).str.to_datetime(format='%d/%m/%Y', strict=False).dt.year().item()
    except ValueError:
        return None


def extract_clinical_report(
    sheet: pl.DataFrame,
    extract_diseases: Callable[[pl.Series], pl.Series],
) -> ClinicalReport:
    """Extract clinical reports from the EMA list of human drugs.

    Args:
        sheet: the ``Medicine`` sheet as read, with the column names in its first row.
        extract_diseases: disease NER, mapping a String series of texts to a ``List(String)`` series of the
            same length.

    Returns:
        ClinicalReport with one report per EMA product.
    """
    raw_df = sheet.clone()
    raw_df.columns = list(raw_df.iter_rows().__next__())  # Assign columns names from first row

    human_indications = raw_df.slice(1).filter(  # drop header
        pl.col('Category') == 'Human'
    )

    logger.info('(ema): apply ner to extract diseases from therapeutic indications')
    ner_extracted_indication = (
        extract_per_distinct_text(
            human_indications.with_columns(
                therapeutic_indication=pl.col('Therapeutic indication').fill_null('').str.strip_chars()
            ),
            input_col='therapeutic_indication',
            output_col='extracted_diseases',
            extract=extract_diseases,
        )
        # Explode the extracted diseases
        .explode('extracted_diseases', empty_as_null=True)
        .rename({'extracted_diseases': 'extracted_disease'})
    )

    reports = (
        ner_extracted_indication
        .select(
            id=pl.col('EMA product number').str.to_lowercase(),
            phaseFromSource=pl.col('Medicine status').str.to_lowercase(),
            origin=pl.lit(ClinicalReportOrigin.REGULATORY.value),
            type=pl.lit(ClinicalReportType.INDICATION.value),
            drugFromSource=pl
            .coalesce(
                'International non-proprietary name (INN) / common name',
                'Active substance',
                'Name of medicine',
            )
            .str.to_lowercase()
            .str.split(';'),
            diseaseFromSource=pl
            .coalesce(
                # Prioritise MeSH terms over automatically extracted diseases
                'Therapeutic area (MeSH)',
                'extracted_disease',
            )
            .str.to_lowercase()
            .str.split(';'),
            source=pl.lit(ClinicalSource.EMA_HUMAN_DRUGS.value),
            provider=pl.lit(ClinicalProvider.EMA.value),
            url=pl.col('Medicine URL'),
            year=pl.col('Marketing authorisation date').map_elements(extract_marketing_year, return_dtype=pl.Int32),
        )
        .explode('drugFromSource', empty_as_null=True)
        .explode('diseaseFromSource', empty_as_null=True)
        # After extracting diseases, some rows may have null values (25 currently)
        .filter(pl.col('drugFromSource').is_not_null() & pl.col('diseaseFromSource').is_not_null())
        .with_columns(
            disease=pl.struct(
                pl.lit(None, dtype=pl.String).alias('diseaseId'),
                pl.col('diseaseFromSource'),
            ),
            drug=pl.struct(
                pl.col('drugFromSource'),
                pl.lit(None, dtype=pl.String).alias('drugId'),
            ),
        )
        .drop(['diseaseFromSource', 'drugFromSource'])
        .unique()
    )

    return ClinicalReport(
        df=(
            reports.group_by([c for c in reports.columns if c not in ['disease', 'drug']]).agg(
                pl.col('disease').unique().alias('diseases'),
                pl.col('drug').unique().alias('drugs'),
            )
        )
    )
