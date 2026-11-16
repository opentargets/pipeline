import polars as pl
import polars_hash as plh

from pts.transformers.clinical_mira.datasets import ClinicalReport
from pts.transformers.clinical_mira.schemas import (
    ClinicalProvider,
    ClinicalReportOrigin,
    ClinicalReportType,
    ClinicalSource,
)

TTD_DRUG_URL = 'https://ttd.idrblab.cn/data/drug/details/'
"""Prefix of a TTD drug page; the TTD drug id completes it."""


def ttd_report_id(ttd_id: pl.Expr, indication: pl.Expr) -> pl.Expr:
    """The id of the TTD report for a TTD drug id and an indication text.

    Args:
        ttd_id: the TTD drug id (`TTDDRUID`).
        indication: the indication text, as TTD writes it; case does not matter.

    Returns:
        The lowercase hex SHA-256 of the drug id followed by the lowercased indication.
    """
    return plh.concat_str(ttd_id, indication.str.to_lowercase()).chash.sha2_256().str.to_lowercase()


def extract_indication(ttd_input: str | list[str]) -> pl.DataFrame:
    """Extract indications from TTD Indications dataset.

    Args:
        ttd_input: Path to TTD Indications dataset or list of strings containing the dataset.
    """
    if isinstance(ttd_input, str):
        # If input is a file path, read the file
        with open(ttd_input, encoding='utf-8') as file:
            lines = file.readlines()
    else:
        # If input has already been read, use it as is
        lines = ttd_input

    # Initialize variables
    data = []
    current_drug = {'TTDDRUID': None, 'DRUGNAME': None}

    # Parse the lines
    for line in lines:
        line = line.strip()
        if line.startswith('TTDDRUID'):
            current_drug['TTDDRUID'] = line.split('\t')[1]
        elif line.startswith('DRUGNAME'):
            current_drug['DRUGNAME'] = line.split('\t')[1]
        elif line.startswith('INDICATI'):
            parts = line.split('\t')
            indication = parts[1]
            icd = parts[2].replace('ICD-11: ', '').strip()
            clinical_stage = parts[3]
            # Append a new row to the data list
            data.append([
                current_drug['TTDDRUID'],
                current_drug['DRUGNAME'],
                indication,
                icd,
                clinical_stage,
            ])

    return pl.DataFrame(
        data,
        schema=[
            'ttd_id',
            'drugFromSource',
            'diseaseFromSource',
            'icd11_id',
            'clinical_stage',
        ],
        orient='row',
    ).slice(1)  # remove first line that includes column names


def extract_clinical_report(
    indications: pl.DataFrame,
) -> ClinicalReport:
    """Extract clinical reports from TTD drug/disease dataset."""
    reports = indications.select(
        id=ttd_report_id(pl.col('ttd_id'), pl.col('diseaseFromSource')),
        origin=pl.lit(ClinicalReportOrigin.CURATED_RESOURCE),
        url=pl.concat_str([pl.lit(TTD_DRUG_URL), pl.col('ttd_id')]),
        phaseFromSource=pl.col('clinical_stage').str.to_lowercase(),
        drug=pl.struct(
            pl.col('drugFromSource').str.to_lowercase(),
            pl.lit(None, dtype=pl.String).alias('drugId'),
        ),
        disease=pl.struct(
            pl.lit(None, dtype=pl.String).alias('diseaseId'),
            pl.col('diseaseFromSource').str.to_lowercase(),
        ),
        type=pl.lit(ClinicalReportType.INDICATION.value),
        source=pl.lit(ClinicalSource.TTD.value),
        provider=pl.lit(ClinicalProvider.TTD.value),
    ).unique()

    return ClinicalReport(
        df=(
            reports.group_by([c for c in reports.columns if c not in ['disease', 'drug']]).agg(
                pl.col('disease').unique().alias('diseases'),
                pl.col('drug').unique().alias('drugs'),
            )
        )
    )
