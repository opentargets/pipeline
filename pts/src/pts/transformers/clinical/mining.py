"""The TTD, EMA and PMDA reports, built the same way for staging and for the release."""

from collections.abc import Callable, Iterable
from io import BytesIO

import pdfplumber
import polars as pl
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.clinical.datasets import ClinicalReport
from pts.transformers.clinical.providers.ema import extract_clinical_report as extract_ema_clinical_report
from pts.transformers.clinical.providers.pmda import extract_clinical_report as extract_pmda_clinical_report
from pts.transformers.clinical.providers.pmda import parse_pmda_approvals
from pts.transformers.clinical.providers.ttd import extract_clinical_report as extract_ttd_clinical_report
from pts.transformers.clinical.providers.ttd import extract_indication as extract_ttd_indication
from pts.transformers.utils.dataset import read_spreadsheet

EMA_SHEET_NAME = 'Medicine'


def read_ema_sheet(path: str) -> pl.DataFrame:
    """The EMA list of medicines, every cell a string, the sheet's banner row as column names.

    Blank rows are dropped. Blanks and `read_spreadsheet`'s missing-value strings are null;
    a whitespace-only cell is ''.

    Args:
        path: the EMA medicines workbook.

    Returns:
        The `Medicine` sheet, with the real column names in its first row.
    """
    sheet = read_spreadsheet(path, EMA_SHEET_NAME, infer_types=False)
    sheet = sheet.filter(pl.any_horizontal(pl.all().is_not_null()))
    return sheet.with_columns(
        pl.when(pl.all().str.strip_chars().eq('')).then(pl.lit('')).otherwise(pl.all()).name.keep()
    )


def build_mining_reports(
    ttd: str,
    ema: str,
    pmda: str,
    extract_diseases: Callable[[pl.Series], pl.Series],
) -> dict[str, ClinicalReport]:
    """The TTD, EMA and PMDA reports.

    Args:
        ttd: the TTD indications text file.
        ema: the EMA medicines workbook.
        pmda: the PMDA approvals PDF.
        extract_diseases: disease NER for the EMA and PMDA indication texts, mapping a String series of texts
            to a ``List(String)`` series of the same length.

    Returns:
        The reports, keyed `ttd`, `ema` and `pmda`.
    """
    pmda_bytes, _ = StorageHandle(pmda).read()
    pmda_pdf = pdfplumber.open(BytesIO(pmda_bytes))
    ttd_lines = StorageHandle(ttd).open('r').read().split('\n')
    return {
        'pmda': extract_pmda_clinical_report(df=parse_pmda_approvals(pmda_pdf), extract_diseases=extract_diseases),
        'ttd': extract_ttd_clinical_report(indications=extract_ttd_indication(ttd_lines)),
        'ema': extract_ema_clinical_report(read_ema_sheet(ema), extract_diseases=extract_diseases),
    }


def report_drug_labels(reports: Iterable[pl.DataFrame]) -> pl.Series:
    """The distinct drug labels of `reports` that have no drug id, sorted.

    Args:
        reports: report frames with a `drugs` list of `drugFromSource` and `drugId` structs.

    Returns:
        The distinct non-null `drugFromSource` whose `drugId` is null.
    """
    labels = [
        report
        .select(pl.col('drugs').explode().struct.unnest())
        .filter(pl.col('drugId').is_null())
        .get_column('drugFromSource')
        for report in reports
    ]
    if not labels:
        return pl.Series('drugFromSource', [], dtype=pl.String)
    return pl.concat(labels).drop_nulls().unique().sort()


class RecordingNer:
    """A disease NER that finds nothing and remembers every text it is asked about."""

    def __init__(self) -> None:
        self.texts: set[str] = set()

    def __call__(self, texts: pl.Series) -> pl.Series:
        self.texts.update(texts.to_list())
        return pl.Series(texts.name, [[] for _ in range(texts.len())], dtype=pl.List(pl.String))


def staged_lookup(staged: pl.DataFrame, key: str, value: str, what: str) -> Callable[[pl.Series], pl.Series]:
    """A NER stand-in that reads its results from `staged`.

    Args:
        staged: the staged results.
        key: the column of the texts.
        value: the `List(String)` column of the results.
        what: what the texts are, for the error.

    Returns:
        A function mapping a String series of texts to a `List(String)` series of the same length.

    Raises:
        ValueError: from the returned function, naming `what`, how many texts are not staged and up to 5 of them.
    """
    results = dict(zip(staged[key].to_list(), staged[value].to_list(), strict=True))

    def lookup(texts: pl.Series) -> pl.Series:
        wanted = texts.to_list()
        missing = sorted({t for t in wanted if results.get(t) is None})
        if missing:
            raise ValueError(f'{len(missing)} {what} not staged, e.g. {missing[:5]}')
        return pl.Series(texts.name, [results[t] for t in wanted], dtype=pl.List(pl.String))

    return lookup
