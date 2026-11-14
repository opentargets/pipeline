"""Clinical reports from the FDA's approved applications, as openFDA publishes Drugs@FDA.

One report per NDA or BLA approved in its original submission, with the application's active ingredients
as drugs and no disease. Generic applications (ANDAs) and tentative approvals are left out.
"""

import json
import re
import zipfile
from io import BytesIO

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.clinical_mira.datasets import ClinicalReport
from pts.transformers.clinical_mira.schemas import (
    ClinicalProvider,
    ClinicalReportOrigin,
    ClinicalReportType,
    ClinicalSource,
)
from pts.transformers.ontoma_lut.build import DRUG_COLUMNS, build_lut, drug_label_lut
from pts.transformers.utils.disease_mapping import map_entities

APPLICATION_KINDS = ('NDA', 'BLA')
"""Application types that make a report."""

TENTATIVE = 'None (Tentative Approval)'
"""The marketing status of a product that is only tentatively approved."""

APPLICATION_URL = 'https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo='
"""Drugs@FDA overview page of an application; the application's digits complete it."""

_DRUG = pl.Struct({'drugFromSource': pl.String, 'drugId': pl.String})
_DISEASES = pl.List(pl.Struct({'diseaseId': pl.String, 'diseaseFromSource': pl.String}))

_SEPARATOR = r'\s*(?:,|/|;|\sAND\s)\s*'
_BIOLOGIC_SUFFIX = r'-[A-Z]{4}$'


def read_applications(path: str) -> list[dict]:
    """The applications in the openFDA Drugs@FDA zip at `path`.

    Args:
        path: the zip, holding one JSON file with a `results` list.

    Returns:
        The `results` list.
    """
    content, _ = StorageHandle(path).read()
    with zipfile.ZipFile(BytesIO(content)) as archive:
        (member,) = [name for name in archive.namelist() if name.endswith('.json')]
        return json.loads(archive.read(member))['results']


def _original_approval_year(submissions: list[dict]) -> tuple[bool, int | None]:
    """Whether an original submission was approved, and the year of the earliest dated one."""
    dates = [
        s.get('submission_status_date') or ''
        for s in submissions
        if s.get('submission_type') == 'ORIG' and s.get('submission_status') == 'AP'
    ]
    if not dates:
        return False, None
    dated = [d for d in dates if d]
    return True, int(min(dated)[:4]) if dated else None


def approved_ingredients(applications: list[dict]) -> pl.DataFrame:
    """The active ingredients of the NDAs and BLAs approved in their original submission.

    Args:
        applications: Drugs@FDA applications, as :func:`read_applications` returns them.

    Returns:
        One row per application and ingredient: `application` (the number, lowercased), `year` (of the
        earliest approved original submission, null when undated) and `ingredient` (upper case, trimmed).
    """
    rows = []
    for application in applications:
        number = application.get('application_number') or ''
        kind = re.match(r'[A-Z]*', number)
        if kind is None or kind.group(0) not in APPLICATION_KINDS:
            continue
        approved, year = _original_approval_year(application.get('submissions') or [])
        if not approved:
            continue
        for product in application.get('products') or []:
            if product.get('marketing_status') == TENTATIVE:
                continue
            for ingredient in product.get('active_ingredients') or []:
                name = (ingredient.get('name') or '').strip().upper()
                if name:
                    rows.append((number.lower(), year, name))
    return pl.DataFrame(
        rows, schema={'application': pl.String, 'year': pl.Int32, 'ingredient': pl.String}, orient='row'
    ).unique(maintain_order=True)


def _map_names(names: pl.DataFrame, lut: pl.LazyFrame) -> pl.DataFrame:
    """The drug ids of each distinct `name`, null when the dictionary has none."""
    return map_entities(
        names.select('name').unique().lazy(), lut, 'drugIds', 'name', 'label', type_col=pl.lit('CD')
    ).collect()


def query_names(ingredient: pl.Expr) -> pl.Expr:
    """The names to match for an ingredient: combinations split, the biologic suffix dropped."""
    return (
        ingredient
        .str.replace_all(_SEPARATOR, '\x00')
        .str.split('\x00')
        .list.eval(
            pl.element().str.replace(_BIOLOGIC_SUFFIX, '').str.strip_chars().filter(pl.element().str.len_chars() > 0)
        )
    )


def extract_clinical_report(applications: list[dict], drug: pl.DataFrame) -> ClinicalReport:
    """One FDA report per approved NDA or BLA, its ingredients mapped to drugs with the drug dictionary.

    Names the dictionary does not map are logged and dropped; an application left without a drug gives no
    report. The drug ids are set here, so the reports' names never reach drug NER.

    Args:
        applications: Drugs@FDA applications, as :func:`read_applications` returns them.
        drug: Drug molecules, with the `DRUG_COLUMNS` of `intermediate/chembl_molecule`.

    Returns:
        The reports, with `diseases` null.
    """
    lut = build_lut([drug_label_lut(drug.select(DRUG_COLUMNS))]).lazy()
    ingredients = approved_ingredients(applications).with_columns(
        name=pl.col('ingredient').str.replace(_BIOLOGIC_SUFFIX, '').str.strip_chars()
    )
    whole = _map_names(ingredients, lut)
    unmapped_whole = whole.filter(pl.col('drugIds').is_null()).select(pl.col('name').alias('whole'))
    split = (
        ingredients
        .join(unmapped_whole, left_on='name', right_on='whole', how='semi')
        .with_columns(name=query_names(pl.col('ingredient')))
        .explode('name')
        .drop_nulls('name')
    )
    names = pl.concat([ingredients.join(unmapped_whole, left_on='name', right_on='whole', how='anti'), split])
    mapped = pl.concat([whole.drop_nulls('drugIds'), _map_names(split, lut)]).unique('name', keep='first')
    unmapped = names.join(mapped.filter(pl.col('drugIds').is_null()), on='name', how='semi')
    if unmapped.height:
        commonest = unmapped.group_by('name').len().sort(['len', 'name'], descending=[True, False]).head(20)
        count = unmapped['name'].n_unique()
        logger.info(f'drugs@fda: {count} ingredient names unmapped, e.g. {commonest["name"].to_list()}')
    drugs = (
        names
        .join(mapped.drop_nulls('drugIds'), on='name', how='inner')
        .explode('drugIds')
        .select(
            'application',
            'year',
            drug=pl.struct(
                pl.col('name').str.to_lowercase().alias('drugFromSource'),
                pl.col('drugIds').alias('drugId'),
            ),
        )
        .unique(maintain_order=True)
    )
    reports = (
        drugs
        .group_by('application', maintain_order=True)
        .agg(pl.col('year').first(), pl.col('drug').alias('drugs'))
        .select(
            id='application',
            phaseFromSource=pl.lit('approved'),
            year=pl.col('year'),
            origin=pl.lit(ClinicalReportOrigin.REGULATORY.value),
            url=pl.lit(APPLICATION_URL) + pl.col('application').str.replace(r'^[a-z]+', ''),
            type=pl.lit(ClinicalReportType.INDICATION.value),
            source=pl.lit(ClinicalSource.DRUGS_AT_FDA.value),
            provider=pl.lit(ClinicalProvider.FDA.value),
            drugs=pl.col('drugs').cast(pl.List(_DRUG)),
            diseases=pl.lit(None, dtype=_DISEASES),
        )
    )
    logger.info(f'drugs@fda: {reports.height} reports')
    return ClinicalReport(df=reports)
