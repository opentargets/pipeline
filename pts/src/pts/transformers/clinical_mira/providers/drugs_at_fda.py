"""Clinical reports from the FDA's approved applications, as openFDA publishes Drugs@FDA.

One report per NDA or BLA approved in its original submission, with the application's active ingredients
as drugs and no disease. Generic applications (ANDAs) and tentative approvals are left out.
"""

import json
import re
import zipfile
from io import BytesIO

import polars as pl
from otter.storage.synchronous.handle import StorageHandle

APPLICATION_KINDS = ('NDA', 'BLA')
"""Application types that make a report."""

TENTATIVE = 'None (Tentative Approval)'
"""The marketing status of a product that is only tentatively approved."""

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
