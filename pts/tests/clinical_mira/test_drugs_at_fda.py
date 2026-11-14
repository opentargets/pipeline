"""Tests for the FDA clinical reports built from the openFDA Drugs@FDA file."""

import io
import json
import zipfile
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.clinical_mira.providers.drugs_at_fda import (
    approved_ingredients,
    query_names,
    read_applications,
)


def _app(
    number: str,
    orig_ap: str | None,
    ingredients: list[str] | None = None,
    status: str = 'Prescription',
    suppl_ap: str | None = None,
) -> dict:
    """A Drugs@FDA application: one product with `ingredients`, an original submission approved on `orig_ap`."""
    submissions = []
    if orig_ap is not None:
        submissions.append({'submission_type': 'ORIG', 'submission_status': 'AP', 'submission_status_date': orig_ap})
    if suppl_ap is not None:
        submissions.append({'submission_type': 'SUPPL', 'submission_status': 'AP', 'submission_status_date': suppl_ap})
    names = ingredients or ['ASPIRIN']
    return {
        'application_number': number,
        'products': [
            {'marketing_status': status, 'active_ingredients': [{'name': n, 'strength': '1MG'} for n in names]}
        ],
        'submissions': submissions,
    }


def test_read_applications_reads_the_json_in_the_zip(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('drug-drugsfda-0001-of-0001.json', json.dumps({'results': [_app('NDA017001', '20240314')]}))
    (tmp_path / 'drugs_at_fda.json.zip').write_bytes(buffer.getvalue())

    applications = read_applications(str(tmp_path / 'drugs_at_fda.json.zip'))

    assert [a['application_number'] for a in applications] == ['NDA017001']


def test_keeps_approved_nda_and_bla_only() -> None:
    apps = [
        _app('NDA017001', orig_ap='20240314'),
        _app('BLA761001', orig_ap='20200101'),
        _app('ANDA070001', orig_ap='19900101'),
        _app('NDA017002', orig_ap=None),
    ]
    assert sorted(approved_ingredients(apps)['application'].unique()) == ['bla761001', 'nda017001']


def test_year_is_the_earliest_original_approval() -> None:
    out = approved_ingredients([_app('NDA017001', orig_ap='20240314', suppl_ap='20180101')])
    assert out['year'].unique().to_list() == [2024]


def test_original_approval_without_a_date_counts_with_a_null_year() -> None:
    out = approved_ingredients([_app('NDA017001', orig_ap='')])
    assert out.height == 1
    assert out['year'][0] is None


def test_tentative_products_contribute_nothing() -> None:
    apps = [_app('NDA017001', orig_ap='20240314', status='None (Tentative Approval)')]
    assert approved_ingredients(apps).is_empty()


def test_ingredients_are_upper_case_and_trimmed() -> None:
    out = approved_ingredients([_app('NDA017001', orig_ap='20240314', ingredients=[' Resmetirom '])])
    assert out['ingredient'].to_list() == ['RESMETIROM']


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        ('ELEXACAFTOR, IVACAFTOR, TEZACAFTOR', ['ELEXACAFTOR', 'IVACAFTOR', 'TEZACAFTOR']),
        ('DARUNAVIR AND COBICISTAT', ['DARUNAVIR', 'COBICISTAT']),
        ('CONJUGATED ESTROGENS/MEDROXYPROGESTERONE ACETATE', ['CONJUGATED ESTROGENS', 'MEDROXYPROGESTERONE ACETATE']),
        ('NARSOPLIMAB-WUUG', ['NARSOPLIMAB']),
        ('TELISOTUZUMAB VEDOTIN-TLLV', ['TELISOTUZUMAB VEDOTIN']),
        ('FLUDEOXYGLUCOSE F-18', ['FLUDEOXYGLUCOSE F-18']),
        ('TECHNETIUM TC-99M SESTAMIBI', ['TECHNETIUM TC-99M SESTAMIBI']),
        ('RESMETIROM', ['RESMETIROM']),
    ],
)
def test_query_names(raw: str, expected: list[str]) -> None:
    assert pl.select(query_names(pl.lit(raw))).item().to_list() == expected
