"""Tests for the ClinicalIndication column order."""

import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl

from pts.transformers.clinical import ClinicalIndication

COLUMNS = [
    'id',
    'drugName',
    'diseaseName',
    'maxClinicalStage',
    'mappingStatus',
    'clinicalReportIds',
    'drugId',
    'diseaseId',
]

INDICATION_COLUMNS_SCRIPT = """
import json
from pts.transformers.clinical import ClinicalIndication
from tests.clinical.test_clinical_indication_dataset import _report
print(json.dumps(ClinicalIndication.from_report(_report()).df.columns))
"""


def _report() -> pl.DataFrame:
    return pl.DataFrame({
        'id': ['NCT1', 'NCT2'],
        'clinicalStage': ['PHASE_2', 'APPROVAL'],
        'origin': ['CLINICAL_TRIAL', 'CLINICAL_TRIAL'],
        'type': ['INDICATION', 'INDICATION'],
        'source': ['ClinicalTrials.gov', 'ClinicalTrials.gov'],
        'provider': ['AACT', 'AACT'],
        'drugs': [
            [{'drugFromSource': 'aspirin', 'drugId': 'CHEMBL25'}],
            [{'drugFromSource': 'unknownol', 'drugId': None}],
        ],
        'diseases': [
            [{'diseaseFromSource': 'pain', 'diseaseId': 'EFO_0003843'}],
            [{'diseaseFromSource': 'headache', 'diseaseId': None}],
        ],
    })


def test_from_report_columns_are_fixed() -> None:
    assert ClinicalIndication.from_report(_report()).df.columns == COLUMNS


def test_from_report_columns_do_not_depend_on_the_hash_seed() -> None:
    orders = []
    for seed in ('0', '2'):
        result = subprocess.run(
            [sys.executable, '-c', INDICATION_COLUMNS_SCRIPT],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).parents[2],
            env={**os.environ, 'PYTHONHASHSEED': seed},
        )
        assert result.returncode == 0, result.stderr
        orders.append(json.loads(result.stdout))

    assert orders == [COLUMNS, COLUMNS]
