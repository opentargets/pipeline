"""The upstream fetches of the `clinical_mining` step."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'
"""The config file that ships in the image, not a fixture."""

TTD_URL = 'https://ttd.idrblab.cn/files/download/P1-05-Drug_disease.txt'
EMA_URL = 'https://www.ema.europa.eu/en/documents/report/medicines-output-medicines-report_en.xlsx'
PMDA_PAGE = 'https://www.pmda.go.jp/english/review-services/reviews/approved-information/drugs/0002.html'
CHEMBL_MAPPING = 'gs://ot-team/irene/clinical_mining/2026-05-27/input/clinical_report/chembl_mapping.parquet'


def test_clinical_mining_fetches_upstream() -> None:
    """TTD, EMA and PMDA come from upstream, dated by the run; only the ChEMBL mapping is frozen."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    tasks = config['steps']['clinical_mining']
    assert [t['name'] for t in tasks[:4]] == [
        'fetch_once ttd',
        'fetch_once ema medicines',
        'fetch_pmda pmda',
        'copy chembl mapping',
    ]

    ttd, ema, pmda, chembl = tasks[:4]
    assert ttd['source'] == TTD_URL
    assert ttd['destination'] == 'input/ttd.txt'
    assert ema['source'] == EMA_URL
    assert ema['destination'] == 'input/ema_medicines.xlsx'
    assert pmda['page'] == PMDA_PAGE
    assert pmda['destination'] == 'input/pmda.pdf'
    for task in (ttd, ema, pmda):
        assert task['as_of'] == '${clinical_mining_version}'
    assert chembl['source'] == CHEMBL_MAPPING
    assert chembl['destination'] == 'input/chembl_mapping.parquet'

    assert [t['name'] for t in tasks if 'gs://ot-team' in str(t)] == ['copy chembl mapping']
