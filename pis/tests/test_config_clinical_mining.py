"""The upstream copies of the `clinical_mining` step."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'
"""The config file that ships in the image, not a fixture."""

TTD_URL = 'https://ttd.idrblab.cn/files/download/P1-05-Drug_disease.txt'
EMA_URL = 'https://www.ema.europa.eu/en/documents/report/medicines-output-medicines-report_en.xlsx'
PMDA_URL = 'https://www.pmda.go.jp/files/${pmda_file}.pdf'
CHEMBL_MAPPING = 'gs://ot-team/irene/clinical_mining/2026-05-27/input/clinical_report/chembl_mapping.parquet'


def test_clinical_mining_copies_upstream() -> None:
    """TTD, EMA and PMDA come from upstream, PMDA by its pinned file; only the ChEMBL mapping is frozen."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert config['scratchpad']['pmda_file'].isdigit()

    tasks = config['steps']['clinical_mining']
    assert [t['name'] for t in tasks[:4]] == ['copy ttd', 'copy ema medicines', 'copy pmda', 'copy chembl mapping']

    ttd, ema, pmda, chembl = tasks[:4]
    assert (ttd['source'], ttd['destination']) == (TTD_URL, 'input/ttd.txt')
    assert (ema['source'], ema['destination']) == (EMA_URL, 'input/ema_medicines.xlsx')
    assert (pmda['source'], pmda['destination']) == (PMDA_URL, 'input/pmda.pdf')
    assert (chembl['source'], chembl['destination']) == (CHEMBL_MAPPING, 'input/chembl_mapping.parquet')

    assert [t['name'] for t in tasks if 'gs://ot-team' in str(t)] == ['copy chembl mapping']
