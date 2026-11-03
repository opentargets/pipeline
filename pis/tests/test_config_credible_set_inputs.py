"""The `gwas` and `molqtl` steps stage each source's study, credible sets and credible set variants.

The credible set variants of a source are only valid next to the credible sets of the same run,
so all three datasets of a source must come from one run folder.
"""

from pathlib import Path
from typing import Any

from otter.config.yaml import parse_yaml

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'
"""The config file that ships in the image, not a fixture."""

DATASETS = ('study', 'credible_set', 'credible_set_variant')

EXPECTED_RUNS = {
    'gwas': {
        'gwas/{dataset}/gcca': 'gs://gwas_catalog_top_hits/prod/20261028T234317/output',
        'gwas/{dataset}/gcss_pics': 'gs://gwas_catalog_sumstats_pics/prod/20261030T072623/output',
        'gwas/{dataset}/gcss_susie': 'gs://gwas_catalog_sumstats_susie/prod/20261101T152321/output',
        'gwas/{dataset}/finngen': 'gs://finngen_data/gwas/prod/20261029T152531/output',
    },
    'molqtl': {
        'molqtl/{dataset}/eqtl_catalogue/r8': 'gs://eqtl_catalogue_data/r8/prod/20261030T164426/output',
        'molqtl/{dataset}/ukb_ppp_eur': 'gs://ukb_ppp_eur_data/prod/20261026T161010/output',
        'molqtl/{dataset}/finngen_multiome': 'gs://finngen_data/multiome/prod/20261029T125742/output',
    },
}


def _expected(step: str) -> dict[str, str]:
    expected = {}
    for destination, run in EXPECTED_RUNS[step].items():
        for dataset in DATASETS:
            # eQTL Catalogue keeps its study under `study_index`.
            folder = 'study_index' if run.startswith('gs://eqtl_catalogue_data') and dataset == 'study' else dataset
            expected[f'input/{destination.format(dataset=dataset)}'] = f'{run}/{folder}/*.parquet'
    return expected


def _copies(tasks: list[dict[str, Any]]) -> dict[str, str]:
    return {task['destination']: task['sources'] for task in tasks if task['name'].startswith('copy_many ')}


def test_gwas_sources_stage_new_schema_credible_sets() -> None:
    assert _copies(parse_yaml(CONFIG_PATH)['steps']['gwas']) == _expected('gwas')


def test_molqtl_sources_stage_new_schema_credible_sets() -> None:
    assert _copies(parse_yaml(CONFIG_PATH)['steps']['molqtl']) == _expected('molqtl')
