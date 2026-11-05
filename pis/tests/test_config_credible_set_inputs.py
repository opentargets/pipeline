"""The credible set steps of PIS stage each source's study, credible sets and credible set variants.

The steps are `gwas`, `gwas_ppp`, `molqtl` and `molqtl_ppp`; each source lands in its own folder,
`input/<kind>/<source>/<dataset>`.

The credible set variants of a source are only valid next to the credible sets of the same run,
so all three datasets of a source must come from one run folder.
"""

from pathlib import Path
from string import Template

from otter.config.yaml import parse_yaml

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'
"""The config file that ships in the image, not a fixture."""

DATASETS = ('study', 'credible_set', 'credible_set_variant')

EXPECTED_RUNS = {
    'gwas': {
        'gwas/gcca/{dataset}': 'gs://gwas_catalog_top_hits/prod/20261028T234317/output',
        'gwas/gcss_pics/{dataset}': 'gs://gwas_catalog_sumstats_pics/prod/20261030T072623/output',
        'gwas/gcss_susie/{dataset}': 'gs://gwas_catalog_sumstats_susie/prod/20261101T152321/output',
        'gwas/finngen_gwas/{dataset}': 'gs://finngen_data/gwas/prod/20261029T152531/output',
    },
    'gwas_ppp': {
        'gwas/finngen_meta/{dataset}': 'gs://finngen_data/meta/three-way/prod/20261102T181409/output',
    },
    'molqtl': {
        'molqtl/eqtl_catalogue/{dataset}': 'gs://eqtl_catalogue_data/r8/prod/20261030T164426/output',
        'molqtl/ukb_ppp_eur/{dataset}': 'gs://ukb_ppp_eur_data/prod/20261026T161010/output',
        'molqtl/finngen_multiome/{dataset}': 'gs://finngen_data/multiome/prod/20261029T125742/output',
    },
    'molqtl_ppp': {
        'molqtl/otar2077/microglia/{dataset}': 'gs://eqtl_catalogue_data/otar2065_microglia/prod/20261103T163322/output',
        'molqtl/eQTLGen/single_cell_eqtl/{dataset}': 'gs://eqtlgen_data/prod/20261103T135048/output/single_cell_eqtl',
        'molqtl/eQTLGen/coeqtl/{dataset}': 'gs://eqtlgen_data/prod/20261103T135048/output/coeqtl',
        'molqtl/deCODE_proteomics/{dataset}': 'gs://decode_data/prod/20261103T122701/output',
        'molqtl/sle_map/{dataset}': 'gs://sle_map_data/prod/20261103T145657/output',
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


def _copies(step: str) -> dict[str, str]:
    """The copies of a step, with explodes expanded and the scratchpad run timestamps filled in."""
    config = parse_yaml(CONFIG_PATH)
    scratchpad = config['scratchpad']
    copies = {}
    for task in config['steps'][step]:
        for each, spec in [(each, spec) for each in task.get('foreach', []) for spec in task['do']] or [(None, task)]:
            if spec['name'].startswith('copy_many '):
                values = {**scratchpad, 'each': each} if each else scratchpad
                copies[Template(spec['destination']).substitute(values)] = Template(spec['sources']).substitute(values)
    return copies


def test_gwas_sources_stage_new_schema_credible_sets() -> None:
    assert _copies('gwas') == _expected('gwas')


def test_molqtl_sources_stage_new_schema_credible_sets() -> None:
    assert _copies('molqtl') == _expected('molqtl')


def test_gwas_ppp_sources_stage_new_schema_credible_sets() -> None:
    assert _copies('gwas_ppp') == _expected('gwas_ppp')


def test_molqtl_ppp_sources_stage_new_schema_credible_sets() -> None:
    assert _copies('molqtl_ppp') == _expected('molqtl_ppp')
