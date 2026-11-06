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
        'gwas/gcca/{dataset}': 'gs://gwas_catalog_top_hits/prod/${staging_gcca_ts}/output',
        'gwas/gcss_pics/{dataset}': 'gs://gwas_catalog_sumstats_pics/prod/${staging_gcss_pics_ts}/output',
        'gwas/gcss_susie/{dataset}': 'gs://gwas_catalog_sumstats_susie/prod/${staging_gcss_susie_ts}/output',
        'gwas/finngen_gwas/{dataset}': 'gs://finngen_data/gwas/prod/${staging_finngen_gwas_ts}/output',
    },
    'gwas_ppp': {
        'gwas/finngen_meta/{dataset}': 'gs://finngen_data/meta/three-way/prod/${staging_finngen_meta_ts}/output',
    },
    'molqtl': {
        'molqtl/eqtl_catalogue/{dataset}': 'gs://eqtl_catalogue_data/r8/prod/${staging_eqtl_catalogue_ts}/output',
        'molqtl/ukb_ppp_eur/{dataset}': 'gs://ukb_ppp_eur_data/prod/${staging_ukb_ppp_eur_ts}/output',
        'molqtl/finngen_multiome/{dataset}': 'gs://finngen_data/multiome/prod/${staging_finngen_multiome_ts}/output',
    },
    'molqtl_ppp': {
        'molqtl/otar2077/microglia/{dataset}': 'gs://eqtl_catalogue_data/otar2065_microglia/prod/${staging_otar2065_microglia_ts}/output',
        'molqtl/eQTLGen/single_cell_eqtl/{dataset}': 'gs://eqtlgen_data/prod/${staging_eqtlgen_ts}/output/single_cell_eqtl',
        'molqtl/eQTLGen/coeqtl/{dataset}': 'gs://eqtlgen_data/prod/${staging_eqtlgen_ts}/output/coeqtl',
        'molqtl/deCODE_proteomics/{dataset}': 'gs://decode_data/prod/${staging_decode_ts}/output',
        'molqtl/sle_map/{dataset}': 'gs://sle_map_data/prod/${staging_sle_map_ts}/output',
    },
}


def _expected(step: str) -> dict[str, str]:
    """The expected copies of a step, each run read through its source's own staging timestamp."""
    scratchpad = parse_yaml(CONFIG_PATH)['scratchpad']
    expected = {}
    for destination, run_template in EXPECTED_RUNS[step].items():
        run = Template(run_template).substitute(scratchpad)
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
