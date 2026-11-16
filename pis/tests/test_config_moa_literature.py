"""The published release inputs the `moa_literature` step copies."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'
"""The config file that ships in the image, not a fixture."""

RELEASES = 'gs://open-targets-data-releases/${moa_literature_input_release}'
INPUT = 'input/literature/${moa_literature_input_release}'


def test_moa_literature_copies_the_match_and_drug_molecule_of_the_input_release() -> None:
    """Every parquet file of both datasets, billed to the dev project, lands under the input release."""
    tasks = yaml.safe_load(CONFIG_PATH.read_text())['steps']['moa_literature']

    assert tasks == [
        {
            'name': 'copy_many literature match',
            'sources': f'{RELEASES}/intermediate/literature/match/*.parquet',
            'settings': {'billing_project': 'open-targets-eu-dev'},
            'destination': f'{INPUT}/match',
        },
        {
            'name': 'copy_many literature drug molecule',
            'sources': f'{RELEASES}/output/drug_molecule/*.parquet',
            'settings': {'billing_project': 'open-targets-eu-dev'},
            'destination': f'{INPUT}/drug_molecule',
        },
    ]
