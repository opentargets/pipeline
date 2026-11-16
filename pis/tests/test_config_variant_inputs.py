"""The `variant` step mirrors the fastVEP reference inputs from their original sources."""

from pathlib import Path
from typing import Any

from otter.config.yaml import parse_yaml

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'
"""The config file that ships in the image, not a fixture."""

ENSEMBL = 'https://ftp.ensembl.org/pub/release-${ensembl_version}'
VARIATION_DB = f'{ENSEMBL}/mysql/homo_sapiens_variation_${{ensembl_version}}_38'
LOFTEE = 'https://personal.broadinstitute.org/konradk/loftee_data/GRCh38/human_ancestor.fa.gz'

EXPECTED_SOURCES = {
    'input/variant/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz': (
        f'{ENSEMBL}/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz'
    ),
    'input/variant/reference/Homo_sapiens.GRCh38.116.gff3.gz': (
        f'{ENSEMBL}/gff3/homo_sapiens/Homo_sapiens.GRCh38.${{ensembl_version}}.gff3.gz'
    ),
    'input/variant/gerp/gerp_conservation_scores.homo_sapiens.GRCh38.bw': (
        f'{ENSEMBL}/compara/conservation_scores/92_mammals.gerp_conservation_score/'
        'gerp_conservation_scores.homo_sapiens.GRCh38.bw'
    ),
    'input/variant/vep_cache/homo_sapiens_vep_116_GRCh38.tar.gz': (
        f'{ENSEMBL}/variation/indexed_vep_cache/homo_sapiens_vep_${{ensembl_version}}_GRCh38.tar.gz'
    ),
    'input/variant/variation/protein_function_predictions.txt.gz': (
        f'{VARIATION_DB}/protein_function_predictions.txt.gz'
    ),
    'input/variant/variation/translation_md5.txt.gz': f'{VARIATION_DB}/translation_md5.txt.gz',
    'input/variant/variation/attrib.txt.gz': f'{VARIATION_DB}/attrib.txt.gz',
    'input/variant/reference/Homo_sapiens.GRCh38.pep.all.fa.gz': (
        f'{ENSEMBL}/fasta/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz'
    ),
    'input/variant/alphamissense/AlphaMissense_hg38.tsv.gz': 'gs://dm_alphamissense/AlphaMissense_hg38.tsv.gz',
    'input/variant/loftee/human_ancestor.fa.gz': LOFTEE,
}


def _copies(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [task for task in tasks if task['name'].startswith('copy ')]


def test_variant_inputs_are_mirrored_from_their_sources() -> None:
    copies = _copies(parse_yaml(CONFIG_PATH)['steps']['variant'])
    mirrored = {task['destination']: task['source'] for task in copies}
    assert {d: s for d, s in mirrored.items() if d in EXPECTED_SOURCES} == EXPECTED_SOURCES
    assert len([task for task in copies if task['destination'] in EXPECTED_SOURCES]) == len(EXPECTED_SOURCES)


def test_the_ancestor_is_mirrored_without_its_indexes() -> None:
    copies = _copies(parse_yaml(CONFIG_PATH)['steps']['variant'])
    assert [task['destination'] for task in copies if task['destination'].startswith('input/variant/loftee/')] == [
        'input/variant/loftee/human_ancestor.fa.gz'
    ]
