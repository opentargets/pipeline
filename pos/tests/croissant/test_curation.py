"""Tests for the Platform instance curation and the dataset descriptions croissant publishes."""

import json
import re
from importlib import resources
from pathlib import Path

from ot_datamodels.discovery import discover_datasets
from pydantic import ValidationError

from pos import croissant
from pos.croissant.models import InstanceAnnotation

ASSETS_DIR = Path(croissant.__file__).resolve().parent / 'assets'


def _repo_root() -> Path:
    return next(p for p in Path(__file__).resolve().parents if (p / 'pis/config.yaml').is_file())


def test_chembl_bioactivity_names_the_chembl_release_the_pipeline_downloads():
    """The release is in the description rather than in every row, so a ChEMBL bump must update it."""
    pis_config = (_repo_root() / 'pis/config.yaml').read_text()
    match = re.search(r"^\s*chembl_version:\s*'?(\d+)'?\s*$", pis_config, re.MULTILINE)
    assert match, 'chembl_version not found in pis/config.yaml'
    assert f'ChEMBL {match.group(1)} ' in discover_datasets()['chembl_bioactivity'].dataset_description


def test_instance_file_conforms_to_model():
    """Every entry in instance.json must validate as InstanceAnnotation."""
    errors = []
    for entry in json.loads((ASSETS_DIR / 'instance.json').read_text()):
        try:
            InstanceAnnotation.model_validate(entry)
        except ValidationError as exc:
            errors.append(f'{entry.get("id", "?")}: {exc}')
    assert not errors, 'Instance entries failed model validation:\n' + '\n'.join(errors)


def test_assets_ship_with_the_package():
    assert (resources.files('pos.croissant') / 'assets/instance.json').is_file()
