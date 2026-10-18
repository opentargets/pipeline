"""Tests for the polars OnToma lookup-table steps."""

from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from pts.schemas.ontoma_lut import ENTITY_LUT
from pts.transformers.ontoma_lut.literature_ontoma import literature_ontoma
from pts.transformers.ontoma_lut.ontoma import ontoma
from tests.ontoma_lut_fixtures import DISEASE, DISEASE_SCHEMA, DRUG, DRUG_SCHEMA, TARGET, TARGET_SCHEMA
from tests.ontoma_lut_spark_rows import STEP_DISEASE_ID, STEP_DISEASE_LABEL, STEP_LITERATURE

CURATION_SOURCES = {
    'ot_disease_curation': 'ot_curation',
    'eva_clinvar': 'eva_clinvar',
    'clinvar_xrefs': 'clinvar_xrefs',
}
CURATION_SCHEMA = {'SEMANTIC_TAG': pl.String, 'PROPERTY_VALUE': pl.String}
CURATION_LABELS = {
    'ot_disease_curation': 'Zorblax from ot',
    'eva_clinvar': 'Quillon from eva',
    'clinvar_xrefs': 'Wexlar from xrefs',
}


def _write(frame: pl.DataFrame, path: Path) -> str:
    path.mkdir(parents=True)
    frame.write_parquet(path / 'part.parquet')
    return str(path)


def _inputs(
    root: Path, disease: pl.DataFrame, target: pl.DataFrame, drug: pl.DataFrame, labels: dict[str, str] | None
) -> dict[str, str]:
    source = {
        'disease_index': _write(disease, root / 'disease'),
        'target_index': _write(target, root / 'target'),
        'drug_index': _write(drug, root / 'drug'),
    }
    for key in CURATION_SOURCES:
        curation = pl.DataFrame(
            {
                'SEMANTIC_TAG': ['http://x/MONDO_0000001'] * (labels is not None),
                'PROPERTY_VALUE': [labels[key]] if labels is not None else [],
            },
            schema=CURATION_SCHEMA,
        )
        curation.write_csv(root / f'{key}.tsv', separator='\t')
        source[key] = str(root / f'{key}.tsv')
    return source


def _read(path: str) -> pl.DataFrame:
    return pl.read_parquet(Path(path) / '*.parquet')


@pytest.fixture
def source(tmp_path: Path) -> dict[str, str]:
    return _inputs(tmp_path / 'in', DISEASE, TARGET, DRUG, CURATION_LABELS)


def test_ontoma_step_writes_both_luts(source: dict[str, str], tmp_path: Path) -> None:
    destination = {
        'disease_label_lut': str(tmp_path / 'out' / 'label.parquet'),
        'disease_id_lut': str(tmp_path / 'out' / 'id.parquet'),
    }
    ontoma(source, destination, None, None)

    label = _read(destination['disease_label_lut'])
    ids = _read(destination['disease_id_lut'])
    assert label.schema == ENTITY_LUT
    assert ids.schema == ENTITY_LUT
    assert label.rows() == STEP_DISEASE_LABEL
    assert ids.rows() == STEP_DISEASE_ID
    assert label.height > 0
    assert ids.height > 0


def test_each_curation_keeps_its_source(source: dict[str, str], tmp_path: Path) -> None:
    destination = {
        'disease_label_lut': str(tmp_path / 'out' / 'label.parquet'),
        'disease_id_lut': str(tmp_path / 'out' / 'id.parquet'),
    }
    ontoma(source, destination, None, None)

    label = _read(destination['disease_label_lut'])
    for key, name in CURATION_SOURCES.items():
        word = CURATION_LABELS[key].split()[0].lower()
        sources = label.filter(pl.col('entityLabelNormalised').str.contains(word)).get_column('entityIds').to_list()
        assert [{item['entitySource'] for item in ids} for ids in sources] == [{name}], key


def test_literature_ontoma_step_writes_one_lut(source: dict[str, str], tmp_path: Path) -> None:
    destination = {'disease_target_drug_label_lut': str(tmp_path / 'out' / 'lit.parquet')}
    literature_ontoma(source, destination, None, None)

    lut = _read(destination['disease_target_drug_label_lut'])
    assert lut.schema == ENTITY_LUT
    assert lut.rows() == STEP_LITERATURE


@pytest.mark.parametrize(
    ('run', 'keys'),
    [(ontoma, ['disease_label_lut', 'disease_id_lut']), (literature_ontoma, ['disease_target_drug_label_lut'])],
)
def test_steps_refuse_existing_destination(
    run: Callable[..., None], keys: list[str], source: dict[str, str], tmp_path: Path
) -> None:
    destination = {key: str(tmp_path / 'out' / f'{key}.parquet') for key in keys}
    run(source, destination, None, None)
    first = {key: _read(path).rows() for key, path in destination.items()}

    with pytest.raises(ValueError, match='already exists'):
        run(source, destination, None, None)
    assert {key: _read(path).rows() for key, path in destination.items()} == first


def test_empty_inputs_write_empty_luts(tmp_path: Path) -> None:
    source = _inputs(
        tmp_path / 'in',
        pl.DataFrame(schema=DISEASE_SCHEMA),
        pl.DataFrame(schema=TARGET_SCHEMA),
        pl.DataFrame(schema=DRUG_SCHEMA),
        None,
    )
    destination = {
        'disease_label_lut': str(tmp_path / 'out' / 'label.parquet'),
        'disease_id_lut': str(tmp_path / 'out' / 'id.parquet'),
    }
    literature = {'disease_target_drug_label_lut': str(tmp_path / 'out' / 'lit.parquet')}
    ontoma(source, destination, None, None)
    literature_ontoma(source, literature, None, None)

    for path in (*destination.values(), *literature.values()):
        lut = _read(path)
        assert lut.schema == ENTITY_LUT
        assert lut.height == 0
