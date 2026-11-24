"""Tests for the drug view transformer."""

from pathlib import Path

import polars as pl
from ot_datamodels.outputs.drug_molecule import DrugMolecule
from ot_datamodels.outputs.pharmacogenomics import Pharmacogenomics
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.views.drug_view import DrugView

from pts.schemas.dataset.validate import validate_dataset
from pts.task_outputs import OutputValidation
from pts.transformers import drug_view as module
from pts.transformers.drug_view import drug_view, drug_view_frame

DRUG_MOLECULE = to_polars_schema(DrugMolecule)
PHARMACOGENOMICS = to_polars_schema(Pharmacogenomics)


def _drugs(*ids: str) -> pl.DataFrame:
    return pl.DataFrame([DrugMolecule.example | {'id': drug_id} for drug_id in ids], schema=DRUG_MOLECULE)


def _pgx(rows: list[tuple[int, list[str | None] | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            Pharmacogenomics.example
            | {
                'pharmacogenomicsId': row_id,
                'drugs': None if drug_ids is None else [{'drugFromSource': None, 'drugId': d} for d in drug_ids],
            }
            for row_id, drug_ids in rows
        ],
        schema=PHARMACOGENOMICS,
    )


def _ids(view: pl.DataFrame) -> dict[str, list[int]]:
    return dict(zip(view['id'], view['pharmacogenomicsIds'].to_list(), strict=True))


def test_drug_without_pharmacogenomics_gets_an_empty_list():
    view = drug_view_frame(_drugs('A', 'B'), _pgx([(1, ['A'])]))

    assert _ids(view) == {'A': [1], 'B': []}


def test_null_drug_ids_are_ignored():
    view = drug_view_frame(_drugs('A'), _pgx([(1, [None, 'A']), (2, [None]), (3, []), (4, None)]))

    assert _ids(view) == {'A': [1]}


def test_a_row_naming_a_drug_twice_contributes_its_id_twice():
    view = drug_view_frame(_drugs('A'), _pgx([(5, ['A', 'A']), (2, ['A'])]))

    assert _ids(view) == {'A': [2, 5, 5]}


def test_drugs_absent_from_drug_molecule_are_dropped():
    view = drug_view_frame(_drugs('A'), _pgx([(1, ['A', 'Z']), (2, ['Z'])]))

    assert view['id'].to_list() == ['A']
    assert _ids(view) == {'A': [1]}


def test_ids_sorted_and_drug_rows_and_columns_kept():
    drugs = _drugs('C', 'A', 'B')

    view = drug_view_frame(drugs, _pgx([(9, ['A']), (3, ['B', 'A']), (7, ['A']), (1, ['C'])]))

    assert view.columns == [*drugs.columns, 'pharmacogenomicsIds']
    assert view.drop('pharmacogenomicsIds').equals(drugs)
    assert view['pharmacogenomicsIds'].to_list() == [[1], [3, 7, 9], [3]]


def test_writes_its_model(tmp_path: Path):
    source = {}
    for name, frame in {
        'drug_molecule': _drugs(Pharmacogenomics.example['drugs'][0]['drugId'], 'CHEMBL100'),
        'pharmacogenomics': pl.DataFrame([Pharmacogenomics.example], schema=PHARMACOGENOMICS),
    }.items():
        (tmp_path / name).mkdir()
        frame.write_parquet(tmp_path / name / 'part-0.parquet')
        source[name] = str(tmp_path / name)
    destination = {'view': str(tmp_path / 'view')}

    drug_view(source, destination, {}, None)

    assert OutputValidation.of(drug_view, module).outputs == {'view': DrugView}
    validate_dataset(DrugView, destination['view'])
    written = pl.read_parquet(destination['view'])
    assert written.schema == to_polars_schema(DrugView)
    assert written['pharmacogenomicsIds'].to_list() == [[Pharmacogenomics.example['pharmacogenomicsId']], []]
