"""Tests for the small compute view transformers.

Each case carries over a semantic of the serving script it replaces (`disease_hpo.sql`, `mouse_phenotypes.sql`,
`drug_warnings.sql`, `mechanism_of_action.sql`, `pharmacovigilance.sql`), or the order pts now fixes.
"""

import importlib
from pathlib import Path

import polars as pl
import pytest
from ot_datamodels import DatasetModel
from ot_datamodels.outputs.disease_phenotype import DiseasePhenotype
from ot_datamodels.outputs.drug_mechanism_of_action import DrugMechanismOfAction
from ot_datamodels.outputs.drug_warning import DrugWarning
from ot_datamodels.outputs.mouse_phenotype import MousePhenotype
from ot_datamodels.outputs.pharmacovigilance import Pharmacovigilance
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.views.disease_phenotype_by_disease import DiseasePhenotypeByDisease
from ot_datamodels.views.drug_mechanism_of_action_by_drug import MECHANISM_FIELDS, DrugMechanismOfActionByDrug
from ot_datamodels.views.drug_warning_by_drug import DrugWarningByDrug
from ot_datamodels.views.mouse_phenotype_by_target import MousePhenotypeByTarget
from ot_datamodels.views.pharmacovigilance_by_drug import PharmacovigilanceByDrug

from pts.schemas.dataset.validate import validate_dataset
from pts.task_outputs import OutputValidation
from pts.transformers.disease_phenotype_by_disease import disease_phenotype_by_disease_frame
from pts.transformers.drug_mechanism_of_action_by_drug import drug_mechanism_of_action_by_drug_frame
from pts.transformers.drug_warning_by_drug import drug_warning_by_drug_frame
from pts.transformers.mouse_phenotype_by_target import mouse_phenotype_by_target_frame
from pts.transformers.pharmacovigilance_by_drug import pharmacovigilance_by_drug_frame


def _rows(model: type[DatasetModel], *rows: dict) -> pl.DataFrame:
    """Output rows, the example overridden, in the output schema (Enums stored)."""
    return pl.DataFrame([model.example | row for row in rows], schema=to_polars_schema(model))


def _item(view: pl.DataFrame, column: str) -> dict:
    """The struct schema of the items in `column`."""
    dtype = view.schema[column]
    assert isinstance(dtype, pl.List)
    assert isinstance(dtype.inner, pl.Struct)
    return dtype.inner.to_schema()


def _by_key(view: pl.DataFrame) -> dict:
    key, items = view.columns[:2]
    return dict(zip(view[key], view[items].to_list(), strict=True))


# disease_hpo


def test_phenotypes_collect_per_disease_by_phenotype():
    phenotypes = _rows(
        DiseasePhenotype,
        {'disease': 'MONDO_2', 'phenotype': 'HP_0000002'},
        {'disease': 'MONDO_1', 'phenotype': 'HP_0000009'},
        {'disease': 'MONDO_1', 'phenotype': 'HP_0000001'},
    )

    view = disease_phenotype_by_disease_frame(phenotypes)

    assert view.columns == ['disease', 'phenotypes']
    assert view['disease'].to_list() == ['MONDO_1', 'MONDO_2']
    assert [[p['phenotype'] for p in v] for v in _by_key(view).values()] == [
        ['HP_0000001', 'HP_0000009'],
        ['HP_0000002'],
    ]
    assert list(_by_key(view)['MONDO_2'][0]) == ['disease', 'phenotype', 'evidence']


def test_phenotype_evidence_is_the_output_struct_with_qualifier_and_enums():
    phenotypes = _rows(DiseasePhenotype, {})

    view = disease_phenotype_by_disease_frame(phenotypes)

    assert _item(view, 'phenotypes')['evidence'] == phenotypes.schema['evidence']
    evidence = _item(pl.DataFrame(schema={'e': phenotypes.schema['evidence']}), 'e')
    assert 'qualifier' in evidence
    assert isinstance(evidence['aspect'], pl.Enum)


def test_a_disease_with_empty_evidence_keeps_its_phenotype():
    view = disease_phenotype_by_disease_frame(_rows(DiseasePhenotype, {'evidence': []}))

    assert view['phenotypes'].to_list()[0][0]['evidence'] == []


# mouse_phenotypes


def test_mouse_phenotypes_collect_per_target_in_output_field_order():
    phenotypes = _rows(
        MousePhenotype,
        {'targetFromSourceId': 'ENSG00000000002', 'modelPhenotypeId': 'MP:1'},
        {'targetFromSourceId': 'ENSG00000000001', 'modelPhenotypeId': 'MP:2', 'targetInModelMgiId': 'MGI:1'},
        {'targetFromSourceId': 'ENSG00000000001', 'modelPhenotypeId': 'MP:1', 'targetInModelMgiId': 'MGI:2'},
        {'targetFromSourceId': 'ENSG00000000001', 'modelPhenotypeId': 'MP:1', 'targetInModelMgiId': 'MGI:1'},
    )

    view = mouse_phenotype_by_target_frame(phenotypes)

    assert view.columns == ['targetFromSourceId', 'mouse_phenotypes']
    by_target = _by_key(view)
    assert list(by_target) == ['ENSG00000000001', 'ENSG00000000002']
    assert [(p['modelPhenotypeId'], p['targetInModelMgiId']) for p in by_target['ENSG00000000001']] == [
        ('MP:1', 'MGI:1'),
        ('MP:1', 'MGI:2'),
        ('MP:2', 'MGI:1'),
    ]
    assert list(by_target['ENSG00000000002'][0]) == list(MousePhenotype.model_fields)


def test_null_literature_stays_null():
    model = MousePhenotype.example['biologicalModels'][0] | {'literature': None}
    view = mouse_phenotype_by_target_frame(_rows(MousePhenotype, {'biologicalModels': [model]}))

    assert view['mouse_phenotypes'].to_list()[0][0]['biologicalModels'][0]['literature'] is None


# drug_warnings


def test_warnings_fan_out_to_each_molecule_by_id_keeping_chembl_ids():
    warnings = _rows(
        DrugWarning,
        {'id': 3, 'chemblIds': ['CHEMBL2', 'CHEMBL1']},
        {'id': 1, 'chemblIds': ['CHEMBL1']},
        {'id': 2, 'chemblIds': []},
    )

    view = drug_warning_by_drug_frame(warnings)

    assert view.columns == ['chemblId', 'drugWarnings']
    by_drug = _by_key(view)
    assert list(by_drug) == ['CHEMBL1', 'CHEMBL2']
    assert [w['id'] for w in by_drug['CHEMBL1']] == [1, 3]
    assert by_drug['CHEMBL2'][0]['chemblIds'] == ['CHEMBL2', 'CHEMBL1']
    assert list(by_drug['CHEMBL2'][0]) == list(DrugWarning.model_fields)
    assert _item(view, 'drugWarnings')['warningType'] == warnings.schema['warningType']


def test_a_molecule_named_twice_repeats_the_warning():
    view = drug_warning_by_drug_frame(_rows(DrugWarning, {'chemblIds': ['CHEMBL1', 'CHEMBL1']}))

    assert [len(w) for w in view['drugWarnings'].to_list()] == [2]


# mechanism_of_action


def test_mechanisms_fan_out_sorted_with_types_in_vocabulary_order():
    mechanisms = _rows(
        DrugMechanismOfAction,
        {'chemblIds': ['CHEMBL1'], 'mechanismOfAction': 'B', 'actionType': 'INHIBITOR', 'targetType': 'unknown'},
        {'chemblIds': ['CHEMBL1', 'CHEMBL2'], 'mechanismOfAction': 'A', 'actionType': 'OTHER'},
        {'chemblIds': ['CHEMBL1'], 'mechanismOfAction': 'A', 'actionType': 'AGONIST', 'targetType': 'admet'},
        {'chemblIds': [], 'mechanismOfAction': 'C'},
    )

    view = drug_mechanism_of_action_by_drug_frame(mechanisms)

    assert view.columns == ['chemblId', 'rows', 'uniqueTargetTypes', 'uniqueActionTypes']
    one = view.row(0, named=True)
    assert one['chemblId'] == 'CHEMBL1'
    assert [(r['mechanismOfAction'], r['actionType']) for r in one['rows']] == [
        ('A', 'AGONIST'),
        ('A', 'OTHER'),
        ('B', 'INHIBITOR'),
    ]
    assert one['uniqueTargetTypes'] == ['admet', 'single protein', 'unknown']
    assert one['uniqueActionTypes'] == ['AGONIST', 'INHIBITOR', 'OTHER']
    assert view['chemblId'].to_list() == ['CHEMBL1', 'CHEMBL2']
    assert list(one['rows'][0]) == MECHANISM_FIELDS
    assert list(one['rows'][0]['references'][0]) == ['source', 'ids', 'urls']


def test_mechanism_ties_keep_output_order_and_enums_stay_enums():
    mechanisms = _rows(
        DrugMechanismOfAction,
        {'targets': ['ENSG00000000002']},
        {'targets': ['ENSG00000000001']},
    )

    view = drug_mechanism_of_action_by_drug_frame(mechanisms)

    assert [r['targets'] for r in view['rows'].to_list()[0]] == [['ENSG00000000002'], ['ENSG00000000001']]
    assert view.schema['uniqueTargetTypes'] == pl.List(mechanisms.schema['targetType'])
    assert view.schema['uniqueActionTypes'] == pl.List(mechanisms.schema['actionType'])
    rows = _item(view, 'rows')
    assert rows['actionType'] == mechanisms.schema['actionType']
    assert rows['curation'] == mechanisms.schema['curation']


# pharmacovigilance


def test_adverse_events_by_llr_descending_ties_by_name():
    events = _rows(
        Pharmacovigilance,
        {'chembl_id': 'CHEMBL2', 'event': 'a', 'llr': 1.0},
        {'chembl_id': 'CHEMBL1', 'event': 'c', 'llr': 2.0, 'critval': 5.0},
        {'chembl_id': 'CHEMBL1', 'event': 'b', 'llr': 2.0, 'critval': 5.0},
        {'chembl_id': 'CHEMBL1', 'event': 'a', 'llr': 3.0, 'critval': 5.0},
        {'chembl_id': 'CHEMBL1', 'event': 'd', 'llr': 0.5, 'critval': 5.0},
    )

    view = pharmacovigilance_by_drug_frame(events)

    assert view.columns == ['chembl_id', 'adverse_events', 'criticalValue']
    by_drug = _by_key(view)
    assert list(by_drug) == ['CHEMBL1', 'CHEMBL2']
    assert [(e['name'], e['logLR']) for e in by_drug['CHEMBL1']] == [('a', 3.0), ('b', 2.0), ('c', 2.0), ('d', 0.5)]
    assert list(by_drug['CHEMBL1'][0]) == ['name', 'count', 'logLR', 'critval', 'meddraCode']
    assert view['criticalValue'].to_list() == [5.0, Pharmacovigilance.example['critval']]


def test_critical_value_is_the_first_event_s():
    events = _rows(
        Pharmacovigilance,
        {'event': 'a', 'llr': 1.0, 'critval': 1.0},
        {'event': 'b', 'llr': 9.0, 'critval': 9.0},
    )

    assert pharmacovigilance_by_drug_frame(events)['criticalValue'].to_list() == [9.0]


# every view

VIEWS = [
    ('disease_phenotype_by_disease', DiseasePhenotype, DiseasePhenotypeByDisease, 'disease'),
    ('mouse_phenotype_by_target', MousePhenotype, MousePhenotypeByTarget, 'targetFromSourceId'),
    ('drug_warning_by_drug', DrugWarning, DrugWarningByDrug, 'chemblIds'),
    ('drug_mechanism_of_action_by_drug', DrugMechanismOfAction, DrugMechanismOfActionByDrug, 'chemblIds'),
    ('pharmacovigilance_by_drug', Pharmacovigilance, PharmacovigilanceByDrug, 'chembl_id'),
]


def _key(column: str, i: int) -> object:
    if column == 'chemblIds':
        return [f'CHEMBL{i // 3}']
    if column == 'targetFromSourceId':
        return f'ENSG{i // 3:011d}'
    if column == 'disease':
        return f'MONDO_{i // 3}'
    return f'CHEMBL{i // 3}'


def _distinct(output: type[DatasetModel], i: int) -> dict:
    """Overrides making row `i` unique on the output's key."""
    return {
        'disease_phenotype': {'phenotype': f'HP_{i:07d}'},
        'mouse_phenotype': {'modelPhenotypeId': f'MP:{i}'},
        'drug_warning': {'id': i},
        'drug_mechanism_of_action': {},
        'pharmacovigilance': {'event': f'e{i}'},
    }[output.dataset_name]


@pytest.mark.parametrize(('name', 'output', 'view', 'key'), VIEWS, ids=[v[0] for v in VIEWS])
def test_many_chunks_build(name, output, view, key):
    parts = [
        _rows(output, *({key: _key(key, i)} | _distinct(output, i) for i in range(start, start + 600)))
        for start in range(0, 3000, 600)
    ]
    frame = pl.concat(parts, rechunk=False)
    assert frame.n_chunks() > 1
    function = getattr(importlib.import_module(f'pts.transformers.{name}'), f'{name}_frame')

    result = function(frame)

    assert result.height == 1000
    assert result[result.columns[1]].list.len().sum() == 3000


@pytest.mark.parametrize(('name', 'output', 'view', 'key'), VIEWS, ids=[v[0] for v in VIEWS])
def test_writes_its_model(tmp_path: Path, name, output, view, key):
    module = importlib.import_module(f'pts.transformers.{name}')
    transformer = getattr(module, name)
    (tmp_path / 'output').mkdir()
    rows = _rows(output, *({key: _key(key, i)} | _distinct(output, i) for i in range(6)))
    rows.write_parquet(tmp_path / 'output' / 'part-0.parquet')
    destination = {'view': str(tmp_path / 'view')}

    transformer(str(tmp_path / 'output'), destination, {}, None)

    assert OutputValidation.of(transformer, module).outputs == {'view': view}
    validate_dataset(view, destination['view'])
    written = pl.read_parquet(destination['view'])
    assert written.schema == to_polars_schema(view)
    assert written.height == 2
