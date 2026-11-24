"""Tests for the disease view transformer.

Each case carries over a semantic of the `disease.sql` serving script it replaces.
"""

from pathlib import Path

import polars as pl
from ot_datamodels.outputs.disease import Disease
from ot_datamodels.outputs.study import Study
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.views.disease_view import DiseaseView

from pts.schemas.dataset.validate import validate_dataset
from pts.task_outputs import OutputValidation
from pts.transformers import disease_view as module
from pts.transformers.disease_view import disease_view, disease_view_frame

DISEASE = to_polars_schema(Disease)
STUDY = pl.Schema({'studyId': pl.String, 'diseaseIds': pl.List(pl.String)})
RELATIONS = ['hasExactSynonym', 'hasRelatedSynonym', 'hasNarrowSynonym', 'hasBroadSynonym']


def _diseases(*rows: str | tuple[str, list[str]] | dict) -> pl.DataFrame:
    """Rows as an id, an `(id, descendants)` pair, or a dict of overrides."""
    records = []
    for row in rows:
        if isinstance(row, str):
            row = {'id': row, 'descendants': []}
        elif isinstance(row, tuple):
            row = {'id': row[0], 'descendants': row[1]}
        records.append(Disease.example | row)
    return pl.DataFrame(records, schema=DISEASE)


def _studies(*rows: tuple[str, list[str] | None]) -> pl.DataFrame:
    return pl.DataFrame([{'studyId': s, 'diseaseIds': d} for s, d in rows], schema=STUDY)


def _col(view: pl.DataFrame, column: str) -> dict[str, list]:
    return dict(zip(view['id'], view[column].to_list(), strict=True))


def test_disease_without_studies_gets_empty_lists():
    view = disease_view_frame(_diseases('A', ('B', ['A'])), _studies(('S1', ['Z'])))

    assert _col(view, 'studyIds') == {'A': [], 'B': []}
    assert _col(view, 'indirectStudyIds') == {'A': [], 'B': []}


def test_disease_without_descendants_gets_no_indirect_studies():
    view = disease_view_frame(_diseases('A'), _studies(('S1', ['A'])))

    assert _col(view, 'studyIds') == {'A': ['S1']}
    assert _col(view, 'indirectStudyIds') == {'A': []}


def test_study_listing_a_disease_twice_appears_once():
    view = disease_view_frame(_diseases(('P', ['A']), 'A'), _studies(('S1', ['A', 'A']), ('S1', ['A'])))

    assert _col(view, 'studyIds') == {'P': [], 'A': ['S1']}
    assert _col(view, 'indirectStudyIds') == {'P': ['S1'], 'A': []}


def test_empty_study_ids_and_null_disease_ids_are_ignored():
    view = disease_view_frame(
        _diseases(('P', ['A']), 'A'),
        _studies(('', ['A']), ('S1', ['A', None]), ('S2', None), ('S3', [])),
    )

    assert _col(view, 'studyIds') == {'P': [], 'A': ['S1']}
    assert _col(view, 'indirectStudyIds') == {'P': ['S1'], 'A': []}


def test_indirect_studies_come_from_descendants_only_distinct_and_sorted():
    view = disease_view_frame(
        _diseases(('P', ['A', 'B']), 'A', 'B'),
        _studies(('S9', ['A']), ('S1', ['B', 'A']), ('S5', ['B']), ('S0', ['P']), ('S2', ['P', 'A'])),
    )

    assert _col(view, 'studyIds') == {'P': ['S0', 'S2'], 'A': ['S1', 'S2', 'S9'], 'B': ['S1', 'S5']}
    assert _col(view, 'indirectStudyIds') == {'P': ['S1', 'S2', 'S5', 'S9'], 'A': [], 'B': []}


def test_study_disease_ids_absent_from_disease_are_dropped():
    view = disease_view_frame(_diseases('A'), _studies(('S1', ['A', 'Z']), ('S2', ['Z'])))

    assert view['id'].to_list() == ['A']
    assert _col(view, 'studyIds') == {'A': ['S1']}


def test_descendant_studies_count_even_when_the_descendant_is_not_a_disease_row():
    # disease.sql joins descendants to every study disease id, not only to disease rows
    view = disease_view_frame(_diseases(('P', ['Z'])), _studies(('S1', ['Z'])))

    assert _col(view, 'indirectStudyIds') == {'P': ['S1']}


def test_synonyms_are_four_relation_terms_pairs_in_model_order():
    synonyms = {
        'hasExactSynonym': ['e1', 'e2'],
        'hasRelatedSynonym': [],
        'hasNarrowSynonym': ['n'],
        'hasBroadSynonym': [],
    }
    view = disease_view_frame(_diseases({'id': 'A', 'synonyms': synonyms}), _studies())

    assert view['synonyms'].to_list() == [
        [
            {'relation': 'hasExactSynonym', 'terms': ['e1', 'e2']},
            {'relation': 'hasRelatedSynonym', 'terms': []},
            {'relation': 'hasNarrowSynonym', 'terms': ['n']},
            {'relation': 'hasBroadSynonym', 'terms': []},
        ]
    ]


def test_null_synonym_groups_become_empty_terms():
    view = disease_view_frame(
        _diseases(
            {
                'id': 'A',
                'synonyms': {
                    'hasExactSynonym': None,
                    'hasRelatedSynonym': ['r'],
                    'hasNarrowSynonym': None,
                    'hasBroadSynonym': None,
                },
            },
            {'id': 'B', 'synonyms': None},
        ),
        _studies(),
    )

    assert view['synonyms'].to_list() == [
        [{'relation': r, 'terms': ['r'] if r == 'hasRelatedSynonym' else []} for r in RELATIONS],
        [{'relation': r, 'terms': []} for r in RELATIONS],
    ]


def test_is_therapeutic_area_comes_from_ontology():
    ontology = Disease.example['ontology']
    view = disease_view_frame(
        _diseases(
            {'id': 'A', 'ontology': ontology | {'isTherapeuticArea': True}},
            {'id': 'B', 'ontology': ontology | {'isTherapeuticArea': False}},
        ),
        _studies(),
    )

    assert _col(view, 'isTherapeuticArea') == {'A': True, 'B': False}


def test_rows_in_disease_order_and_columns_are_the_model():
    diseases = _diseases('C', ('A', ['C']), 'B')

    view = disease_view_frame(diseases, _studies(('S1', ['C', 'B'])))

    assert view.columns == list(DiseaseView.model_fields)
    assert view['id'].to_list() == ['C', 'A', 'B']
    kept = [c for c in view.columns if c in diseases.columns and c != 'synonyms']
    assert len(kept) == 10
    assert view.select(kept).equals(diseases.select(kept))


def test_many_diseases_joined_to_studies_build():
    n = 3_000
    ids = [f'D{i:05d}' for i in range(n)]
    rows = [(d, [ids[(i + 1) % n]]) for i, d in enumerate(ids)]
    # several parts, as collected from a multi-part dataset
    diseases = pl.concat([_diseases(*rows[i : i + 700]) for i in range(0, n, 700)], rechunk=False)
    studies = _studies(*[(f'S{i:05d}', [ids[i], ids[(i * 7) % n]]) for i in range(0, n, 2)])
    assert diseases.n_chunks() > 1

    view = disease_view_frame(diseases, studies)

    assert view['id'].to_list() == ids
    assert view.height == n
    assert len(view['synonyms'].list.len().unique()) == 1
    assert _col(view, 'studyIds')['D00000'] == ['S00000']
    assert _col(view, 'indirectStudyIds')['D02999'] == ['S00000']


def test_writes_its_model(tmp_path: Path):
    disease_id = Study.example['diseaseIds'][0]
    source = {}
    for name, frame in {
        'disease': _diseases(('EFO_0000001', [disease_id]), disease_id),
        'study': pl.DataFrame([Study.example], schema=to_polars_schema(Study)),
    }.items():
        (tmp_path / name).mkdir()
        frame.write_parquet(tmp_path / name / 'part-0.parquet')
        source[name] = str(tmp_path / name)
    destination = {'view': str(tmp_path / 'view')}

    disease_view(source, destination, {}, None)

    assert OutputValidation.of(disease_view, module).outputs == {'view': DiseaseView}
    validate_dataset(DiseaseView, destination['view'])
    written = pl.read_parquet(destination['view'])
    assert written.schema == to_polars_schema(DiseaseView)
    assert written['studyIds'].to_list() == [[], [Study.example['studyId']]]
    assert written['indirectStudyIds'].to_list() == [[Study.example['studyId']], []]
