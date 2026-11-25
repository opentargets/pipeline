"""Tests for the colocalisation_view and l2g_prediction_view transformers and the bucket helper.

Each case carries over a semantic of the serving script it replaces (`colocalisation.sql`, `l2g_predictions.sql`),
or the order pts now fixes.
"""

import importlib
from pathlib import Path

import polars as pl
import pytest
from ot_datamodels import DatasetModel
from ot_datamodels.outputs.colocalisation import Colocalisation
from ot_datamodels.outputs.l2g_prediction import L2gPrediction
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.views.colocalisation_by_credible_set import ColocalisationByCredibleSet
from ot_datamodels.views.l2g_prediction_by_credible_set import L2gPredictionByCredibleSet
from otter.config.model import Config

from pts.schemas.dataset.validate import validate_dataset
from pts.task_outputs import OutputValidation
from pts.transformers.colocalisation_view import FIELDS, colocalisation_view, colocalisation_view_frame
from pts.transformers.l2g_prediction_view import l2g_prediction_view_frame
from pts.transformers.utils.buckets import for_each_bucket, read_bucket, write_buckets


def _rows(model: type[DatasetModel], *rows: dict) -> pl.DataFrame:
    """Output rows, the example overridden, in the output schema (Enums stored)."""
    return pl.DataFrame([model.example | row for row in rows], schema=to_polars_schema(model))


def _id(i: int) -> str:
    return f'{i:032x}'


def _by_key(view: pl.DataFrame) -> dict:
    key, items = view.columns[:2]
    return dict(zip(view[key], view[items].to_list(), strict=True))


# colocalisation


def test_every_row_is_served_from_both_sides():
    rows = _rows(Colocalisation, {'leftLocusId': _id(1), 'rightLocusId': _id(2), 'rightStudyType': 'eqtl'})

    view = colocalisation_view_frame(rows)

    assert view.columns == ['credibleSetId', 'colocalisation']
    by_set = _by_key(view)
    assert list(by_set) == [_id(1), _id(2)]
    (left,), (right,) = by_set[_id(1)], by_set[_id(2)]
    assert (left['credibleSetId'], left['otherCredibleSetId'], left['rightStudyType']) == (_id(1), _id(2), 'eqtl')
    assert (right['credibleSetId'], right['otherCredibleSetId'], right['rightStudyType']) == (_id(2), _id(1), 'gwas')
    assert list(left) == FIELDS
    shared = ['chromosome', 'colocalisationMethod', 'numberColocalisingVariants', 'h3', 'h4', 'clpp']
    expected = {k: Colocalisation.example[k] for k in shared}
    assert {k: left[k] for k in shared} == {k: right[k] for k in shared} == expected


def test_a_gwas_pair_listed_both_ways_is_served_once_per_set():
    """`groupArrayDistinct`: the swapped side of (A, B, gwas) is (B, A, gwas), the as-is side of the reverse row."""
    rows = _rows(
        Colocalisation,
        {'leftLocusId': _id(1), 'rightLocusId': _id(2), 'rightStudyType': 'gwas', 'betaRatioSignAverage': None},
        {'leftLocusId': _id(2), 'rightLocusId': _id(1), 'rightStudyType': 'gwas', 'betaRatioSignAverage': None},
        {'leftLocusId': _id(1), 'rightLocusId': _id(2), 'rightStudyType': 'gwas', 'betaRatioSignAverage': None},
    )

    by_set = _by_key(colocalisation_view_frame(rows))

    assert [len(by_set[_id(1)]), len(by_set[_id(2)])] == [1, 1]
    assert by_set[_id(1)][0]['betaRatioSignAverage'] is None


def test_tuples_differing_in_any_field_are_kept():
    rows = _rows(
        Colocalisation,
        {'leftLocusId': _id(1), 'rightLocusId': _id(2), 'rightStudyType': 'gwas', 'h3': 0.1},
        {'leftLocusId': _id(2), 'rightLocusId': _id(1), 'rightStudyType': 'gwas', 'h3': 0.2},
    )

    by_set = _by_key(colocalisation_view_frame(rows))

    assert [t['h3'] for t in by_set[_id(1)]] == [0.1, 0.2]


def test_tuples_are_sorted_by_every_field():
    rows = _rows(
        Colocalisation,
        {'leftLocusId': _id(1), 'rightLocusId': _id(9), 'rightStudyType': 'eqtl'},
        {'leftLocusId': _id(1), 'rightLocusId': _id(3), 'rightStudyType': 'pqtl', 'h4': 0.9},
        {'leftLocusId': _id(1), 'rightLocusId': _id(3), 'rightStudyType': 'pqtl', 'h4': 0.5},
        {'leftLocusId': _id(1), 'rightLocusId': _id(3), 'rightStudyType': 'tuqtl'},
        {
            'leftLocusId': _id(1),
            'rightLocusId': _id(3),
            'rightStudyType': 'pqtl',
            'h4': 0.5,
            'betaRatioSignAverage': None,
        },
    )

    (tuples,) = _by_key(colocalisation_view_frame(rows).filter(pl.col('credibleSetId') == _id(1))).values()

    # rightStudyType in the vocabulary's order, as the Enum8; a null last
    assert [(t['otherCredibleSetId'][-1], t['rightStudyType'], t['h4'], t['betaRatioSignAverage']) for t in tuples] == [
        ('3', 'tuqtl', Colocalisation.example['h4'], 1.0),
        ('3', 'pqtl', 0.5, 1.0),
        ('3', 'pqtl', 0.5, None),
        ('3', 'pqtl', 0.9, 1.0),
        ('9', 'eqtl', Colocalisation.example['h4'], 1.0),
    ]


def _coloc_rows(count: int) -> pl.DataFrame:
    """`count` rows over 50 sets, a third of them repeating a GWAS pair the other way round."""
    rows = []
    for i in range(count):
        left, right = i % 50, (i * 7 + 1) % 50
        if left == right:
            right = (right + 1) % 50
        if i % 3 == 0:
            left, right = right, left
        rows.append({
            'leftLocusId': _id(left),
            'rightLocusId': _id(right),
            'rightStudyType': 'gwas' if i % 2 else 'eqtl',
            'numberColocalisingVariants': 1 + i % 4,
        })
    return _rows(Colocalisation, *rows)


def _sorted(view: pl.DataFrame) -> pl.DataFrame:
    return view.sort('credibleSetId')


def _write_parts(rows: pl.DataFrame, folder: Path, parts: int) -> None:
    folder.mkdir()
    size = -(-rows.height // parts)
    for n, start in enumerate(range(0, rows.height, size)):
        rows.slice(start, size).write_parquet(folder / f'part-{n:05d}.parquet')


@pytest.mark.parametrize('bucket_count', [1, 7])
def test_bucketed_step_writes_the_unbucketed_view(tmp_path: Path, bucket_count: int):
    rows = _coloc_rows(3000)
    _write_parts(rows, tmp_path / 'output', 4)
    destination = {'view': str(tmp_path / 'view')}
    config = Config(step='colocalisation_view', steps=['colocalisation_view'], work_path=tmp_path / 'work')

    settings = {'bucket_count': bucket_count, 'concurrency': 3}

    colocalisation_view(str(tmp_path / 'output'), destination, settings, config)

    module = importlib.import_module(colocalisation_view.__module__)
    assert OutputValidation.of(colocalisation_view, module).outputs == {'view': ColocalisationByCredibleSet}
    validate_dataset(ColocalisationByCredibleSet, destination['view'])
    written = pl.read_parquet(destination['view'])
    assert written.schema == to_polars_schema(ColocalisationByCredibleSet)
    assert len(list((tmp_path / 'view').glob('*.parquet'))) == bucket_count
    assert _sorted(written).equals(_sorted(colocalisation_view_frame(rows)))
    assert written['credibleSetId'].n_unique() == written.height == 50
    assert not list((tmp_path / 'work').iterdir())


def test_the_step_refuses_an_existing_destination(tmp_path: Path):
    _write_parts(_coloc_rows(10), tmp_path / 'output', 1)
    (tmp_path / 'view').mkdir()

    with pytest.raises(ValueError, match='already exists'):
        colocalisation_view(str(tmp_path / 'output'), {'view': str(tmp_path / 'view')}, {}, None)


def test_many_chunks_build_the_coloc_view():
    rows = pl.concat([_coloc_rows(600) for _ in range(5)], rechunk=False)
    assert rows.n_chunks() > 1

    view = colocalisation_view_frame(rows)

    assert view.height == 50
    assert view['colocalisation'].list.len().sum() == colocalisation_view_frame(_coloc_rows(600))[
        'colocalisation'
    ].list.len().sum()


# buckets


def test_buckets_keep_every_row_once(tmp_path: Path):
    frame = pl.DataFrame({'key': [f'k{i % 37}' for i in range(1000)], 'n': range(1000)})
    parts = []
    for n in range(3):
        parts.append(str(tmp_path / f'{n}.parquet'))
        frame.filter(pl.col('n') % 3 == n).write_parquet(parts[-1])
    (tmp_path / 'b').mkdir()

    written = write_buckets(parts, lambda f: f, 'key', 5, tmp_path / 'b', 2)
    buckets = for_each_bucket(lambda b: read_bucket(tmp_path / 'b', b), 5, 2)

    assert written == 1000
    present = [b for b in buckets if b is not None]
    assert sorted(pl.concat(present)['n'].to_list()) == list(range(1000))
    keys = [set(b['key']) for b in present]
    assert sum(len(k) for k in keys) == len(set().union(*keys)) == 37


def test_a_bucket_no_part_reached_is_none(tmp_path: Path):
    assert read_bucket(tmp_path, 3) is None


# l2g_predictions


def test_predictions_by_score_descending_ties_by_gene():
    """`reverse(arraySort(score))` left ties in insertion order; pts orders them by geneId."""
    predictions = _rows(
        L2gPrediction,
        {'credibleSetId': _id(2), 'geneId': 'ENSG00000000001', 'score': 0.1},
        {'credibleSetId': _id(1), 'geneId': 'ENSG00000000003', 'score': 0.5},
        {'credibleSetId': _id(1), 'geneId': 'ENSG00000000002', 'score': 0.9},
        {'credibleSetId': _id(1), 'geneId': 'ENSG00000000004', 'score': 0.5},
        {'credibleSetId': _id(1), 'geneId': 'ENSG00000000001', 'score': 0.5},
    )

    view = l2g_prediction_view_frame(predictions)

    assert view.columns == ['credibleSetId', 'l2g_predictions']
    by_set = _by_key(view)
    assert list(by_set) == [_id(1), _id(2)]
    assert [(p['geneId'][-1], p['score']) for p in by_set[_id(1)]] == [
        ('2', 0.9),
        ('1', 0.5),
        ('3', 0.5),
        ('4', 0.5),
    ]
    assert list(by_set[_id(1)][0]) == ['credibleSetId', 'geneId', 'score', 'features', 'shapBaseValue']
    assert by_set[_id(2)][0]['features'] == L2gPrediction.example['features']


def _predictions(count: int, start: int = 0) -> pl.DataFrame:
    return _rows(
        L2gPrediction,
        *(
            {'credibleSetId': _id(i // 3), 'geneId': f'ENSG{i:011d}', 'score': (i % 7) / 7}
            for i in range(start, start + count)
        ),
    )


def test_many_chunks_build_the_l2g_view():
    frame = pl.concat([_predictions(600, start) for start in range(0, 3000, 600)], rechunk=False)
    assert frame.n_chunks() > 1

    view = l2g_prediction_view_frame(frame)

    assert view.height == 1000
    assert view['l2g_predictions'].list.len().sum() == 3000
    scores = view['l2g_predictions'].list.eval(pl.element().struct.field('score'))
    assert scores.to_list() == scores.list.sort(descending=True).to_list()


def test_l2g_step_writes_its_model(tmp_path: Path):
    module = importlib.import_module('pts.transformers.l2g_prediction_view')
    (tmp_path / 'output').mkdir()
    _predictions(6).write_parquet(tmp_path / 'output' / 'part-0.parquet')
    destination = {'view': str(tmp_path / 'view')}

    module.l2g_prediction_view(str(tmp_path / 'output'), destination, {}, None)

    assert OutputValidation.of(module.l2g_prediction_view, module).outputs == {'view': L2gPredictionByCredibleSet}
    validate_dataset(L2gPredictionByCredibleSet, destination['view'])
    written = pl.read_parquet(destination['view'])
    assert written.schema == to_polars_schema(L2gPredictionByCredibleSet)
    assert written.height == 2
