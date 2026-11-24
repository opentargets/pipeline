"""Tests for the target view transformer."""

import json
from pathlib import Path

import polars as pl
import pytest
from ot_datamodels import DatasetModel
from ot_datamodels.outputs.chemical_probe import ChemicalProbe
from ot_datamodels.outputs.credible_set import CredibleSet
from ot_datamodels.outputs.homology import Homology
from ot_datamodels.outputs.pharmacogenomics import Pharmacogenomics
from ot_datamodels.outputs.study import Study
from ot_datamodels.outputs.target import Target
from ot_datamodels.outputs.target_safety_event import TargetSafetyEvent
from ot_datamodels.outputs.target_tractability import TargetTractability
from ot_datamodels.outputs.transcript import Transcript
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.views.target_by_region import TargetByRegion
from ot_datamodels.views.target_view import TargetView
from ot_datamodels.views.transcript_view import TranscriptView

from pts.schemas.dataset.conform import conform, without_enums
from pts.schemas.dataset.validate import validate_dataset
from pts.schemas.target import TARGET
from pts.task_outputs import OutputValidation
from pts.transformers import target_view as module
from pts.transformers.target_view import target_view
from tests.credible_set_fixtures import published_credible_set
from tests.target_view_spark_rows import EMPTY_TRANSCRIPT, NULLED_BY_EMPTY, OUTPUT

TARGET_VIEW = to_polars_schema(TargetView)
#: the id lists targets.sql joined on, after the columns the spark golden rows hold
ID_LISTS = ['credibleSetIds', 'pharmacogenomicsIds']
VIEW_COLUMNS = [name for name in TARGET_VIEW if name not in TARGET and name not in ID_LISTS]

EXON = pl.Struct({'exonId': pl.String, 'chromosome': pl.String, 'start': pl.Int64, 'end': pl.Int64, 'strand': pl.Int32})
FLAG = pl.Struct({'label': pl.String, 'value': pl.String})
EFFECT = pl.Struct({'direction': pl.String, 'dosing': pl.String})
BIOSAMPLE = pl.Struct({
    'tissueLabel': pl.String,
    'tissueId': pl.String,
    'cellLabel': pl.String,
    'cellFormat': pl.String,
    'cellId': pl.String,
})
STUDY = pl.Struct({'description': pl.String, 'name': pl.String, 'type': pl.String})
URL = pl.Struct({'niceName': pl.String, 'url': pl.String})

INPUT_SCHEMAS: dict[str, pl.Schema] = {
    'target': TARGET,
    'homology': pl.Schema({
        'targetId': pl.String,
        'speciesId': pl.String,
        'speciesName': pl.String,
        'homologyType': pl.String,
        'targetGeneId': pl.String,
        'isHighConfidence': pl.Boolean,
        'targetGeneSymbol': pl.String,
        'queryPercentageIdentity': pl.Float64,
        'targetPercentageIdentity': pl.Float64,
        'priority': pl.Int32,
    }),
    'target_tractability': pl.Schema({
        'targetId': pl.String,
        'modality': pl.String,
        'category': pl.String,
        'value': pl.Boolean,
    }),
    'target_safety_event': pl.Schema({
        'targetId': pl.String,
        'event': pl.String,
        'eventId': pl.String,
        'effects': pl.List(EFFECT),
        'biosamples': pl.List(BIOSAMPLE),
        'datasource': pl.String,
        'literature': pl.String,
        'url': pl.String,
        'studies': pl.List(STUDY),
    }),
    'chemical_probes': pl.Schema({
        'targetFromSourceId': pl.String,
        'drugFromSource': pl.String,
        'drugFromSourceId': pl.String,
        'drugId': pl.String,
        'mechanismOfAction': pl.List(pl.String),
        'origin': pl.List(pl.String),
        'control': pl.String,
        'isHighQuality': pl.Boolean,
        'probesDrugsScore': pl.Float64,
        'scoreInCells': pl.Float64,
        'scoreInOrganisms': pl.Float64,
        'urls': pl.List(URL),
        'targetId': pl.String,
    }),
    'transcript': pl.Schema({
        'targetId': pl.String,
        'transcriptId': pl.String,
        'biotype': pl.String,
        'proteinId': pl.String,
        'uniprotSwissprotIds': pl.List(pl.String),
        'uniprotTremblIds': pl.List(pl.String),
        'uniprotIsoformIds': pl.List(pl.String),
        'alphafoldIds': pl.List(pl.String),
        'dbXrefs': pl.List(pl.Struct({'id': pl.String, 'source': pl.String})),
        'isEnsemblCanonical': pl.Boolean,
        'chromosome': pl.String,
        'start': pl.Int64,
        'end': pl.Int64,
        'strand': pl.Int32,
        'transcriptionStartSite': pl.Int32,
        'flags': pl.List(FLAG),
        'exons': pl.List(EXON),
    }),
    'study': pl.Schema({'studyId': pl.String, 'geneId': pl.String}),
    'pharmacogenomics': pl.Schema({'pharmacogenomicsId': pl.Int64, 'targetFromSourceId': pl.String}),
}


def _target(target_id: str | None, symbol: str | None = None, **kwargs) -> dict:
    return {'id': target_id, 'approvedSymbol': symbol, **kwargs}


def _homologue(target_id: str | None, gene: str, priority: int | None, species: str = '10090') -> dict:
    return {
        'targetId': target_id,
        'speciesId': species,
        'speciesName': f'species {species}',
        'homologyType': 'ortholog_one2one',
        'targetGeneId': gene,
        'isHighConfidence': True,
        'targetGeneSymbol': gene.lower(),
        'queryPercentageIdentity': 90.5,
        'targetPercentageIdentity': 88.25,
        'priority': priority,
    }


def _exon(start: int | None, end: int | None) -> dict:
    return {'exonId': f'E{start}', 'chromosome': '1', 'start': start, 'end': end, 'strand': 1}


def _transcript(target_id: str | None, transcript_id: str | None, canonical: bool | None = False, **kwargs) -> dict:
    return {
        'targetId': target_id,
        'transcriptId': transcript_id,
        'biotype': 'protein_coding',
        'proteinId': f'P-{transcript_id}',
        'isEnsemblCanonical': canonical,
        'chromosome': '1',
        'start': 100,
        'end': 900,
        'strand': -1,
        'transcriptionStartSite': 900,
        'flags': [{'label': 'mane', 'value': 'select'}],
        'exons': [],
        **kwargs,
    }


ROWS: dict[str, list[dict]] = {
    'target': [
        _target('ENSG1', 'A', genomicLocation={'chromosome': '1', 'start': 1, 'end': 2, 'strand': 1}, tss=5),
        _target('ENSG2', 'B'),
        _target('ENSG3', 'C'),
        _target('ENSG4', 'D'),
        _target('ENSG5', 'E'),
        _target('ENSG6', 'F'),
        _target('ENSG7', 'G'),
        _target(None, 'H'),
    ],
    'homology': [
        _homologue('ENSG1', 'G2', 2, '9598'),
        _homologue('ENSG1', 'G0', 0),
        _homologue('ENSG1', 'G1b', 1, '10116'),
        _homologue('ENSG1', 'G1a', 1, '10116'),
        _homologue('ENSG1', 'G0', 0),
        _homologue('ENSG6', 'G6', None),
        _homologue('ENSG9', 'G9', 0),
        _homologue(None, 'GN', 0),
    ],
    'target_tractability': [
        {'targetId': 'ENSG1', 'modality': 'SM', 'category': 'Bucket_2', 'value': True},
        {'targetId': 'ENSG1', 'modality': 'AB', 'category': 'Bucket_1', 'value': False},
        {'targetId': 'ENSG1', 'modality': 'SM', 'category': 'Bucket_2', 'value': True},
        {'targetId': 'ENSG1', 'modality': 'SM', 'category': 'Bucket_1', 'value': None},
        {'targetId': 'ENSG3', 'modality': 'PR', 'category': 'Bucket_3', 'value': True},
        {'targetId': None, 'modality': 'PR', 'category': 'Bucket_3', 'value': True},
    ],
    'target_safety_event': [
        {
            'targetId': 'ENSG1',
            'event': 'cardiotoxicity',
            'eventId': 'EFO_1',
            'effects': [{'direction': 'activation', 'dosing': 'chronic'}, {'direction': None, 'dosing': 'acute'}],
            'biosamples': [],
            'datasource': 'AOPWiki',
            'literature': None,
            'url': 'https://aopwiki.org',
            'studies': None,
        },
        {
            'targetId': 'ENSG1',
            'event': 'arrhythmia',
            'eventId': None,
            'effects': None,
            'biosamples': [
                {'tissueLabel': 'heart', 'tissueId': 'UBERON_1', 'cellLabel': None, 'cellFormat': None, 'cellId': None}
            ],
            'datasource': 'Brennan',
            'literature': '123',
            'url': None,
            'studies': [{'description': 'd', 'name': 'n', 'type': 'clinical'}],
        },
        {
            'targetId': 'ENSG1',
            'event': 'cardiotoxicity',
            'eventId': 'EFO_1',
            'effects': [{'direction': 'activation', 'dosing': 'chronic'}, {'direction': None, 'dosing': 'acute'}],
            'biosamples': [],
            'datasource': 'AOPWiki',
            'literature': None,
            'url': 'https://aopwiki.org',
            'studies': None,
        },
        {'targetId': 'ENSG5', 'event': 'liver', 'datasource': 'ToxCast', 'effects': [], 'studies': []},
    ],
    'chemical_probes': [
        {
            'targetFromSourceId': 'A',
            'drugFromSource': 'probe-2',
            'drugFromSourceId': 'PD2',
            'drugId': 'CHEMBL2',
            'mechanismOfAction': ['inhibitor', 'antagonist'],
            'origin': ['calculated'],
            'control': 'ctrl',
            'isHighQuality': True,
            'probesDrugsScore': 0.5,
            'scoreInCells': 0.25,
            'scoreInOrganisms': None,
            'urls': [{'niceName': 'Probes & Drugs', 'url': 'https://pd'}, {'niceName': 'SGC', 'url': None}],
            'targetId': 'ENSG1',
        },
        {
            'targetFromSourceId': 'A',
            'drugFromSource': 'probe-1',
            'mechanismOfAction': [],
            'origin': None,
            'isHighQuality': False,
            'urls': None,
            'targetId': 'ENSG1',
        },
        {'targetId': 'ENSG4'},
        {'drugFromSource': 'probe-9', 'targetId': 'ENSG9'},
    ],
    'transcript': [
        _transcript(
            'ENSG1',
            'ENST11',
            True,
            uniprotSwissprotIds=['P1', 'P2'],
            uniprotTremblIds=['A1'],
            exons=[_exon(500, 600), _exon(100, 200), _exon(100, 150), _exon(100, 200)],
        ),
        _transcript('ENSG1', 'ENST12', uniprotSwissprotIds=[], uniprotTremblIds=['A2', 'A3']),
        _transcript(
            'ENSG1',
            'ENST13',
            None,
            uniprotIsoformIds=['P1-2', 'P1-3'],
            alphafoldIds=['AF1', 'AF2'],
            proteinId=None,
        ),
        _transcript('ENSG1', None, uniprotTremblIds=[], alphafoldIds=[], uniprotIsoformIds=[]),
        _transcript('ENSG3', 'ENST31'),
        _transcript('ENSG3', 'ENST32', None),
        _transcript('ENSG4', 'ENST41', True, exons=[], chromosome='X', start=5, end=50, strand=1),
        _transcript('ENSG5', 'ENST51', True, exons=[_exon(300, 400)]),
        _transcript('ENSG5', 'ENST52', True, exons=[_exon(10, 20)], start=10, end=20),
        _transcript('ENSG6', 'ENST61', True, exons=[None, _exon(None, 5), _exon(7, None), _exon(3, 4)]),
        _transcript('ENSG7', 'ENST71', True, exons=None),
        _transcript('ENSG9', 'ENST91', True, exons=[_exon(1, 2)]),
        _transcript(None, 'ENSTN', True, exons=[_exon(1, 2)]),
    ],
    'study': [
        {'studyId': 'S1', 'geneId': 'ENSG1'},
        {'studyId': 'S2', 'geneId': 'ENSG1'},
        {'studyId': 'S3', 'geneId': None},
        {'studyId': 'S4', 'geneId': 'ENSG9'},
        {'studyId': 'S5', 'geneId': 'ENSG3'},
    ],
    'credible_set': [
        {'id': 'cs-b', 'studyId': 'S1'},
        {'id': 'cs-a', 'studyId': 'S1'},
        {'id': 'cs-a', 'studyId': 'S2'},
        {'id': 'cs-c', 'studyId': 'S2'},
        {'id': '', 'studyId': 'S2'},
        {'id': 'cs-x', 'studyId': 'S3'},
        {'id': 'cs-y', 'studyId': 'S4'},
        {'id': 'cs-z', 'studyId': 'S9'},
    ],
    'pharmacogenomics': [
        {'pharmacogenomicsId': 3, 'targetFromSourceId': 'ENSG1'},
        {'pharmacogenomicsId': 1, 'targetFromSourceId': 'ENSG1'},
        {'pharmacogenomicsId': 2, 'targetFromSourceId': None},
        {'pharmacogenomicsId': 5, 'targetFromSourceId': 'ENSG9'},
        {'pharmacogenomicsId': 4, 'targetFromSourceId': 'ENSG4'},
    ],
}


#: The inputs pts writes with a model.
MODELS: dict[str, type[DatasetModel]] = {
    'target': Target,
    'homology': Homology,
    'target_tractability': TargetTractability,
    'target_safety_event': TargetSafetyEvent,
    'transcript': Transcript,
}


def _as_written(name: str, frame: pl.DataFrame) -> pl.DataFrame:
    """The input as pts writes it: its model's Enums."""
    return conform(frame, MODELS[name])


def _write(root: Path, rows: dict[str, list[dict]], *, typed: bool = False) -> dict[str, str]:
    source = {}
    for name, schema in INPUT_SCHEMAS.items():
        (root / name).mkdir(parents=True)
        frame = pl.DataFrame(rows.get(name, []), schema=schema)
        if typed and name in MODELS:
            frame = _as_written(name, frame)
        frame.write_parquet(root / name / 'part-0.parquet')
        source[name] = str(root / name)
    (root / 'credible_set').mkdir()
    published_credible_set(*rows.get('credible_set', [])).write_parquet(root / 'credible_set' / 'part-0.parquet')
    source['credible_set'] = str(root / 'credible_set')
    return source


DESTINATIONS = ('view', 'by_region', 'transcript')


def _run_all(tmp_path: Path, rows: dict[str, list[dict]] = ROWS, *, typed: bool = False) -> dict[str, pl.DataFrame]:
    destination = {key: str(tmp_path / 'out' / key) for key in DESTINATIONS}
    target_view(_write(tmp_path / 'src', rows, typed=typed), destination, {}, None)
    return {key: pl.read_parquet(path) for key, path in destination.items()}


def _run(tmp_path: Path, rows: dict[str, list[dict]] = ROWS, *, typed: bool = False) -> pl.DataFrame:
    return _run_all(tmp_path, rows, typed=typed)['view']


def _key(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _normalise(row: dict) -> dict:
    return {
        column: sorted(value, key=_key) if isinstance(value, list) and column != 'canonicalExons' else value
        for column, value in row.items()
    }


def _rows(df: pl.DataFrame) -> list[dict]:
    return sorted((_normalise(r) for r in df.to_dicts()), key=_key)


def _view_rows(df: pl.DataFrame) -> list[dict]:
    return _rows(df.select('id', *VIEW_COLUMNS))


def test_matches_spark_golden(tmp_path: Path):
    result = _run(tmp_path)

    assert _view_rows(result) == OUTPUT
    # the fixture reaches every case it is built for
    by_id = {row['id']: row for row in result.to_dicts()}
    assert result['id'].to_list().count('ENSG5') == 2
    assert by_id['ENSG5']['canonicalExons'] == ['10', '20', '300', '400']
    assert by_id['ENSG6']['canonicalExons'] == [None, None, None, '5', '3', '4', '7', None]
    assert by_id['ENSG1']['canonicalExons'] == ['100', '150', '100', '200', '100', '200', '500', '600']
    assert by_id['ENSG4']['canonicalTranscript']['id'] == 'ENST41'
    assert by_id['ENSG4']['canonicalExons'] is None
    assert by_id['ENSG7']['canonicalExons'] is None
    assert by_id['ENSG3']['canonicalTranscript'] is None
    assert by_id['ENSG1']['transcriptIds'] == ['ENST11', 'ENST12', 'ENST13']
    assert len(by_id['ENSG1']['transcripts']) == 4
    reviewed = {t['transcriptId']: (t['uniprotId'], t['isUniprotReviewed']) for t in by_id['ENSG1']['transcripts']}
    assert reviewed == {'ENST11': ('P1', True), 'ENST12': ('A2', True), 'ENST13': (None, None), None: (None, False)}
    assert len(by_id['ENSG1']['homologues']) == 5
    assert len(by_id['ENSG1']['tractability']) == 4
    assert len(by_id['ENSG1']['safetyLiabilities']) == 3
    assert by_id['ENSG6']['homologues'][0]['priority'] is None
    assert by_id['ENSG2']['homologues'] is None
    assert by_id[None]['transcripts'] is None


@pytest.mark.parametrize('empty', list(NULLED_BY_EMPTY))
def test_empty_input_nulls_its_columns(tmp_path: Path, empty: str):
    result = _run(tmp_path, ROWS | {empty: []})
    expected = [{k: None if k in NULLED_BY_EMPTY[empty] else v for k, v in row.items()} for row in OUTPUT]

    assert result.schema == TARGET_VIEW
    assert _view_rows(result) == sorted(expected, key=_key)


def test_empty_transcript_matches_spark_golden(tmp_path: Path):
    result = _run(tmp_path, ROWS | {'transcript': []})

    assert result.schema == TARGET_VIEW
    assert _view_rows(result) == EMPTY_TRANSCRIPT


def test_empty_target_writes_no_rows(tmp_path: Path):
    result = _run(tmp_path, ROWS | {'target': []})

    assert result.schema == TARGET_VIEW
    assert result.is_empty()


def test_null_priority_among_homologues_raises(tmp_path: Path):
    rows = ROWS | {'homology': [_homologue('ENSG1', 'G0', 0), _homologue('ENSG1', 'GN', None)]}
    with pytest.raises(ValueError, match='without a priority'):
        _run(tmp_path, rows)


def test_target_columns_pass_through(tmp_path: Path):
    result = _run(tmp_path).select(list(TARGET)).unique(maintain_order=True)
    assert result.equals(pl.DataFrame(ROWS['target'], schema=TARGET))


def test_schema_is_published(tmp_path: Path):
    assert _run(tmp_path).schema == TARGET_VIEW


def test_inputs_as_pts_writes_them_give_the_same_view(tmp_path: Path):
    """The inputs' Enums reach the view as `TargetView`'s, and homology's bool as a bool."""
    written = {name: _as_written(name, pl.DataFrame(ROWS.get(name, []), schema=INPUT_SCHEMAS[name])) for name in MODELS}
    assert isinstance(written['transcript'].schema['chromosome'], pl.Enum)
    assert isinstance(written['homology'].schema['homologyType'], pl.Enum)

    result = _run(tmp_path, typed=True)

    assert result.schema == TARGET_VIEW
    homologues = result['homologues'].explode().struct.field('isHighConfidence')
    assert homologues.drop_nulls().unique().to_list() == [True]
    assert _view_rows(result) == OUTPUT


def test_writes_its_model(tmp_path: Path):
    """On one row of each input, its model's example for the same target, the three views pass their models."""
    target_id = Target.example['id']
    examples: dict[str, tuple[type[DatasetModel], dict]] = {
        **{name: (model, {'targetId': target_id}) for name, model in MODELS.items() if model is not Target},
        'target': (Target, {}),
        'chemical_probes': (ChemicalProbe, {'targetId': target_id}),
        'study': (Study, {'geneId': target_id}),
        'credible_set': (CredibleSet, {'studyId': Study.example['studyId']}),
        'pharmacogenomics': (Pharmacogenomics, {'targetFromSourceId': target_id}),
    }
    source = {}
    for name, (model, overrides) in examples.items():
        (tmp_path / name).mkdir()
        frame = pl.DataFrame([model.example | overrides], schema=to_polars_schema(model))
        frame.write_parquet(tmp_path / name / 'part-0.parquet')
        source[name] = str(tmp_path / name)
    destination = {key: str(tmp_path / 'out' / key) for key in DESTINATIONS}

    target_view(source, destination, {}, None)

    models = {'view': TargetView, 'by_region': TargetByRegion, 'transcript': TranscriptView}
    assert OutputValidation.of(target_view, module).outputs == models
    for key, model in models.items():
        validate_dataset(model, destination[key])
        assert pl.read_parquet(destination[key]).schema == to_polars_schema(model), key
    written = pl.read_parquet(destination['view'])
    assert written['homologues'].explode().struct.field('isHighConfidence').to_list() == [
        Homology.example['isHighConfidence']
    ]
    ids = ([CredibleSet.example['id']], [Pharmacogenomics.example['pharmacogenomicsId']])
    assert written.select(ID_LISTS).row(0) == ids


def test_rows_in_target_order(tmp_path: Path):
    ids = [row['id'] for row in ROWS['target']]
    assert _run(tmp_path)['id'].to_list() == [*ids[:5], 'ENSG5', *ids[5:]]


def test_homologues_by_priority_then_content(tmp_path: Path):
    row = _run(tmp_path).row(0, named=True)
    assert [(h['priority'], h['targetGeneId']) for h in row['homologues']] == [
        (0, 'G0'),
        (0, 'G0'),
        (1, 'G1a'),
        (1, 'G1b'),
        (2, 'G2'),
    ]


def test_lists_sorted(tmp_path: Path):
    """By their values as strings, as the view sorts them before its Enums are cast."""
    result = without_enums(_run(tmp_path))
    for column in ['tractability', 'safetyLiabilities', 'chemicalProbes', 'transcriptIds', 'transcripts']:
        assert result[column].to_list() == result[column].list.sort().to_list(), column


def test_alternative_genes_null(tmp_path: Path):
    assert _run(tmp_path)['alternativeGenes'].null_count() == len(ROWS['target']) + 1


def test_credible_set_ids(tmp_path: Path):
    """Distinct and sorted, from every study of the gene; `[]` for a target with none, a study without credible
    sets included; null `geneId`, genes that are not targets, empty ids and credible sets of no study ignored."""
    ids = dict(_run(tmp_path).select('id', 'credibleSetIds').unique('id', keep='first').iter_rows())
    assert ids == {**{t['id']: [] for t in ROWS['target']}, 'ENSG1': ['cs-a', 'cs-b', 'cs-c']}


def test_pharmacogenomics_ids(tmp_path: Path):
    """Ascending; `[]` for none; null `targetFromSourceId` and ids that are not targets ignored."""
    ids = dict(_run(tmp_path).select('id', 'pharmacogenomicsIds').unique('id', keep='first').iter_rows())
    assert ids == {**{t['id']: [] for t in ROWS['target']}, 'ENSG1': [1, 3], 'ENSG4': [4]}


def test_empty_id_sources_give_empty_lists(tmp_path: Path):
    result = _run(tmp_path, ROWS | {'study': [], 'credible_set': [], 'pharmacogenomics': []})
    assert result.schema == TARGET_VIEW
    for column in ID_LISTS:
        assert result[column].to_list() == [[]] * result.height, column


def test_target_by_region_is_one_row_per_target(tmp_path: Path):
    frames = _run_all(tmp_path)
    view, by_region = frames['view'], frames['by_region']
    assert by_region.schema == to_polars_schema(TargetByRegion)
    assert sorted(by_region['target'].struct.field('id').to_list(), key=str) == sorted(view['id'].to_list(), key=str)
    (located,) = by_region.filter(pl.col('target').struct.field('id') == 'ENSG1').to_dicts()
    assert located == {'chromosome': '1', 'start': 1, 'end': 2, 'target': {'id': 'ENSG1'}}


def test_transcript_view_groups_the_transcripts_by_target(tmp_path: Path):
    """One row per targetId; each transcript row without targetId, in the transcript's column order, by
    transcriptId; flags as strings."""
    transcript = _run_all(tmp_path, typed=True)['transcript']
    assert transcript.schema == to_polars_schema(TranscriptView)
    rows = {r['targetId']: r['transcripts'] for r in transcript.to_dicts()}
    assert set(rows) == {t['targetId'] for t in ROWS['transcript']}
    assert [t['transcriptId'] for t in rows['ENSG1']] == [None, 'ENST11', 'ENST12', 'ENST13']
    assert [t['transcriptId'] for t in rows['ENSG5']] == ['ENST51', 'ENST52']
    first = rows['ENSG1'][1]
    assert list(first) == [name for name in INPUT_SCHEMAS['transcript'] if name != 'targetId']
    source = next(t for t in ROWS['transcript'] if t['transcriptId'] == 'ENST11')
    (written,) = pl.DataFrame([source], schema=INPUT_SCHEMAS['transcript']).drop('targetId').to_dicts()
    assert first == written
