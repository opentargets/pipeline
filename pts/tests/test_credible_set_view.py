"""Tests for the credible_set_view transformer, which replaces `credible_sets.sql` and the three read-model scripts.

Each case carries over a semantic of the scripts (their LEFT JOIN defaults, the lead-row INNER JOIN), or the tie
order pts now fixes (locus: posteriorProbability descending, then variantId; top L2G: score descending, then
geneId).
"""

import importlib
from pathlib import Path

import polars as pl
import pytest
from ot_datamodels import DatasetModel
from ot_datamodels.outputs.biosample import Biosample
from ot_datamodels.outputs.disease import Disease
from ot_datamodels.outputs.l2g_prediction import L2gPrediction
from ot_datamodels.outputs.study import Study
from ot_datamodels.outputs.target import Target
from ot_datamodels.outputs.variant import Variant
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.views.credible_set_locus import LOCUS_FIELDS
from otter.config.model import Config

from pts.schemas.dataset.validate import validate_dataset
from pts.task_outputs import OutputValidation
from pts.transformers.credible_set_view import VIEWS, credible_set_view, credible_set_view_frames
from tests.credible_set_fixtures import published_credible_set, published_credible_set_variant

S1, S2, S3, S4, S5 = (c * 32 for c in 'abcde')
T1, T2, NOT_A_TARGET = 'ENSG00000000001', 'ENSG00000000002', 'ENSG00000000099'
G1, G2, G3 = 'ENSG00000000011', 'ENSG00000000012', 'ENSG00000000013'
D1, D2, TA1, TA2, UNKNOWN = 'EFO_0000001', 'EFO_0000002', 'EFO_0000010', 'EFO_0000020', 'EFO_0009999'
BLOOD, NOWHERE = 'UBERON_0000178', 'UBERON_0009999'


def _rows(model: type[DatasetModel], *rows: dict) -> pl.DataFrame:
    return pl.DataFrame([model.example | row for row in rows], schema=to_polars_schema(model))


def _set(set_id: str, study: str, lead: str, study_type: str = 'eqtl', **row) -> dict:
    return {
        'id': set_id,
        'studyId': study,
        'studyType': study_type,
        'chromosome': lead.split('_', maxsplit=1)[0],
        'leadVariantId': lead,
        'locusStart': 1,
        'locusEnd': 1000,
        'fineMappingMethod': 'SuSiE',
        'confidence': 3,
        **row,
    }


def _member(set_id: str, variant: str, pp: float, **row) -> dict:
    return {'credibleSetId': set_id, 'variantId': variant, 'posteriorProbability': pp, **row}


def _inputs() -> dict[str, pl.DataFrame]:
    """The design's matrix: S1 a GWAS set with tied PPs, a tied top score, repeated and unknown disease ids and a
    therapeutic area that is no disease; S2 a trans QTL set without L2G whose lead is not a variant; S3 a set whose
    top gene is not a target and whose study's biosample is unknown; S4 a second set of S2's study; S5 a set of a
    study whose gene is not a target."""
    credible_set = published_credible_set(
        _set(S1, 'GWAS1', '1_100_A_G', 'gwas', qualityControls=['Study locus from curated top hit']),
        _set(S2, 'QTL1', '2_200_C_A', qualityControls=['Trans QTL']),
        _set(S3, 'QTL2', '3_300_T_C'),
        _set(S4, 'QTL1', '4_400_G_T', 'gwas'),
        _set(S5, 'QTL3', '5_500_A_C'),
    )
    credible_set_variant = published_credible_set_variant(
        _member(S1, '1_100_A_G', 0.5, pValueMantissa=1.5, pValueExponent=-10, reportedBeta=0.2, sampleSize=100),
        _member(S1, '1_150_C_T', 0.25),
        _member(S1, '1_120_G_C', 0.25),
        _member(S2, '2_200_C_A', 0.9, qualityControls=['Allele frequency taken from gnomAD']),
        _member(S3, '3_300_T_C', 1.0),
        _member(S4, '4_400_G_T', 0.6),
        _member(S4, '4_410_G_T', 0.4),
        _member(S5, '5_500_A_C', 1.0),
    )
    variant = _rows(
        Variant,
        *(
            {'variantId': v, 'referenceAllele': v.split('_')[2], 'alternateAllele': v.split('_')[3]}
            for v in ('1_100_A_G', '3_300_T_C', '4_400_G_T', '5_500_A_C')
        ),
    )
    study = _rows(
        Study,
        {
            'studyId': 'GWAS1',
            'studyType': 'gwas',
            'traitFromSource': 'trait one',
            'diseaseIds': [D1, UNKNOWN, D1, D2],
            'geneId': None,
            'biosampleFromSourceId': None,
            'condition': None,
        },
        {
            'studyId': 'QTL1',
            'traitFromSource': 'expression of one',
            'diseaseIds': [],
            'geneId': T1,
            'biosampleFromSourceId': BLOOD,
            'condition': 'naive',
        },
        {
            'studyId': 'QTL2',
            'traitFromSource': 'expression of two',
            'diseaseIds': [TA1],
            'geneId': T2,
            'biosampleFromSourceId': NOWHERE,
            'condition': None,
        },
        {
            'studyId': 'QTL3',
            'traitFromSource': 'expression of none',
            'diseaseIds': [],
            'geneId': NOT_A_TARGET,
            'biosampleFromSourceId': BLOOD,
            'condition': 'treated',
        },
    )
    l2g_prediction = _rows(
        L2gPrediction,
        {'credibleSetId': S1, 'geneId': G2, 'score': 0.9},
        {'credibleSetId': S1, 'geneId': G3, 'score': 0.1},
        {'credibleSetId': S1, 'geneId': G1, 'score': 0.9},
        {'credibleSetId': S3, 'geneId': NOT_A_TARGET, 'score': 0.7},
        {'credibleSetId': S4, 'geneId': T1, 'score': 0.3},
    )
    target = _rows(
        Target,
        {'id': T1, 'approvedSymbol': 'ONE'},
        {'id': T2, 'approvedSymbol': 'TWO'},
        {'id': G1, 'approvedSymbol': 'GONE'},
        {'id': G2, 'approvedSymbol': 'GTWO'},
    )
    disease = _rows(
        Disease,
        {'id': D1, 'name': 'disease one', 'therapeuticAreas': [TA2, 'EFO_0000404', TA1]},
        {'id': D2, 'name': 'disease two', 'therapeuticAreas': [TA1]},
        {'id': TA1, 'name': 'area one', 'therapeuticAreas': []},
        {'id': TA2, 'name': 'area two', 'therapeuticAreas': [TA2]},
    )
    biosample = _rows(Biosample, {'biosampleId': BLOOD, 'biosampleName': 'blood'})
    return {
        'credible_set': credible_set,
        'credible_set_variant': credible_set_variant,
        'variant': variant,
        'study': study,
        'l2g_prediction': l2g_prediction,
        'target': target,
        'disease': disease,
        'biosample': biosample,
    }


def _views(inputs: dict[str, pl.DataFrame] | None = None) -> dict[str, pl.DataFrame]:
    return credible_set_view_frames(**(inputs or _inputs()))


def _by(frame: pl.DataFrame, key: str) -> dict:
    return {row[key]: row for row in frame.to_dicts()}


def test_views_have_their_models_schema():
    for key, frame in _views().items():
        assert frame.schema == to_polars_schema(VIEWS[key]), key


def test_credible_set_carries_its_lead_statistics_and_size():
    view = _by(_views()['view'], 'credibleSetId')

    assert list(view) == [S1, S2, S3, S4, S5]
    s1 = view[S1]
    assert (s1['leadPosition'], s1['locusSize'], s1['isTransQtl']) == (100, 3, 0)
    assert (s1['leadPValueMantissa'], s1['leadPValueExponent'], s1['leadBeta'], s1['leadSampleSize']) == (
        1.5,
        -10,
        0.2,
        100,
    )
    assert s1['qualityControls'] == ['Study locus from curated top hit']
    assert (view[S2]['isTransQtl'], view[S2]['locusSize'], view[S2]['leadPosition']) == (1, 1, 200)


def test_locus_by_probability_ties_by_variant():
    """`arrayReverseSort` left equal probabilities in groupArray order; pts orders them by variantId."""
    locus = _by(_views()['locus'], 'credibleSetId')

    assert [(v['variantId'], v['posteriorProbability']) for v in locus[S1]['locus']] == [
        ('1_100_A_G', 0.5),
        ('1_120_G_C', 0.25),
        ('1_150_C_T', 0.25),
    ]
    assert list(locus[S2]['locus'][0]) == LOCUS_FIELDS
    assert locus[S2]['locus'][0]['qualityControls'] == ['Allele frequency taken from gnomAD']


def test_study_credible_sets_are_the_gwas_sets_with_their_top_gene():
    study = _by(_views()['study'], 'credibleSetId')

    assert list(study) == [S1, S4]
    s1 = study[S1]
    # tied top score: the lowest geneId
    assert (s1['topL2GGeneId'], s1['topL2GApprovedSymbol'], s1['topL2GScore']) == (G1, 'GONE', 0.9)
    assert (s1['leadReferenceAllele'], s1['leadAlternateAllele'], s1['leadChromosome']) == ('A', 'G', '1')
    assert (s1['pValueMantissa'], s1['pValueExponent'], s1['beta'], s1['locusSize']) == (1.5, -10, 0.2, 3)
    assert (study[S4]['topL2GGeneId'], study[S4]['topL2GApprovedSymbol']) == (T1, 'ONE')


def test_top_gene_null_without_predictions_symbol_null_for_a_non_target():
    variant = _views()['variant']
    top = {
        r['credibleSetId']: (r['topL2GGeneId'], r['topL2GApprovedSymbol'], r['topL2GScore']) for r in variant.to_dicts()
    }

    assert top[S2] == (None, None, None)
    assert top[S3] == (NOT_A_TARGET, None, 0.7)


def test_target_credible_sets_from_the_studies_of_each_target():
    """targets.credibleSetIds: the sets of the studies whose geneId is a target; a study gene that is not a target
    gives no row; the biosample name is '' when unknown."""
    target = _views()['target']

    assert target.select('targetId', 'credibleSetId').rows() == [(T1, S2), (T1, S4), (T2, S3)]
    rows = {r['credibleSetId']: r for r in target.to_dicts()}
    assert (rows[S2]['biosampleId'], rows[S2]['biosampleName'], rows[S2]['condition']) == (BLOOD, 'blood', 'naive')
    assert (rows[S3]['biosampleId'], rows[S3]['biosampleName'], rows[S3]['condition']) == (NOWHERE, '', None)
    # the lead is not a variant: alleles ''
    assert (rows[S2]['leadReferenceAllele'], rows[S2]['leadAlternateAllele'], rows[S2]['isTransQtl']) == ('', '', 1)
    assert (rows[S2]['studyType'], rows[S4]['studyType']) == ('eqtl', 'gwas')


def test_variant_credible_sets_one_row_per_set_variant():
    variant = _views()['variant']

    assert variant.height == 8
    assert sorted(variant.select('credibleSetId', 'variantId', 'posteriorProbability').rows())[:3] == [
        (S1, '1_100_A_G', 0.5),
        (S1, '1_120_G_C', 0.25),
        (S1, '1_150_C_T', 0.25),
    ]


def test_variant_credible_sets_study_info():
    """Diseases in diseaseIds order, a repeat kept, an unknown id left out; their therapeutic areas distinct by id,
    one that is no disease left out; NULL target, symbol and biosample name when unknown."""
    rows = {r['credibleSetId']: r for r in _views()['variant'].unique('credibleSetId').to_dicts()}

    s1 = rows[S1]
    assert s1['diseases'] == [
        {'id': D1, 'name': 'disease one'},
        {'id': D1, 'name': 'disease one'},
        {'id': D2, 'name': 'disease two'},
    ]
    assert s1['diseaseTherapeuticAreas'] == [{'id': TA1, 'name': 'area one'}, {'id': TA2, 'name': 'area two'}]
    assert (s1['traitFromSource'], s1['condition'], s1['studyTargetId'], s1['studyTargetApprovedSymbol']) == (
        'trait one',
        None,
        None,
        None,
    )
    assert (s1['biosampleId'], s1['biosampleName']) == (None, None)
    s2 = rows[S2]
    assert (s2['diseases'], s2['diseaseTherapeuticAreas']) == ([], [])
    assert (s2['studyTargetId'], s2['studyTargetApprovedSymbol'], s2['biosampleName']) == (T1, 'ONE', 'blood')
    assert (rows[S3]['biosampleId'], rows[S3]['biosampleName']) == (NOWHERE, None)
    assert rows[S3]['diseases'] == [{'id': TA1, 'name': 'area one'}]
    assert rows[S3]['diseaseTherapeuticAreas'] == []
    assert (rows[S5]['studyTargetId'], rows[S5]['studyTargetApprovedSymbol']) == (NOT_A_TARGET, None)


def test_a_set_without_its_lead_row_raises():
    inputs = _inputs()
    inputs['credible_set_variant'] = inputs['credible_set_variant'].filter(pl.col('variantId') != '3_300_T_C')

    with pytest.raises(ValueError, match='1 credible sets have no credible_set_variant row for their lead'):
        _views(inputs)


def _many(count: int) -> dict[str, pl.DataFrame]:
    """`count` sets over `count // 3` studies, 1-4 variants each with tied probabilities, written in many chunks."""
    inputs = _inputs()
    sets, members, predictions = [], [], []
    for i in range(count):
        set_id = f'{i:032x}'
        lead = f'{1 + i % 22}_{1000 + i}_A_G'
        sets.append(_set(set_id, ['GWAS1', 'QTL1', 'QTL2', 'QTL3'][i % 4], lead, ['gwas', 'eqtl'][i % 2]))
        members.append(_member(set_id, lead, 0.5))
        members += [_member(set_id, f'{1 + i % 22}_{2000 + i + k}_C_T', 0.125) for k in range(i % 4)]
        if i % 3:
            predictions += [{'credibleSetId': set_id, 'geneId': g, 'score': 0.5} for g in (G2, G1)]
    chunked = [published_credible_set(*sets[n : n + 500]) for n in range(0, count, 500)]
    inputs['credible_set'] = pl.concat(chunked, rechunk=False)
    chunked = [published_credible_set_variant(*members[n : n + 700]) for n in range(0, len(members), 700)]
    inputs['credible_set_variant'] = pl.concat(chunked, rechunk=False)
    inputs['l2g_prediction'] = _rows(L2gPrediction, *predictions)
    return inputs


def test_many_chunks_build_the_views():
    inputs = _many(3000)
    assert inputs['credible_set_variant'].n_chunks() > 1

    views = _views(inputs)

    assert views['view'].height == views['locus'].height == 3000
    assert views['variant'].height == inputs['credible_set_variant'].height
    assert views['study'].height == 1500
    assert views['target'].height == 1500
    assert views['locus']['locus'].list.len().sum() == inputs['credible_set_variant'].height
    first = views['locus']['locus'].list.first().struct.field('posteriorProbability')
    assert first.unique().to_list() == [0.5]
    top = views['study'].filter(pl.col('topL2GGeneId').is_not_null())['topL2GGeneId'].unique().to_list()
    assert top == [G1]


def _write(frame: pl.DataFrame, folder: Path, parts: int) -> str:
    folder.mkdir(parents=True)
    size = max(1, -(-frame.height // parts))
    for n, start in enumerate(range(0, frame.height, size)):
        frame.slice(start, size).write_parquet(folder / f'part-{n:05d}.parquet')
    return str(folder)


def _sorted(frame: pl.DataFrame) -> pl.DataFrame:
    keys = [c for c in ('targetId', 'credibleSetId', 'variantId') if c in frame.columns]
    return frame.sort(keys)


@pytest.mark.parametrize('bucket_count', [1, 5])
def test_bucketed_step_writes_the_unbucketed_views(tmp_path: Path, bucket_count: int):
    inputs = _many(1200)
    source = {name: _write(frame, tmp_path / 'run' / name, 3) for name, frame in inputs.items()}
    destination = {key: str(tmp_path / 'run' / 'view' / VIEWS[key].dataset_name) for key in VIEWS}
    config = Config(step='credible_set_view', steps=['credible_set_view'], work_path=tmp_path / 'work')

    credible_set_view(source, destination, {'bucket_count': bucket_count, 'concurrency': 2}, config)

    module = importlib.import_module(credible_set_view.__module__)
    assert OutputValidation.of(credible_set_view, module).outputs == VIEWS
    expected = _views(inputs)
    for key, model in VIEWS.items():
        validate_dataset(model, destination[key])
        written = pl.read_parquet(destination[key])
        assert written.schema == to_polars_schema(model), key
        assert len(list(Path(destination[key]).glob('*.parquet'))) == bucket_count, key
        assert _sorted(written).equals(_sorted(expected[key])), key
    assert not list((tmp_path / 'work').iterdir())


def test_bucketed_step_writes_the_hand_written_views(tmp_path: Path):
    source = {name: _write(frame, tmp_path / 'run' / name, 2) for name, frame in _inputs().items()}
    destination = {key: str(tmp_path / 'view' / key) for key in VIEWS}
    config = Config(step='credible_set_view', steps=['credible_set_view'], work_path=tmp_path / 'work')

    credible_set_view(source, destination, {'bucket_count': 3, 'concurrency': 3}, config)

    written = {key: pl.read_parquet(destination[key]) for key in VIEWS}
    assert sorted(written['target'].select('targetId', 'credibleSetId').rows()) == [(T1, S2), (T1, S4), (T2, S3)]
    assert sorted(written['study']['credibleSetId']) == [S1, S4]
    locus = _by(written['locus'], 'credibleSetId')
    assert [v['variantId'] for v in locus[S1]['locus']] == ['1_100_A_G', '1_120_G_C', '1_150_C_T']
    tops = dict(written['study'].select('credibleSetId', 'topL2GGeneId').rows())
    assert tops == {S1: G1, S4: T1}
    assert written['variant'].height == 8


def test_the_step_refuses_an_existing_destination(tmp_path: Path):
    source = {name: _write(frame, tmp_path / 'run' / name, 1) for name, frame in _inputs().items()}
    destination = {key: str(tmp_path / 'view' / key) for key in VIEWS}
    Path(destination['variant']).mkdir(parents=True)

    with pytest.raises(ValueError, match='already exists'):
        credible_set_view(source, destination, {}, None)
