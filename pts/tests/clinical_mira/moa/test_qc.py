import polars as pl
import pytest

from pts.transformers.clinical_mira.moa import __main__ as cli
from pts.transformers.clinical_mira.moa import qc


def _write_dataset(root, name, frame):
    directory = root / name
    directory.mkdir(parents=True)
    frame.write_parquet(directory / 'part-0.parquet')


def _write_release(root, moa, candidates, precedence, associations):
    """Write the datasets the qc command reads under ``root``."""
    pairs = {'targetId': pl.String, 'diseaseId': pl.String}
    _write_dataset(
        root,
        'output/drug_mechanism_of_action',
        pl.DataFrame(
            moa, schema={'chemblIds': pl.List(pl.String), 'targets': pl.List(pl.String), 'curation': pl.String}
        ),
    )
    _write_dataset(
        root,
        'intermediate/drug_mechanism_of_action_candidates',
        pl.DataFrame(
            candidates,
            schema={
                'parentChemblId': pl.String,
                'chemblIds': pl.List(pl.String),
                'targets': pl.List(pl.String),
                'confidence': pl.Float64,
                'maximumClinicalStage': pl.String,
                'drugType': pl.String,
                'source': pl.String,
            },
            orient='row',
        ),
    )
    _write_dataset(root, 'output/evidence_clinical_precedence', pl.DataFrame(precedence, schema=pairs, orient='row'))
    _write_dataset(root, 'output/association_overall_direct', pl.DataFrame(associations, schema=pairs, orient='row'))


def _qc(tmp_path, moa, candidates, precedence_release=(), precedence_without=(), options=()):
    release, without, out = tmp_path / 'release', tmp_path / 'without', tmp_path / 'qc'
    _write_release(release, moa, candidates, list(precedence_release), list(precedence_release))
    _write_release(without, [], [], list(precedence_without), list(precedence_without))
    cli.main(['qc', '--release', str(release), '--without', str(without), '--out', str(out), *options])
    return out


def _candidate(drug, genes, confidence=0.9, stage='PHASE_1', chembl_ids=None, source='aact'):
    return (drug, chembl_ids or [drug], genes, confidence, stage, 'Small molecule', source)


def test_qc_precision_by_source(tmp_path):
    moa = [(['CHEMBL1'], ['ENSG1'], 'manual')]
    candidates = [
        _candidate('CHEMBL1', ['ENSG1']),
        _candidate('CHEMBL1', ['ENSG9']),
        _candidate('CHEMBL1', ['ENSG1'], source='literature'),
    ]
    text = (_qc(tmp_path, moa, candidates) / 'qc.md').read_text()
    section = text.split('## Strict precision by source\n')[1].split('##')[0]
    assert '| aact | 2 | 1 |' in section
    assert '| literature | 1 | 1 |' in section


def test_qc_release_filter_excludes_stages_per_source(tmp_path):
    moa = [(['CHEMBL1'], ['ENSG1'], 'manual')]
    candidates = [
        _candidate('CHEMBL1', ['ENSG1'], stage='APPROVAL'),
        _candidate('CHEMBL1', ['ENSG9'], stage='APPROVAL', source='literature'),
    ]
    default = (_qc(tmp_path / 'default', moa, candidates) / 'qc.md').read_text()
    assert 'candidates passing the release filter: 0' in default
    options = ['--exclude-stages', 'aact:APPROVAL', 'literature:']
    text = (_qc(tmp_path / 'open', moa, candidates, options=options) / 'qc.md').read_text()
    assert 'candidates passing the release filter: 1' in text
    assert '| literature | 1 | 0 |' in text


def test_qc_rejects_a_repeated_exclude_stages_source(tmp_path):
    with pytest.raises(SystemExit, match='aact'):
        _qc(tmp_path, [], [], options=['--exclude-stages', 'aact:APPROVAL', 'aact:', 'literature:'])


def test_release_filter_needs_every_source():
    candidates = pl.DataFrame(
        {'confidence': [0.9], 'maximumClinicalStage': ['PHASE_1'], 'source': ['literature']},
    )
    with pytest.raises(ValueError, match='literature'):
        qc.release_filter(candidates, 0.8, {'aact': ['APPROVAL']})


def test_qc_benchmarks_candidates_against_manual_only(tmp_path):
    moa = [
        (['CHEMBL1'], ['ENSG1'], 'manual'),
        (['CHEMBL3'], ['ENSG3'], 'electronic'),  # an electronic row is never ground truth
    ]
    candidates = [
        _candidate('CHEMBL1', ['ENSG9']),  # gene ChEMBL lacks: wrong
        _candidate('CHEMBL1', ['ENSG1']),  # right
        _candidate('CHEMBL3', ['ENSG3']),  # only an electronic row: not scorable
    ]
    text = (_qc(tmp_path, moa, candidates) / 'qc.md').read_text()
    assert '1/2' in text
    assert 'electronic rows published: 1' in text


def test_qc_applies_the_release_filter_to_candidates(tmp_path):
    moa = [(['CHEMBL1'], ['ENSG1'], 'manual')]
    candidates = [
        _candidate('CHEMBL1', ['ENSG1']),
        _candidate('CHEMBL1', ['ENSG9'], confidence=0.5),
        _candidate('CHEMBL1', ['ENSG9'], stage='APPROVAL'),
    ]
    assert '1/1' in (_qc(tmp_path, moa, candidates) / 'qc.md').read_text()


def test_qc_reports_new_evidence_pairs(tmp_path):
    out = _qc(
        tmp_path,
        [],
        [],
        precedence_release=[('ENSG1', 'EFO_1'), ('ENSG2', 'EFO_2')],
        precedence_without=[('ENSG1', 'EFO_1')],
    )
    assert pl.read_csv(out / 'new_evidence_pairs.csv').rows() == [('ENSG2', 'EFO_2')]
    assert 'new evidence pairs: 1' in (out / 'qc.md').read_text()


def _first_line(out):
    return (out / 'qc.md').read_text().splitlines()[0]


def _source_line(out, source):
    lines = (out / 'qc.md').read_text().splitlines()
    return next(line for line in lines if line.startswith(f'{source} '))


def _scored(rows):
    """Scored candidates of ``(source, phase_group, correct)`` rows."""
    return pl.DataFrame(
        rows, schema={'source': pl.String, 'phase_group': pl.String, 'correct': pl.Boolean}, orient='row'
    )


def test_source_verdicts_judge_each_source_on_its_own_early_phase_candidates():
    scored = _scored(
        [('aact', 'early', True)] * 30
        + [('literature', 'early', True)] * 20
        + [('literature', 'early', False)] * 10
        + [('literature', 'late', True)] * 100
    )
    verdicts = qc.source_verdicts(scored, 30)
    assert list(verdicts) == ['aact', 'literature']
    assert verdicts['aact'].line.startswith('aact GO: early-phase strict target precision 1.000 (30/30;')
    assert verdicts['literature'].line.startswith('literature NO-GO: early-phase strict target precision 0.667 (20/30;')


def test_pooled_precision_would_hide_a_failing_source():
    scored = _scored(
        [('aact', 'early', True)] * 300 + [('literature', 'early', True)] * 25 + [('literature', 'early', False)] * 5
    )
    assert qc.verdict(scored, 30).startswith('GO')
    assert qc.overall_verdict(qc.source_verdicts(scored, 30)).startswith('NO-GO')


def test_source_verdicts_show_a_source_without_scorable_candidates():
    verdicts = qc.source_verdicts(_scored([('aact', 'early', True)] * 30), 30)
    assert verdicts['literature'].label == 'UNSCORED'
    assert verdicts['literature'].line.startswith('literature UNSCORED')


def test_source_verdicts_include_a_source_outside_the_known_ones():
    verdicts = qc.source_verdicts(_scored([('patents', 'early', True)]), 30)
    assert list(verdicts) == ['aact', 'literature', 'patents']
    assert verdicts['patents'].line.startswith('patents INCONCLUSIVE')


@pytest.mark.parametrize(
    ('labels', 'expected'),
    [
        ({'aact': 'GO', 'literature': 'GO'}, 'GO'),
        ({'aact': 'GO', 'literature': 'NO-GO'}, 'NO-GO'),
        ({'aact': 'NO-GO', 'literature': 'INCONCLUSIVE'}, 'NO-GO'),
        ({'aact': 'GO', 'literature': 'INCONCLUSIVE'}, 'GO'),
        ({'aact': 'GO', 'literature': 'UNSCORED'}, 'GO'),
        ({'aact': 'INCONCLUSIVE', 'literature': 'UNSCORED'}, 'INCONCLUSIVE'),
        ({'aact': 'UNSCORED', 'literature': 'UNSCORED'}, 'UNSCORED'),
    ],
)
def test_overall_verdict_is_go_only_when_every_decisive_source_is_go(labels, expected):
    verdicts = {source: qc.Verdict(label, f'{source} {label}: details.') for source, label in labels.items()}
    line = qc.overall_verdict(verdicts)
    assert line.split(':')[0] == expected
    for source, label in labels.items():
        assert f'{source} {label}' in line


def test_qc_md_shows_the_overall_line_then_one_line_per_source(tmp_path):
    moa = [(['CHEMBL1'], ['ENSG1'], 'manual')]
    candidates = [_candidate('CHEMBL1', ['ENSG1'], stage='PHASE_2')] * 30 + [
        _candidate('CHEMBL1', ['ENSG9'], stage='PHASE_2', source='literature')
    ] * 30
    out = _qc(tmp_path, moa, candidates, options=['--exclude-stages', 'aact:APPROVAL', 'literature:APPROVAL'])
    lines = (out / 'qc.md').read_text().splitlines()
    assert lines[0].startswith('NO-GO: overall')
    assert lines[1].startswith('aact GO')
    assert lines[2].startswith('literature NO-GO')


def test_qc_no_go_below_090_with_enough_scorable_rows(tmp_path):
    moa = [(['CHEMBL1'], ['ENSG1'], 'manual')]
    candidates = [_candidate('CHEMBL1', ['ENSG9'], stage='PHASE_2')] * 30
    assert _first_line(_qc(tmp_path, moa, candidates)).startswith('NO-GO')


def test_qc_go_at_or_above_090_with_enough_scorable_rows(tmp_path):
    moa = [(['CHEMBL1'], ['ENSG1'], 'manual')]
    out = _qc(tmp_path, moa, [_candidate('CHEMBL1', ['ENSG1'], stage='PHASE_2')] * 30)
    assert _first_line(out).startswith('GO')


def test_qc_inconclusive_below_min_scorable(tmp_path):
    moa = [(['CHEMBL1'], ['ENSG1'], 'manual')]
    out = _qc(tmp_path, moa, [_candidate('CHEMBL1', ['ENSG1'], stage='PHASE_2')] * 29)
    assert _first_line(out).startswith('INCONCLUSIVE')
    assert _source_line(out, 'aact').startswith('aact INCONCLUSIVE')
    assert '29/29' in _source_line(out, 'aact')


def test_qc_min_scorable_is_an_option(tmp_path):
    moa = [(['CHEMBL1'], ['ENSG1'], 'manual')]
    out = _qc(tmp_path, moa, [_candidate('CHEMBL1', ['ENSG1'], stage='PHASE_2')] * 3, options=['--min-scorable', '3'])
    assert _first_line(out).startswith('GO')


def test_qc_unscored_when_no_early_scorable_rows(tmp_path):
    text = (_qc(tmp_path, [], [_candidate('CHEMBL2', ['ENSG2'])]) / 'qc.md').read_text()
    assert text.splitlines()[0].startswith('UNSCORED')
    assert 'no scorable rows' in text


def test_qc_scores_a_candidate_against_a_manual_row_of_its_salt(tmp_path):
    moa = [
        (['CHEMBL1SALT'], ['ENSG1'], 'manual'),
        (['CHEMBL1SALT2'], ['ENSG2'], 'manual'),
    ]
    candidates = [
        _candidate('CHEMBL1', ['ENSG2'], chembl_ids=['CHEMBL1', 'CHEMBL1SALT', 'CHEMBL1SALT2']),
        _candidate('CHEMBL1', ['ENSG9'], chembl_ids=['CHEMBL1', 'CHEMBL1SALT']),
    ]
    text = (_qc(tmp_path, moa, candidates) / 'qc.md').read_text()
    assert '1/2' in text
    assert 'scorable against a manual mechanism: 2' in text


def _snapshot(rows):
    return pl.DataFrame(
        rows,
        schema={
            'drugName': pl.String,
            'targetSymbolSuggestion': pl.List(pl.String),
            'actionType': pl.String,
        },
        orient='row',
    )


def test_diff_snapshots_reports_gained_and_lost_targets():
    old = _snapshot([
        ('aspirin', ['PTGS1'], 'INHIBITOR'),
        ('aspirin', ['PTGS2'], 'INHIBITOR'),
        ('gone', ['EGFR'], 'INHIBITOR'),
    ])
    new = _snapshot([
        ('aspirin', ['PTGS1'], 'INHIBITOR'),
        ('aspirin', ['B', 'A'], 'MODULATOR'),
        ('fresh', ['KIT'], 'INHIBITOR'),
    ])
    diff = qc.diff_snapshots(old, new).sort('drugName')
    assert diff.columns == ['drugName', 'gained', 'lost']
    assert diff.rows() == [
        ('aspirin', ['A,B|MODULATOR'], ['PTGS2|INHIBITOR']),
        ('fresh', ['KIT|INHIBITOR'], []),
        ('gone', [], ['EGFR|INHIBITOR']),
    ]


def test_diff_snapshots_ignores_unchanged_drugs():
    same = _snapshot([('aspirin', ['B', 'A'], 'INHIBITOR')])
    reordered = _snapshot([('aspirin', ['A', 'B'], 'INHIBITOR')])
    assert qc.diff_snapshots(same, reordered).is_empty()


def test_diff_snapshots_cli_writes_csv(tmp_path, capsys):
    for name, rows in {
        'old': [('a', ['X'], 'INHIBITOR')],
        'new': [('a', ['Y'], 'INHIBITOR'), ('a', ['W'], 'INHIBITOR'), ('b', ['Z'], 'INHIBITOR')],
    }.items():
        (tmp_path / name).mkdir()
        _snapshot(rows).write_parquet(tmp_path / name / 'data.parquet')
    cli.main([
        'diff-snapshots',
        '--old',
        str(tmp_path / 'old'),
        '--new',
        str(tmp_path / 'new'),
        '--out',
        str(tmp_path / 'o'),
    ])
    out = pl.read_csv(tmp_path / 'o' / 'snapshot_diff.csv').sort('drugName')
    assert out.rows() == [
        ('a', 'W|INHIBITOR;Y|INHIBITOR', 'X|INHIBITOR'),
        ('b', 'Z|INHIBITOR', ''),
    ]


def test_diff_snapshots_compares_null_targets_and_actions():
    old = _snapshot([('a', None, 'INHIBITOR'), ('b', ['X'], None)])
    new = _snapshot([('a', ['X'], 'INHIBITOR'), ('b', ['X'], 'INHIBITOR')])
    diff = qc.diff_snapshots(old, new)
    assert diff.rows() == [('a', ['X|INHIBITOR'], ['|INHIBITOR']), ('b', ['X|INHIBITOR'], ['X|'])]


def test_diff_snapshots_with_empty_snapshots():
    empty = _snapshot([])
    full = _snapshot([('a', ['X'], 'INHIBITOR')])
    assert qc.diff_snapshots(empty, empty).is_empty()
    assert qc.diff_snapshots(empty, full).rows() == [('a', ['X|INHIBITOR'], [])]
    assert qc.diff_snapshots(full, empty).rows() == [('a', [], ['X|INHIBITOR'])]
