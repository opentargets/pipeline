import polars as pl

from pts.transformers.clinical_mira.moa import __main__ as cli


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
        'intermediate/drug_mechanism_of_action_aact',
        pl.DataFrame(
            candidates,
            schema={
                'parentChemblId': pl.String,
                'chemblIds': pl.List(pl.String),
                'targets': pl.List(pl.String),
                'confidence': pl.Float64,
                'maximumClinicalStage': pl.String,
                'drugType': pl.String,
            },
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


def _candidate(drug, genes, confidence=0.9, stage='PHASE_1', chembl_ids=None):
    return (drug, chembl_ids or [drug], genes, confidence, stage, 'Small molecule')


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
    assert '29/29' in _first_line(out)


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
