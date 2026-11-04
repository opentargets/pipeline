"""Tests for the pharmacovigilance transformer."""

import json
import zipfile
from pathlib import Path

import polars as pl

from pts.schemas.pharmacovigilance import faers_schema
from pts.transformers.pharmacovigilance import (
    drug_names,
    normalise_event,
    read_archive,
    read_blacklist,
    report_rows,
)

ROW_COLUMNS = ['safetyreportid', 'chembl_id', 'event']


def _event(value: str) -> str:
    return pl.DataFrame({'e': [value]}).select(normalise_event(pl.col('e')))['e'][0]


def _drug(name: str = 'ASPIRIN', characterization: str | None = '1', **openfda) -> dict:
    drug: dict = {'drugcharacterization': characterization, 'medicinalproduct': name}
    if openfda:
        drug['openfda'] = openfda
    return drug


def _report(**overrides) -> dict:
    report: dict = {
        'safetyreportid': 'R1',
        'primarysource': {'qualification': '1'},
        'patient': {
            'reaction': [{'reactionmeddrapt': 'HEADACHE'}],
            'drug': [_drug()],
        },
    }
    report.update(overrides)
    return report


def _reports(*dicts: dict) -> pl.DataFrame:
    return pl.DataFrame({'results': [list(dicts)]}, schema=faers_schema).explode('results').unnest('results')


def _lookup(*pairs: tuple[str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        {'chembl_id': [p[0] for p in pairs], 'drug_name': [p[1] for p in pairs]},
        schema={'chembl_id': pl.String(), 'drug_name': pl.String()},
    )


ASPIRIN = _lookup(('CHEMBL1', 'aspirin'))
NO_BLACKLIST = pl.DataFrame(schema={'event': pl.String()})


def _rows(reports: pl.DataFrame, drugs: pl.DataFrame = ASPIRIN, blacklist: pl.DataFrame = NO_BLACKLIST) -> list[tuple]:
    return sorted(report_rows(reports, drugs, blacklist).select(ROW_COLUMNS).rows())


def _patient(reactions: list[str], drugs: list[dict]) -> dict:
    return {'reaction': [{'reactionmeddrapt': r} for r in reactions], 'drug': drugs}


def test_normalise_event_replaces_caret_and_strips_only_spaces():
    # non-breaking space survives: spark trim strips spaces only
    assert _event(' Crohn^S disease ') == "crohn's disease"
    assert _event('\u00a0headache ') == '\u00a0headache'


def test_drug_names_lowercases_and_covers_all_three_sources():
    label = pl.Struct({'label': pl.String(), 'source': pl.String()})
    molecules = pl.DataFrame(
        {
            'id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
            'name': ['Aspirin', 'Other', None],
            'tradeNames': [
                [{'label': 'Bayer', 'source': 'x'}, {'label': 'ASPIRIN', 'source': 'x'}],
                [],
                [{'label': 'Brand', 'source': 'x'}],
            ],
            'synonyms': [[{'label': 'Acetylsalicylic Acid', 'source': 'x'}], [], []],
        },
        schema={
            'id': pl.String(),
            'name': pl.String(),
            'tradeNames': pl.List(label),
            'synonyms': pl.List(label),
        },
    )
    result = drug_names(molecules)
    assert result.columns == ['chembl_id', 'drug_name']
    assert sorted(result.rows()) == [
        ('CHEMBL1', 'acetylsalicylic acid'),
        ('CHEMBL1', 'aspirin'),
        ('CHEMBL1', 'bayer'),
        ('CHEMBL2', 'other'),
        ('CHEMBL3', 'brand'),
    ]


def test_read_blacklist_normalises_lines():
    assert read_blacklist('  DEATH \nDrug Ineffective\n')['event'].to_list() == ['death', 'drug ineffective']


def test_rows_cross_product_of_reactions_and_drugs():
    patient = _patient(['HEADACHE', 'NAUSEA'], [_drug('ASPIRIN'), _drug('IBUPROFEN')])
    drugs = _lookup(('CHEMBL1', 'aspirin'), ('CHEMBL2', 'ibuprofen'))
    assert _rows(_reports(_report(patient=patient)), drugs) == [
        ('R1', 'CHEMBL1', 'headache'),
        ('R1', 'CHEMBL1', 'nausea'),
        ('R1', 'CHEMBL2', 'headache'),
        ('R1', 'CHEMBL2', 'nausea'),
    ]


def test_null_seriousnessdeath_is_kept_and_one_is_dropped():
    reports = _reports(
        _report(safetyreportid='NULL'),
        _report(safetyreportid='ZERO', seriousnessdeath='0'),
        _report(safetyreportid='DEAD', seriousnessdeath='1'),
    )
    assert [r[0] for r in _rows(reports)] == ['NULL', 'ZERO']


def test_qualification_4_and_null_are_dropped():
    reports = _reports(
        _report(safetyreportid='Q1'),
        _report(safetyreportid='Q3', primarysource={'qualification': '3'}),
        _report(safetyreportid='Q4', primarysource={'qualification': '4'}),
        _report(safetyreportid='QN', primarysource={'qualification': None}),
        _report(safetyreportid='NOSOURCE', primarysource=None),
    )
    assert [r[0] for r in _rows(reports)] == ['Q1', 'Q3']


def test_drugcharacterization_2_is_dropped():
    patient = _patient(['HEADACHE'], [_drug('ASPIRIN', '2'), _drug('IBUPROFEN', '1'), _drug('ASPIRIN', None)])
    drugs = _lookup(('CHEMBL1', 'aspirin'), ('CHEMBL2', 'ibuprofen'))
    assert _rows(_reports(_report(patient=patient)), drugs) == [('R1', 'CHEMBL2', 'headache')]


def test_blacklisted_event_is_dropped():
    patient = _patient(['DEATH', 'Headache '], [_drug()])
    blacklist = read_blacklist('Death\n')
    assert _rows(_reports(_report(patient=patient)), blacklist=blacklist) == [('R1', 'CHEMBL1', 'headache')]


def test_name_matches_through_brand_generic_and_substance():
    drugs = _lookup(('CHEMBL1', 'bayer'), ('CHEMBL2', 'acetylsalicylic acid'), ('CHEMBL3', 'salicylate'))
    patient = _patient(
        ['HEADACHE'],
        [
            _drug('UNKNOWN A', brand_name=['BAYER']),
            _drug('UNKNOWN B', generic_name=['Acetylsalicylic Acid']),
            _drug('UNKNOWN C', substance_name=['SALICYLATE']),
            _drug('UNKNOWN D', brand_name=['nothing']),
        ],
    )
    assert _rows(_reports(_report(patient=patient)), drugs) == [
        ('R1', 'CHEMBL1', 'headache'),
        ('R1', 'CHEMBL2', 'headache'),
        ('R1', 'CHEMBL3', 'headache'),
    ]


def test_empty_medicinalproduct_does_not_match_an_empty_name():
    drugs = _lookup(('CHEMBL1', ''), ('CHEMBL2', 'aspirin'))
    patient = _patient(['HEADACHE'], [_drug(''), _drug(None)])
    assert _rows(_reports(_report(patient=patient)), drugs) == []


def test_rows_are_distinct_when_two_names_map_to_one_drug():
    drugs = _lookup(('CHEMBL1', 'aspirin'), ('CHEMBL1', 'bayer'))
    patient = _patient(['HEADACHE'], [_drug('ASPIRIN', brand_name=['Bayer']), _drug('Bayer')])
    assert _rows(_reports(_report(patient=patient)), drugs) == [('R1', 'CHEMBL1', 'headache')]


def test_drug_without_openfda_block_still_links_by_medicinalproduct():
    patient = _patient(['HEADACHE'], [_drug('aspirin')])
    assert 'openfda' not in patient['drug'][0]
    assert _rows(_reports(_report(patient=patient))) == [('R1', 'CHEMBL1', 'headache')]


def test_null_or_empty_drug_and_reaction_lists_give_no_rows():
    reports = _reports(
        _report(safetyreportid='NODRUG', patient={'reaction': [{'reactionmeddrapt': 'X'}], 'drug': None}),
        _report(safetyreportid='EMPTYDRUG', patient={'reaction': [{'reactionmeddrapt': 'X'}], 'drug': []}),
        _report(safetyreportid='NOREACTION', patient={'reaction': None, 'drug': [_drug()]}),
        _report(safetyreportid='EMPTYREACTION', patient={'reaction': [], 'drug': [_drug()]}),
        _report(safetyreportid='NOPATIENT', patient=None),
    )
    assert _rows(reports) == []


def test_empty_results_gives_empty_frame_with_the_output_columns():
    result = report_rows(_reports(), ASPIRIN, NO_BLACKLIST)
    assert result.columns == ROW_COLUMNS
    assert result.height == 0


def _write_zip(path: Path, results: list[dict]) -> str:
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('events.json', json.dumps({'meta': {'last_updated': '2026-01-01'}, 'results': results}))
    return str(path)


def test_read_archive_ignores_reportduplicate_in_either_shape(tmp_path):
    as_list = _report(safetyreportid='LIST', reportduplicate=[{'duplicatenumb': '1', 'duplicatesource': 'a'}])
    as_struct = _report(safetyreportid='STRUCT', reportduplicate={'duplicatenumb': '2', 'duplicatesource': 'b'})
    first = _write_zip(tmp_path / 'first.zip', [as_list])
    second = _write_zip(tmp_path / 'second.zip', [as_struct])
    assert sorted(read_archive(first, ASPIRIN, NO_BLACKLIST).rows()) == [('LIST', 'CHEMBL1', 'headache')]
    assert sorted(read_archive(second, ASPIRIN, NO_BLACKLIST).rows()) == [('STRUCT', 'CHEMBL1', 'headache')]
