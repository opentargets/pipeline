import polars as pl

from pts.transformers.clinical_mira.moa.quotes import check_mechanisms, pool_status
from pts.transformers.clinical_mira.moa.schemas import MoaExtraction

POOLS = pl.DataFrame({'pool_id': ['p1'], 'nct_ids': [['nct1', 'nct2']]})
TEXTS = pl.DataFrame({
    'id': ['nct1', 'nct2', 'nct9'],
    'prompt': [
        (
            'Trial ID: nct1\nDescription: sotorasib, a selective\nKRAS  G12C inhibitor, in NSCLC. '
            'Enrol patients who met the eligibility criteria.'
        ),
        'Trial ID: nct2\nDescription: patients with EGFR-mutant tumours receive a tyrosine kinase inhibitor.',
        'Trial ID: nct9\nDescription: another trial.',
    ],
})


def _mechanism(target: str, *quotes: tuple[str, str]) -> dict:
    return {
        'targetFromSource': target,
        'targetSymbolSuggestion': [],
        'isHumanTarget': True,
        'targetType': 'single protein',
        'actionType': 'INHIBITOR',
        'mechanismOfAction': f'{target} inhibitor',
        'evidence': [{'nctId': nct, 'quote': quote} for nct, quote in quotes],
        'confidence': 0.9,
    }


def _check(*mechanisms: dict, not_stated: bool = False) -> pl.DataFrame:
    extraction = MoaExtraction.model_validate({
        'drugName': 'sotorasib',
        'notStated': not_stated,
        'mechanisms': list(mechanisms),
    })
    return check_mechanisms({'p1': extraction}, POOLS, TEXTS)


def test_target_must_be_a_whole_word_of_the_quote():
    row = _check(_mechanism('MET', ('nct1', 'patients who met the eligibility criteria'))).row(0, named=True)

    assert row['kept'] is False
    assert row['drop_reason'] == 'target_not_in_quote'


def test_empty_target_is_dropped():
    row = _check(_mechanism('  ', ('nct1', 'a selective KRAS G12C inhibitor'))).row(0, named=True)

    assert row['kept'] is False
    assert row['drop_reason'] == 'empty_target'


def test_target_matches_across_whitespace_differences():
    row = _check(_mechanism('KRAS G12C', ('nct1', 'selective\nKRAS  G12C inhibitor'))).row(0, named=True)

    assert row['kept'] is True


def test_whitespace_differences_pass():
    row = _check(_mechanism('KRAS G12C', ('nct1', 'a selective KRAS G12C inhibitor'))).row(0, named=True)

    assert row['kept'] is True
    assert row['drop_reason'] is None


def test_case_or_wording_difference_fails():
    row = _check(_mechanism('KRAS G12C', ('nct1', 'A Selective KRAS G12C inhibitor'))).row(0, named=True)

    assert row['kept'] is False
    assert row['drop_reason'] == 'quote_not_in_text'


def test_quote_must_name_target():
    row = _check(_mechanism('EGFR', ('nct2', 'a tyrosine kinase inhibitor'))).row(0, named=True)

    assert row['drop_reason'] == 'target_not_in_quote'


def test_nct_outside_pool_dropped_even_if_text_elsewhere():
    row = _check(_mechanism('KRAS G12C', ('nct9', 'KRAS  G12C inhibitor'))).row(0, named=True)

    assert row['drop_reason'] == 'nct_not_in_pool'


def test_quote_must_come_from_the_cited_trial():
    row = _check(_mechanism('KRAS G12C', ('nct2', 'KRAS G12C inhibitor'))).row(0, named=True)

    assert row['drop_reason'] == 'quote_not_in_text'


def test_one_valid_quote_keeps_mechanism():
    row = _check(_mechanism('KRAS G12C', ('nct1', 'KRAS G12C inhibitor'), ('nct1', 'invented KRAS G12C text'))).row(
        0, named=True
    )

    assert row['kept'] is True
    assert row['evidence'] == [{'nctId': 'nct1', 'quote': 'KRAS G12C inhibitor'}]


def test_target_match_is_case_insensitive():
    row = _check(_mechanism('kras g12c', ('nct1', 'KRAS G12C inhibitor'))).row(0, named=True)

    assert row['kept'] is True


def test_not_stated_derived_from_kept():
    pools = pl.DataFrame({'pool_id': ['p1', 'p2']})
    mechanisms = _check(_mechanism('EGFR', ('nct2', 'a tyrosine kinase inhibitor')))
    extractions = {'p1': MoaExtraction(drug_name='sotorasib', not_stated=False, mechanisms=[])}

    status = pool_status(pools, extractions, mechanisms).sort('pool_id').to_dicts()

    assert status == [
        {'pool_id': 'p1', 'extracted': True, 'not_stated': True},
        {'pool_id': 'p2', 'extracted': False, 'not_stated': None},
    ]
