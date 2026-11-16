"""Tests for the polars lookup-table builders, checked against goldens on the shared fixtures."""

import polars as pl
import pytest
from loguru import logger

from pts.schemas.ontoma_lut import ENTITY_LUT, ENTITY_LUT_KEY, RAW_ENTITY_LUT
from pts.transformers.ontoma_lut.build import (
    acronym_key,
    build_lut,
    curation_label_lut,
    disease_id_lut,
    disease_label_lut,
    drug_label_lut,
    is_acronym_label,
    read_acronym_allowlist,
    resolve_acronym_allowlist,
    target_label_lut,
)
from pts.transformers.utils.disease_mapping import normalise as normalise_module
from tests.ontoma_lut_fixtures import CURATION, DISEASE, DRUG, TARGET
from tests.ontoma_lut_spark_rows import (
    CURATION_LABEL,
    DISEASE_ID,
    DISEASE_LABEL,
    DRUG_LABEL,
    LUT,
    TARGET_LABEL,
)


def test_disease_label_lut_matches_goldens() -> None:
    assert disease_label_lut(DISEASE).rows() == DISEASE_LABEL


def test_disease_id_lut_matches_goldens() -> None:
    assert disease_id_lut(DISEASE).rows() == DISEASE_ID


def test_curation_label_lut_matches_goldens() -> None:
    out = curation_label_lut(CURATION, 'ot_curation', DISEASE)
    assert out.rows() == CURATION_LABEL
    ids = {(label, entity_id) for entity_id, label, *_ in CURATION_LABEL}
    assert ('remapped by rank', 'MONDO_0000003') in ids
    assert ('obsolete label', 'MONDO_0000001') in ids
    assert ('label one', 'MONDO_0000010') in ids
    assert not {label for _, label in ids} & {'two-id label', 'no match', 'unicode word', 'carriage return', '1)'}


def test_target_label_lut_matches_goldens() -> None:
    assert target_label_lut(TARGET).rows() == TARGET_LABEL
    assert not {row for row in TARGET_LABEL if row[0] in {'ENSG02', 'ENSG03'}}


def test_drug_label_lut_matches_goldens() -> None:
    assert drug_label_lut(DRUG).rows() == DRUG_LABEL
    labels = {(entity_id, label) for entity_id, label, *_ in DRUG_LABEL}
    assert {('CHEMBL1', 'symkevi'), ('CHEMBL1', 'Trikafta'), ('CHEMBL4', 'ChEMBL4'), ('CHEMBL6', 'CHEMBL6')} <= labels
    assert not {entity_id for entity_id, _ in labels} & {'CHEMBL2', 'CHEMBL3', 'CHEMBL8'}


@pytest.mark.parametrize(
    ('build', 'frame'),
    [(disease_label_lut, DISEASE), (disease_id_lut, DISEASE), (target_label_lut, TARGET), (drug_label_lut, DRUG)],
)
def test_schema_is_raw_entity_lut(build, frame) -> None:
    assert build(frame).schema == RAW_ENTITY_LUT
    assert build(frame.clear()).schema == RAW_ENTITY_LUT


def test_curation_schema_is_raw_entity_lut() -> None:
    assert curation_label_lut(CURATION, 'eva_clinvar', DISEASE).schema == RAW_ENTITY_LUT
    assert curation_label_lut(CURATION.clear(), 'eva_clinvar', DISEASE).schema == RAW_ENTITY_LUT


def _raw(*rows: tuple) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=RAW_ENTITY_LUT, orient='row')


def _ids(lut: pl.DataFrame) -> dict[tuple, list[tuple]]:
    return {
        (kind, entity_type, label): [(i['entityId'], i['entitySource']) for i in ids]
        for kind, entity_type, label, ids in lut.rows()
    }


def test_empty_normalised_label_dropped() -> None:
    lut = build_lut([
        _raw(
            ('MONDO_1', 'the', 1.0, 'term', 'name', 'DS', 'label'),
            ('MONDO_2', '--', 1.0, 'symbol', 'name', 'DS', 'label'),
            ('MONDO_3', 'Cystic fibrosis', 1.0, 'term', 'name', 'DS', 'label'),
        )
    ])
    assert _ids(lut) == {('label', 'DS', 'cysticfibrosi'): [('MONDO_3', 'name')]}


def test_ties_kept_and_lower_scores_dropped() -> None:
    lut = build_lut([
        _raw(
            ('MONDO_2', 'Cystic fibrosis', 1.0, 'term', 'name', 'DS', 'label'),
            ('MONDO_3', 'cystic-fibrosis', 0.999, 'term', 'exact_synonym', 'DS', 'label'),
        ),
        _raw(('MONDO_1', 'fibrosis, cystic', 1.0, 'term', 'ot_curation', 'DS', 'label')),
    ])
    assert _ids(lut) == {('label', 'DS', 'cysticfibrosi'): [('MONDO_1', 'ot_curation'), ('MONDO_2', 'name')]}


def test_same_id_two_sources_two_structs() -> None:
    lut = build_lut([
        _raw(
            ('MONDO_1', 'Cystic fibrosis', 1.0, 'term', 'ot_curation', 'DS', 'label'),
            ('MONDO_1', 'Cystic fibrosis', 1.0, 'term', 'name', 'DS', 'label'),
            ('MONDO_1', 'cystic fibrosis', 1.0, 'term', 'name', 'DS', 'label'),
        )
    ])
    assert _ids(lut) == {('label', 'DS', 'cysticfibrosi'): [('MONDO_1', 'name'), ('MONDO_1', 'ot_curation')]}


def test_types_ranked_separately() -> None:
    lut = build_lut([
        _raw(
            ('MONDO_1', 'Foo', 1.0, 'term', 'name', 'DS', 'label'),
            ('ENSG01', 'FOO', 0.999, 'symbol', 'symbol_synonym', 'GP', 'label'),
            ('MONDO_2', 'FOO', 0.998, 'symbol', 'crossref', 'DS', 'id'),
        )
    ])
    assert _ids(lut) == {
        ('id', 'DS', 'foo'): [('MONDO_2', 'crossref')],
        ('label', 'DS', 'foo'): [('MONDO_1', 'name')],
        ('label', 'GP', 'foo'): [('ENSG01', 'symbol_synonym')],
    }


def test_null_scores_rank_last() -> None:
    lut = build_lut([
        _raw(
            ('MONDO_1', 'Foo', 0.5, 'term', 'name', 'DS', 'label'),
            ('MONDO_2', 'foo', None, 'term', 'name', 'DS', 'label'),
            ('ENSG01', 'Foo', None, 'term', 'name', 'GP', 'label'),
            ('ENSG02', 'foo', None, 'term', 'name', 'GP', 'label'),
        )
    ])
    assert _ids(lut) == {
        ('label', 'DS', 'foo'): [('MONDO_1', 'name')],
        ('label', 'GP', 'foo'): [('ENSG01', 'name'), ('ENSG02', 'name')],
    }


def test_each_label_normalised_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def normalise(label: str, track: str) -> str:
        calls.append((label, track))
        return label.lower()

    monkeypatch.setattr(normalise_module, 'normalise', normalise)
    build_lut([
        _raw(
            ('MONDO_1', 'Foo', 1.0, 'term', 'name', 'DS', 'label'), ('MONDO_2', 'Foo', 1.0, 'term', 'name', 'DS', 'id')
        ),
        _raw(
            ('ENSG01', 'Foo', 1.0, 'term', 'name', 'GP', 'label'),
            ('ENSG01', 'Foo', 1.0, 'symbol', 'name', 'GP', 'label'),
        ),
    ])
    assert sorted(calls) == [('Foo', 'symbol'), ('Foo', 'term')]


def test_lut_schema_and_order() -> None:
    raw = [disease_label_lut(DISEASE), disease_id_lut(DISEASE), target_label_lut(TARGET), drug_label_lut(DRUG)]
    lut = build_lut(raw)
    assert lut.schema == ENTITY_LUT
    assert lut.equals(lut.sort(list(ENTITY_LUT_KEY)))
    assert lut.select(list(ENTITY_LUT_KEY)).is_unique().all()
    assert lut['entityIds'].equals(lut['entityIds'].list.sort())
    assert build_lut([frame.clear() for frame in raw]).schema == ENTITY_LUT


def _raw_luts() -> list[pl.DataFrame]:
    return [
        disease_label_lut(DISEASE),
        disease_id_lut(DISEASE),
        curation_label_lut(CURATION, 'ot_curation', DISEASE),
        target_label_lut(TARGET),
        drug_label_lut(DRUG),
    ]


def test_build_lut_matches_goldens() -> None:
    assert len(LUT) > 50
    assert build_lut(_raw_luts()).rows() == LUT


def _flags(lut: pl.DataFrame, label: str, kind: str = 'label', entity_type: str = 'DS') -> dict[str, bool]:
    row = lut.filter(
        (pl.col('entityLabelNormalised') == label)
        & (pl.col('entityKind') == kind)
        & (pl.col('entityType') == entity_type)
    )
    assert row.height == 1
    return {i['entityId']: i['isAcronym'] for i in row['entityIds'][0]}


def _allowlist(*rows: tuple[str, str]) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema={'acronym': pl.String, 'diseaseId': pl.String}, orient='row')


@pytest.mark.parametrize(
    ('label', 'expected'),
    [
        ('CF', True),
        ('C.F.', True),
        ('vWD', True),
        ('CF1', True),
        ('ΑΣ', True),
        (' ALS ', True),
        ('CF 1', False),
        ('COPD', False),
        ('MASH', False),
        ('Cystic fibrosis', False),
        ('', False),
    ],
)
def test_is_acronym_label(label: str, expected: bool) -> None:
    assert pl.select(is_acronym_label(pl.lit(label))).item() == expected


def test_acronym_key() -> None:
    assert pl.select(acronym_key(pl.lit('C.F. 1-\u03a3'))).item() == 'cf1\u03c3'


def test_acronym_flag_per_id() -> None:
    lut = build_lut(
        [
            _raw(
                ('MONDO_1', 'CF', 0.999, 'symbol', 'exact_synonym', 'DS', 'label'),
                ('MONDO_1', 'Cystic fibrosis', 1.0, 'term', 'name', 'DS', 'label'),
            )
        ],
        acronym_allowlist=_allowlist(),
    )
    assert _flags(lut, 'cf') == {'MONDO_1': True}
    assert _flags(lut, 'cysticfibrosi') == {'MONDO_1': False}


def test_real_name_on_the_same_key_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(normalise_module, 'normalise', lambda label, track: 'key')
    lut = build_lut(
        [
            _raw(
                ('MONDO_1', 'CF', 0.999, 'term', 'exact_synonym', 'DS', 'label'),
                ('MONDO_1', 'Cee Eff', 0.999, 'term', 'exact_synonym', 'DS', 'label'),
            )
        ],
        acronym_allowlist=_allowlist(),
    )
    assert _flags(lut, 'key') == {'MONDO_1': False}


def test_lower_scored_real_name_does_not_unflag_the_acronym(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(normalise_module, 'normalise', lambda label, track: 'key')
    lut = build_lut(
        [
            _raw(
                ('MONDO_1', 'CF', 0.999, 'term', 'exact_synonym', 'DS', 'label'),
                ('MONDO_1', 'Cee Eff', 0.5, 'term', 'exact_synonym', 'DS', 'label'),
            )
        ],
        acronym_allowlist=_allowlist(),
    )
    assert _flags(lut, 'key') == {'MONDO_1': True}


def test_curated_labels_never_flagged() -> None:
    lut = build_lut(
        [_raw(('MONDO_1', 'AD', 1.0, 'symbol', 'ot_curation', 'DS', 'label'))], acronym_allowlist=_allowlist()
    )
    assert _flags(lut, 'ad') == {'MONDO_1': False}


def test_non_disease_rows_never_flagged() -> None:
    lut = build_lut(
        [
            _raw(
                ('ENSG1', 'TNF', 1.0, 'symbol', 'approved_symbol', 'GP', 'label'),
                ('CHEMBL1', 'ASA', 1.0, 'symbol', 'name', 'CD', 'label'),
            )
        ],
        acronym_allowlist=_allowlist(),
    )
    assert _flags(lut, 'tnf', entity_type='GP') == {'ENSG1': False}
    assert _flags(lut, 'asa', entity_type='CD') == {'CHEMBL1': False}


def test_allowlist_unflags_only_the_listed_id() -> None:
    lut = build_lut(
        [
            _raw(
                ('MONDO_0005109', 'HIV', 0.999, 'symbol', 'exact_synonym', 'DS', 'label'),
                ('MONDO_9', 'HIV', 0.999, 'symbol', 'exact_synonym', 'DS', 'label'),
            )
        ],
        acronym_allowlist=_allowlist(('hiv', 'MONDO_0005109')),
    )
    assert _flags(lut, 'hiv') == {'MONDO_0005109': False, 'MONDO_9': True}


def _both_flags(lut: pl.DataFrame, label: str) -> dict[str, tuple[bool, bool]]:
    row = lut.filter((pl.col('entityLabelNormalised') == label) & (pl.col('entityKind') == 'label'))
    assert row.height == 1
    return {i['entityId']: (i['isAcronym'], i['isAllowlistedAcronym']) for i in row['entityIds'][0]}


def test_allowlisted_acronym_flagged_as_allowlisted() -> None:
    lut = build_lut(
        [
            _raw(
                ('MONDO_0005109', 'HIV', 0.999, 'symbol', 'exact_synonym', 'DS', 'label'),
                ('MONDO_9', 'HIV', 0.999, 'symbol', 'exact_synonym', 'DS', 'label'),
            )
        ],
        acronym_allowlist=_allowlist(('hiv', 'MONDO_0005109')),
    )
    assert _both_flags(lut, 'hiv') == {'MONDO_0005109': (False, True), 'MONDO_9': (True, False)}


def test_packaged_allowlist_parses() -> None:
    allowlist = read_acronym_allowlist()
    assert allowlist.schema == pl.Schema({'acronym': pl.String, 'diseaseId': pl.String})
    assert allowlist.height > 0
    assert allowlist.is_duplicated().not_().all()
    assert allowlist.select(same=pl.col('acronym') == acronym_key(pl.col('acronym')))['same'].all()


@pytest.mark.parametrize(
    ('disease_id', 'resolved'),
    [
        ('MONDO_0000001', 'MONDO_0000001'),
        ('EFO_0000001', 'MONDO_0000001'),
        ('MONDO:0000002', 'MONDO_0000001'),
    ],
)
def test_allowlist_follows_obsolete_ids(disease_id: str, resolved: str) -> None:
    allowlist = resolve_acronym_allowlist(_allowlist(('hiv', disease_id)), DISEASE)
    assert allowlist.rows() == [('hiv', resolved)]


@pytest.mark.parametrize('disease_id', ['MONDO_9999999', 'Orphanet:100'])
def test_allowlist_drops_and_logs_ids_naming_no_single_disease(disease_id: str) -> None:
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level='WARNING')
    try:
        allowlist = resolve_acronym_allowlist(_allowlist(('hiv', disease_id)), DISEASE)
    finally:
        logger.remove(sink)
    assert allowlist.is_empty()
    assert any(f'hiv {disease_id}' in m for m in messages)


def test_obsolete_allowlist_id_unflags_current_id() -> None:
    lut = build_lut(
        [_raw(('MONDO_0000001', 'HIV', 0.999, 'symbol', 'exact_synonym', 'DS', 'label'))],
        acronym_allowlist=resolve_acronym_allowlist(_allowlist(('hiv', 'EFO_0000001')), DISEASE),
    )
    assert _flags(lut, 'hiv') == {'MONDO_0000001': False}
