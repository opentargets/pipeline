import polars as pl
import pytest

from pts.transformers.clinical import map_entities
from tests.clinical.mapping_fixtures import (
    CHEMBL_CURATION,
    DISEASE,
    DRUG,
    REPORT_SCHEMA,
    REPORTS,
    STAGED_DRUG_ENTITIES,
)
from tests.clinical.mapping_ontoma_rows import MAPPED, MAPPED_COLUMNS


def _rows(df: pl.DataFrame) -> list[tuple]:
    return df.sort(df.columns, nulls_last=True).rows()


def _reports(*rows: tuple) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=REPORT_SCHEMA, orient='row')


def _staged(*rows: tuple[str, list[str]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=STAGED_DRUG_ENTITIES.schema, orient='row')


def _map(
    reports: pl.DataFrame, curation: pl.DataFrame | None = None, staged: pl.DataFrame = STAGED_DRUG_ENTITIES
) -> pl.DataFrame:
    return map_entities(reports, DISEASE, DRUG, curation, staged)


def _without_preset_fan_out(rows: list[tuple]) -> list[tuple]:
    """`rows` with each repeat of a row whose report had preset ids dropped."""
    preset = set(REPORTS.filter(pl.col('drugId').is_not_null() | pl.col('diseaseId').is_not_null())['id'])
    kept = [row for row in rows if row[0] not in preset]
    return sorted([*kept, *{row for row in rows if row[0] in preset}], key=lambda row: [(v is None, v) for v in row])


def test_map_entities_matches_ontoma() -> None:
    """Mira's rows, except that preset-id rows appear once, unlike mira."""
    mapped = _map(REPORTS, CHEMBL_CURATION)
    assert mapped.columns == MAPPED_COLUMNS
    expected = _without_preset_fan_out(MAPPED)
    assert len(expected) == len(MAPPED) - 3
    assert _rows(mapped) == expected


def test_preset_id_row_not_duplicated() -> None:
    reports = _reports(
        ('NCT05', 'Shared Brand', 'shared syndrome', None, None),
        ('CHEMBL_IND1', 'Shared Brand', 'shared syndrome', 'CHEMBL_PRESET', 'EFO_PRESET'),
        ('CHEMBL_IND2', 'Shared Brand', 'shared syndrome', 'CHEMBL_PRESET', None),
    )
    mapped = _map(reports)
    assert _rows(mapped.select('id', 'drugId', 'diseaseId')) == [
        ('CHEMBL_IND1', 'CHEMBL_PRESET', 'EFO_PRESET'),
        ('CHEMBL_IND2', 'CHEMBL_PRESET', 'EFO_0000001'),
        ('CHEMBL_IND2', 'CHEMBL_PRESET', 'EFO_0000002'),
        ('NCT05', 'CHEMBL2', 'EFO_0000001'),
        ('NCT05', 'CHEMBL2', 'EFO_0000002'),
        ('NCT05', 'CHEMBL3', 'EFO_0000001'),
        ('NCT05', 'CHEMBL3', 'EFO_0000002'),
    ]


def test_curation_and_preset_ids_win() -> None:
    reports = _reports(
        ('NCT01', 'Aspirin', 'Cystic fibrosis', None, None),
        ('NCT02', 'Ivacaftor 150 mg tablet', 'CF', None, None),
        ('NCT12', 'Aspirin', 'Cystic fibrosis', None, None),
        ('NCT14', 'Ivacaftor 150 mg tablet', 'CF', None, None),
        ('CHEMBL_IND1', 'Aspirin', 'Cystic fibrosis', 'CHEMBL_PRESET', 'EFO_PRESET'),
        ('CHEMBL_IND2', 'Ivacaftor 150 mg tablet', 'CF', 'CHEMBL_PRESET', 'EFO_PRESET'),
    )
    curation = pl.concat([
        CHEMBL_CURATION,
        pl.DataFrame(
            [('NCT14', 'Ivacaftor 150 mg tablet', 'CHEMBL_CURATED_NER', 'CF', 'EFO_CURATED_NER')],
            schema=CHEMBL_CURATION.schema,
            orient='row',
        ),
    ])
    mapped = _map(reports, curation)
    assert _rows(mapped.select('id', 'drugId', 'diseaseId')) == [
        ('CHEMBL_IND1', 'CHEMBL_PRESET', 'EFO_PRESET'),
        ('CHEMBL_IND2', 'CHEMBL_PRESET', 'EFO_PRESET'),
        ('NCT01', 'CHEMBL25', 'MONDO_0009061'),
        ('NCT02', 'CHEMBL2010601', 'MONDO_0009061'),
        ('NCT12', 'CHEMBL_CURATED', 'EFO_CURATED_3'),
        ('NCT14', 'CHEMBL_CURATED_NER', 'EFO_CURATED_NER'),
    ]


def test_staged_names_map_labels_without_an_id() -> None:
    staged = _staged(('Ivacaftor 150 mg tablet', ['ivacaftor']), ('aspirin tabs', ['kalydeco']))
    reports = _reports(
        ('NCT07', 'Ivacaftor 150 mg tablet', 'CF', None, None),
        ('NCT09', 'aspirin tabs', 'Cystic fibrosis', None, None),
    )
    mapped = _map(reports, staged=staged)
    assert _rows(mapped.select('id', 'drugId')) == [('NCT07', 'CHEMBL2010601'), ('NCT09', 'CHEMBL2010601')]


def test_unstaged_label_reaching_ner_fails() -> None:
    reports = _reports(
        ('NCT01', 'Aspirin', 'Cystic fibrosis', None, None),
        ('NCT14', 'Ivacaftor 150 mg tablet', 'CF', None, None),
        ('CHEMBL_IND1', 'preset pill', 'CF', 'CHEMBL_PRESET', None),
        ('NCT20', 'new pill a', 'CF', None, None),
        ('NCT21', 'new pill b', 'CF', None, None),
        ('NCT22', 'new pill a', 'Cystic fibrosis', None, None),
    )
    curation = pl.DataFrame(
        [('NCT14', 'Ivacaftor 150 mg tablet', 'CHEMBL_CURATED', 'CF', None)],
        schema=CHEMBL_CURATION.schema,
        orient='row',
    )

    with pytest.raises(ValueError, match=r"^2 drug labels not staged, e\.g\. \['new pill a', 'new pill b'\]$"):
        _map(reports, curation)
    mapped = _map(reports.filter(~pl.col('id').is_in(['NCT20', 'NCT21', 'NCT22'])), curation, _staged())
    assert _rows(mapped.select('id', 'drugId')) == [
        ('CHEMBL_IND1', 'CHEMBL_PRESET'),
        ('NCT01', 'CHEMBL25'),
        ('NCT14', 'CHEMBL_CURATED'),
    ]


def test_ner_ids_fill_drug_labels_only() -> None:
    reports = _reports(('NCT01', 'aspirin tabs', 'aspirin tabs', None, None))
    mapped = _map(reports)
    assert mapped.select('drugId', 'diseaseId').rows() == [('CHEMBL25', None)]


def test_all_mapped_skips_lookup() -> None:
    reports = _reports(('CHEMBL_IND1', 'Aspirin', None, 'CHEMBL_PRESET', None))
    mapped = _map(reports.drop('diseaseId'), staged=_staged())
    assert mapped.rows() == [('CHEMBL_IND1', 'Aspirin', None, 'CHEMBL_PRESET', None)]
    assert mapped.columns == MAPPED_COLUMNS
