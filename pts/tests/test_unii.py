"""Tests for the FDA UNII annotation of ChEMBL molecules."""

import zipfile
from pathlib import Path

import polars as pl
import pytest

from pts.schemas.drug_molecule import cross_reference_schema, label_source_schema
from pts.transformers.utils.unii import (
    NAMES_MEMBER,
    RECORDS_MEMBER,
    annotate_molecules,
    drop_foreign_inn_synonyms,
    inn_substances,
    read_unii,
    unii_annotations,
)

RECORD_COLUMNS = ['UNII', 'DISPLAY_NAME', 'NCIT', 'INN_ID', 'INCHIKEY', 'SUBSTANCE_TYPE']


def records() -> pl.DataFrame:
    return pl.DataFrame(
        [
            ('U_DARA', 'DARAXONRASIB', 'C188048', '12345', 'KEY-DARA', 'chemical'),
            ('U_ADA', 'ADALIMUMAB', 'C65216', '8017', None, 'protein'),
            ('U_TWIN1', 'TWINOL', None, '1', 'KEY-TWIN', 'chemical'),
            ('U_TWIN2', 'TWINAMIDE', None, '2', 'KEY-TWIN', 'chemical'),
            ('U_NOINN', 'SOMETHING', None, None, 'KEY-NOINN', 'chemical'),
        ],
        schema=RECORD_COLUMNS,
        orient='row',
    )


def names() -> pl.DataFrame:
    return pl.DataFrame(
        [
            ('daraxonrasib [INN]', 'cn', 'U_DARA'),
            ('DARAXONRASIB [USAN]', 'cn', 'U_DARA'),
            ('Daraxonrasib', 'of', 'U_DARA'),
            ('RMC-6236', 'cd', 'U_DARA'),
            ('RMC 6236 [WHO-DD]', 'cn', 'U_DARA'),
            ('RM1', 'cd', 'U_DARA'),
            ('(1S,2S)-N-[...]-carboxamide', 'sys', 'U_DARA'),
            ('RASBRAND', 'bn', 'U_DARA'),
            ('adalimumab [INN]', 'cn', 'U_ADA'),
            ('ADALIMUMAB', 'of', 'U_ADA'),
            ('HUMIRA', 'bn', 'U_ADA'),
            ('twinol [INN]', 'cn', 'U_TWIN1'),
            ('twinamide [INN]', 'cn', 'U_TWIN2'),
            ('something [INN]', 'cn', 'U_NOINN'),
        ],
        schema=['NAME', 'TYPE', 'UNII'],
        orient='row',
    )


def _labels(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{'label': label, 'source': source} for label, source in pairs]


def molecules(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            'id': pl.String,
            'inchiKey': pl.String,
            'name': pl.String,
            'parentId': pl.String,
            'synonyms': label_source_schema,
            'tradeNames': label_source_schema,
            'crossReferences': cross_reference_schema,
        },
    )


def by_id(df: pl.DataFrame) -> dict[str, dict]:
    return {r['id']: r for r in df.to_dicts()}


class TestReadUnii:
    def test_reads_the_list_member_as_strings(self, tmp_path: Path) -> None:
        path = tmp_path / 'UNII_Data.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('UNII_Records_4Aug2026.txt', 'UNII\tINN_ID\tNOTE\nU1\t0012\t"quoted\n')
            archive.writestr('READ ME UNII Lists.txt', 'ignored')
        out = read_unii(str(path), RECORDS_MEMBER)
        assert out.rows() == [('U1', '0012', '"quoted')]
        assert out.schema == pl.Schema({'UNII': pl.String, 'INN_ID': pl.String, 'NOTE': pl.String})

    def test_a_zip_without_the_list_fails(self, tmp_path: Path) -> None:
        path = tmp_path / 'UNIIs.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('READ ME UNII Lists.txt', 'ignored')
        with pytest.raises(ValueError, match='expected one UNII_Names_'):
            read_unii(str(path), NAMES_MEMBER)


class TestInnSubstances:
    def test_inn_tagged_names_of_substances_with_an_inn_number(self) -> None:
        out = inn_substances(records(), names())
        assert set(out.rows()) == {
            ('U_DARA', 'daraxonrasib', 'KEY-DARA'),
            ('U_ADA', 'adalimumab', None),
            ('U_TWIN1', 'twinol', 'KEY-TWIN'),
            ('U_TWIN2', 'twinamide', 'KEY-TWIN'),
        }


class TestUniiAnnotations:
    def test_one_row_per_inchikey_of_a_single_inn_substance(self) -> None:
        out = unii_annotations(records(), names())
        # no structure (adalimumab) and a key shared by two INN substances (KEY-TWIN) are out
        assert out['inchiKey'].to_list() == ['KEY-DARA']

    def test_inn_first_then_official_names_and_codes_without_case_duplicates(self) -> None:
        (row,) = unii_annotations(records(), names()).to_dicts()
        assert row['inn'] == 'daraxonrasib'
        # `Daraxonrasib` (of) repeats the INN; `RM1` is too short; sys and cn names are left out
        assert row['synonyms'] == ['daraxonrasib', 'RMC-6236']
        assert row['tradeNames'] == ['RASBRAND']
        assert (row['unii'], row['ncit']) == ('U_DARA', 'C188048')


class TestAnnotateMolecules:
    @staticmethod
    def _annotate(rows: list[dict]) -> dict[str, dict]:
        return by_id(annotate_molecules(molecules(rows), unii_annotations(records(), names())))

    def test_an_unnamed_molecule_takes_the_inn_uppercased(self) -> None:
        out = self._annotate([{'id': 'CHEMBL1', 'inchiKey': 'KEY-DARA'}])
        assert out['CHEMBL1']['name'] == 'DARAXONRASIB'
        assert out['CHEMBL1']['synonyms'] == _labels(('RMC-6236', 'UNII'), ('daraxonrasib', 'UNII'))
        assert out['CHEMBL1']['tradeNames'] == _labels(('RASBRAND', 'UNII'))
        assert out['CHEMBL1']['crossReferences'] == [
            {'source': 'UNII', 'ids': ['U_DARA']},
            {'source': 'NCIt', 'ids': ['C188048']},
        ]

    def test_a_chembl_name_and_labels_stay_and_are_not_repeated(self) -> None:
        out = self._annotate([
            {
                'id': 'CHEMBL1',
                'inchiKey': 'KEY-DARA',
                'name': 'RMC-6236',
                'synonyms': _labels(('DARAXONRASIB', 'ChEMBL'), ('Daraxonrasib', 'ChEMBL')),
                'crossReferences': [{'source': 'drugbank', 'ids': ['DB1']}],
            }
        ])
        assert out['CHEMBL1']['name'] == 'RMC-6236'
        # both ChEMBL spellings stay; the UNII INN repeats one of them and is not added
        assert out['CHEMBL1']['synonyms'] == _labels(
            ('DARAXONRASIB', 'ChEMBL'), ('Daraxonrasib', 'ChEMBL'), ('RMC-6236', 'UNII')
        )
        assert [x['source'] for x in out['CHEMBL1']['crossReferences']] == ['drugbank', 'UNII', 'NCIt']

    def test_a_molecule_without_a_match_is_unchanged(self) -> None:
        out = self._annotate([
            {'id': 'CHEMBL2', 'inchiKey': 'KEY-OTHER', 'synonyms': _labels(('x', 'ChEMBL'))},
            {'id': 'CHEMBL3', 'inchiKey': None},
        ])
        assert out['CHEMBL2']['name'] is None
        assert out['CHEMBL2']['synonyms'] == _labels(('x', 'ChEMBL'))
        assert out['CHEMBL2']['crossReferences'] is None
        assert out['CHEMBL3']['crossReferences'] is None


class TestDropForeignInnSynonyms:
    @staticmethod
    def _drop(rows: list[dict]) -> dict[str, dict]:
        return by_id(drop_foreign_inn_synonyms(molecules(rows), inn_substances(records(), names())))

    def test_another_familys_inn_goes_and_other_synonyms_stay(self) -> None:
        out = self._drop([
            {'id': 'CHEMBL_ADA', 'name': 'ADALIMUMAB'},
            {
                'id': 'CHEMBL_GOLI',
                'name': 'GOLIMUMAB',
                'synonyms': _labels(('adalimumab', 'AACT'), ('cnto-148', 'AACT'), ('Golimumab', 'ChEMBL')),
            },
        ])
        assert out['CHEMBL_GOLI']['synonyms'] == _labels(('Golimumab', 'ChEMBL'), ('cnto-148', 'AACT'))

    def test_an_inn_owned_by_structure_goes(self) -> None:
        # the owner has no name at all: its structure is the INN substance's
        out = self._drop([
            {'id': 'CHEMBL_D', 'inchiKey': 'KEY-DARA'},
            {'id': 'CHEMBL_X', 'name': 'OTHER', 'synonyms': _labels(('daraxonrasib', 'AACT'))},
        ])
        assert out['CHEMBL_X']['synonyms'] == []

    def test_the_parents_inn_on_a_salt_stays(self) -> None:
        out = self._drop([
            {'id': 'CHEMBL_P', 'inchiKey': 'KEY-DARA', 'name': 'DARAXONRASIB'},
            {'id': 'CHEMBL_S', 'parentId': 'CHEMBL_P', 'synonyms': _labels(('daraxonrasib', 'AACT'))},
        ])
        assert out['CHEMBL_S']['synonyms'] == _labels(('daraxonrasib', 'AACT'))

    def test_an_inn_no_family_carries_stays(self) -> None:
        out = self._drop([{'id': 'CHEMBL_C', 'name': 'ABT-1', 'synonyms': _labels(('adalimumab', 'AACT'))}])
        assert out['CHEMBL_C']['synonyms'] == _labels(('adalimumab', 'AACT'))

    def test_a_curated_label_equal_to_another_familys_inn_stays(self) -> None:
        out = self._drop([
            {'id': 'CHEMBL_ADA', 'name': 'ADALIMUMAB'},
            {'id': 'CHEMBL_B', 'name': 'B', 'synonyms': _labels(('adalimumab', 'ChEMBL'))},
        ])
        assert out['CHEMBL_B']['synonyms'] == _labels(('adalimumab', 'ChEMBL'))
