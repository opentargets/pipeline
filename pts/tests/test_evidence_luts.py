"""Tests for pts.transformers.evidence.luts."""

import polars as pl

from pts.transformers.evidence.luts import build_disease_lut, build_publication_lut, build_target_lut


def rows(df: pl.DataFrame) -> list[dict]:
    return sorted(df.to_dicts(), key=lambda r: str(r))


class TestBuildDiseaseLut:
    def test_maps_current_and_obsolete_ids_to_the_same_disease(self) -> None:
        disease_df = pl.DataFrame({
            'id': ['EFO_1', 'EFO_2'],
            'obsoleteTerms': [['EFO_OLD_1', 'EFO_OLD_1b'], None],
        })
        lut = build_disease_lut(disease_df)
        by_source_id = {r['diseaseFromSourceMappedId']: r['diseaseId'] for r in lut.to_dicts()}
        assert by_source_id == {
            'EFO_1': 'EFO_1',
            'EFO_OLD_1': 'EFO_1',
            'EFO_OLD_1b': 'EFO_1',
            'EFO_2': 'EFO_2',
        }

    def test_null_obsolete_terms_do_not_error(self) -> None:
        disease_df = pl.DataFrame(
            {'id': ['EFO_1'], 'obsoleteTerms': [None]}, schema={'id': pl.String, 'obsoleteTerms': pl.List(pl.String)}
        )
        lut = build_disease_lut(disease_df)
        assert lut.to_dicts() == [{'diseaseId': 'EFO_1', 'diseaseFromSourceMappedId': 'EFO_1'}]


class TestBuildTargetLut:
    def test_maps_ensembl_protein_and_symbol_ids_to_the_same_target(self) -> None:
        target_df = pl.DataFrame({
            'id': ['ENSG1'],
            'biotype': ['protein_coding'],
            'proteinIds': [[{'id': 'P100'}, {'id': 'P200'}]],
            'approvedSymbol': ['GENE1'],
        })
        lut = build_target_lut(target_df)
        by_source_id = {r['targetFromSourceId']: r['targetId'] for r in lut.to_dicts()}
        assert by_source_id == {'ENSG1': 'ENSG1', 'P100': 'ENSG1', 'P200': 'ENSG1', 'GENE1': 'ENSG1'}
        assert all(r['biotype'] == 'protein_coding' for r in lut.to_dicts())

    def test_null_protein_ids_do_not_error(self) -> None:
        target_df = pl.DataFrame(
            {'id': ['ENSG1'], 'biotype': ['lncRNA'], 'proteinIds': [None], 'approvedSymbol': ['GENE1']},
            schema={
                'id': pl.String,
                'biotype': pl.String,
                'proteinIds': pl.List(pl.Struct({'id': pl.String})),
                'approvedSymbol': pl.String,
            },
        )
        lut = build_target_lut(target_df)
        assert {r['targetFromSourceId'] for r in lut.to_dicts()} == {'ENSG1', 'GENE1'}


class TestBuildPublicationLut:
    def test_filters_to_accepted_sources_and_explodes_identifiers(self) -> None:
        literature_df = pl.DataFrame({
            'source': ['MED', 'PPR', 'OTHER'],
            'firstPublicationDate': ['2020-01-01', '2019-06-15', '2018-01-01'],
            'pmid': ['111', None, '999'],
            'id': ['MED-1', 'PPR-1', 'OTHER-1'],
            'pmcid': [None, 'PMC2', None],
        })
        lut = build_publication_lut(literature_df)
        by_pub_id = {r['publicationId']: r['publicationDate'] for r in lut.to_dicts()}
        assert by_pub_id == {'111': '2020-01-01', 'MED-1': '2020-01-01', 'PPR-1': '2019-06-15', 'PMC2': '2019-06-15'}
        assert 'OTHER-1' not in by_pub_id
