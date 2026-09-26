"""Tests for pts.transformers.evidence.validation."""

import polars as pl

from pts.transformers.evidence import flags
from pts.transformers.evidence.validation import validate_datasource, validate_diseases, validate_target


class TestValidateDiseases:
    def test_resolves_known_disease_and_flags_unknown(self) -> None:
        df = pl.DataFrame({'diseaseFromSourceMappedId': ['EFO_1', 'EFO_MISSING']})
        disease_lut = pl.DataFrame({'diseaseFromSourceMappedId': ['EFO_1'], 'diseaseId': ['EFO_1']})

        result = validate_diseases(df, disease_lut)
        by_source_id = {r['diseaseFromSourceMappedId']: r for r in result.to_dicts()}

        assert by_source_id['EFO_1']['diseaseId'] == 'EFO_1'
        assert by_source_id['EFO_1']['qualityControls'] == []
        assert by_source_id['EFO_MISSING']['diseaseId'] is None
        assert by_source_id['EFO_MISSING']['qualityControls'] == [flags.INVALID_DISEASE]


class TestValidateTarget:
    def test_resolves_known_target_and_flags_unknown(self) -> None:
        df = pl.DataFrame({'targetFromSourceId': ['ENSG1', 'ENSG_MISSING']})
        target_lut = pl.DataFrame({
            'targetFromSourceId': ['ENSG1'],
            'targetId': ['ENSG1'],
            'biotype': ['protein_coding'],
        })

        result = validate_target(df, target_lut)
        by_source_id = {r['targetFromSourceId']: r for r in result.to_dicts()}

        assert by_source_id['ENSG1']['targetId'] == 'ENSG1'
        assert by_source_id['ENSG1']['qualityControls'] == []
        assert by_source_id['ENSG_MISSING']['targetId'] is None
        assert by_source_id['ENSG_MISSING']['qualityControls'] == [flags.INVALID_TARGET]
        assert 'biotype' not in result.columns

    def test_flags_excluded_biotype_without_dropping_the_row(self) -> None:
        df = pl.DataFrame({'targetFromSourceId': ['ENSG1']})
        target_lut = pl.DataFrame({
            'targetFromSourceId': ['ENSG1'],
            'targetId': ['ENSG1'],
            'biotype': ['pseudogene'],
        })

        result = validate_target(df, target_lut, excluded_biotypes=['pseudogene'])

        assert result.height == 1
        assert result.to_dicts()[0]['qualityControls'] == [flags.INVALID_BIOTYPE]

    def test_no_excluded_biotypes_never_flags(self) -> None:
        df = pl.DataFrame({'targetFromSourceId': ['ENSG1']})
        target_lut = pl.DataFrame({
            'targetFromSourceId': ['ENSG1'],
            'targetId': ['ENSG1'],
            'biotype': ['pseudogene'],
        })

        result = validate_target(df, target_lut)

        assert result.to_dicts()[0]['qualityControls'] == []


class TestValidateDatasource:
    def test_keeps_only_the_matching_datasource_and_drops_the_rest(self) -> None:
        df = pl.DataFrame({'datasourceId': ['encore', 'encore', 'other'], 'value': [1, 2, 3]})

        result = validate_datasource(df, 'encore')

        assert result['value'].to_list() == [1, 2]
