"""Collection of tests for disease deduplication."""

from __future__ import annotations

import hashlib
import json
import os

import polars as pl
import pytest

from pts.transformers.disease import deduplicate_disease


class TestDiseaseDeduplication:
    """This module has test for disease deduplication."""

    @pytest.fixture(autouse=True)
    def _setup(self: TestDiseaseDeduplication) -> None:
        """Reading raw disease index."""
        script_path = os.path.dirname(os.path.realpath(__file__))
        self.disease_df = (
            pl.read_parquet(f'{script_path}/test_data/test_disease_w_duplication.parquet')
            .unnest('synonyms')
            .rename({
                'hasExactSynonym': 'exactSynonyms',
                'hasRelatedSynonym': 'relatedSynonyms',
                'hasNarrowSynonym': 'narrowSynonyms',
                'hasBroadSynonym': 'broadSynonyms',
            })
        )

        self.deduplicated = deduplicate_disease(self.disease_df)

    def test_return_type(self: TestDiseaseDeduplication) -> None:
        """Testing if the returned dataset has the right type."""
        assert isinstance(self.deduplicated, pl.DataFrame)

    def test_return_schema(self: TestDiseaseDeduplication) -> None:
        """Testing if the returned dataset has same schema."""
        schema1 = sorted(self.deduplicated.schema.items())
        schema2 = sorted(self.disease_df.schema.items())
        assert schema1 == schema2, f"Schemas don't match:\n{dict(schema1)}\nvs\n{dict(schema2)}"

    def test_number_of_rows(self: TestDiseaseDeduplication) -> None:
        """Testing if the returned dataset has fewer rows."""
        assert len(self.deduplicated) < len(self.disease_df)

    def test_deduplication_rows(self: TestDiseaseDeduplication) -> None:
        """Testing if the returned dataset has same schema."""
        # The initial dataset has duplication:
        assert len(self.disease_df.select(pl.col('name').str.to_lowercase()).unique()) != len(self.disease_df)

        # The updated data is deduplicated:
        assert len(self.deduplicated.select(pl.col('name').str.to_lowercase()).unique()) == len(self.deduplicated)

    def test_missing_id_in_obsoleted(self: TestDiseaseDeduplication) -> None:
        """Testing if the returned dataset has same schema."""
        all_diseases = self.disease_df['id'].to_list()
        dedup_diseases = self.deduplicated['id'].to_list()

        # Dropped disease - assert more than zero:
        dropped_diseases = [di for di in all_diseases if di not in dedup_diseases]
        assert len(dropped_diseases) > 0

    def test_dropped_id_in_obsoleted(self: TestDiseaseDeduplication) -> None:
        """Test if the dropped disease id is moved to the obsoleted column."""
        all_diseases = self.disease_df['id'].to_list()
        dedup_diseases = self.deduplicated['id'].to_list()

        # Dropped disease - assert more than zero:
        dropped_id = [di for di in all_diseases if di not in dedup_diseases][0]

        # Get the row merged, test if exists:
        current_disease = self.deduplicated.filter(pl.col('obsoleteTerms').list.contains(dropped_id))
        assert len(current_disease) == 1

    @pytest.mark.parametrize(
        'column_name',
        [
            ('parents'),
            ('children'),
            ('ancestors'),
            ('therapeuticAreas'),
            ('descendants'),
        ],
    )
    def test_columns_propagated(self: TestDiseaseDeduplication, column_name: str) -> None:
        """Test if the dropped disease id is moved to the obsoleted column."""
        all_diseases = self.disease_df['id'].to_list()
        dedup_diseases = self.deduplicated['id'].to_list()

        # Dropped disease - assert more than zero:
        dropped_id = [di for di in all_diseases if di not in dedup_diseases][0]

        # Get dropped value:
        depricated_disease_values = self.disease_df.filter(pl.col('id') == dropped_id).to_dicts()[0][column_name]

        # Get the row merged, test if exists:
        current_disease_values = (
            # Extract current disease
            self.deduplicated.filter(pl.col('obsoleteTerms').list.contains(dropped_id))
            # Convert to dictionary:
            .to_dicts()[0][column_name]
        )

        # Assert all values in the obsoleted values are propagated:
        for x in depricated_disease_values:
            assert x in current_disease_values

    def test_unique_disease_unchanged(self: TestDiseaseDeduplication) -> None:
        """Testing if a disease, which is not duplicated remain unchanged."""
        test_id = 'GO_0006954'

        hash_pre = self._get_md5hash(self.disease_df.filter(pl.col('id') == test_id))
        hash_post = self._get_md5hash(self.deduplicated.filter(pl.col('id') == test_id))

        assert hash_post == hash_pre

    @staticmethod
    def _get_md5hash(df: pl.DataFrame) -> str:
        """Calculate md5 sum of the first row of the provided dataframe.

        Args:
            df (pl.DataFrame): dataframe to test.

        Returns:
            str: md5 sum of the first row of the dataframe.
        """
        for col_name, dtype in df.schema.items():
            if isinstance(dtype, pl.List):
                # Serialize to JSON string
                df = df.with_columns(pl.col(col_name).list.sort())

        row_dict = df.to_dicts()[0]
        json_string = json.dumps(row_dict, sort_keys=True)
        return hashlib.md5(json_string.encode('utf-8')).hexdigest()
