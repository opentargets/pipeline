"""Test polars to GCS transfer."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from pps.io.transfer import PolarsDataFrameToGCSTransferableObject


class TestPolarsDataFrameToGCSTransferableObject:
    @pytest.fixture
    def df(self) -> pl.DataFrame:
        """Fixture to create a sample Polars DataFrame."""
        return pl.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})

    def test_validation_success(self, df):
        """Test successful creation with valid DataFrame and destination."""
        obj = PolarsDataFrameToGCSTransferableObject(source=df, destination='gs://test-bucket/data.tsv')
        assert obj.source.equals(df)
        assert obj.destination == 'gs://test-bucket/data.tsv'

    @pytest.mark.asyncio
    async def test_transfer(self):
        """Test the transfer method with mocked DataFrame write_csv operation."""
        # Create a mock DataFrame
        mock_df = MagicMock(spec=pl.DataFrame)

        # Create the transferable object
        obj = PolarsDataFrameToGCSTransferableObject(source=mock_df, destination='gs://test-bucket/output.tsv')

        # Execute the transfer
        await obj.transfer()

        # Verify the write_csv method was called with correct destination
        mock_df.write_csv.assert_called_once_with('gs://test-bucket/output.tsv', separator='\t', include_header=True)

    @pytest.mark.asyncio
    async def test_transfer_current_implementation(self, df):
        """Test transfer with current implementation (direct write_csv call)."""
        with patch.object(df, 'write_csv') as mock_write_csv:
            obj = PolarsDataFrameToGCSTransferableObject(source=df, destination='gs://test-bucket/study_data.tsv')

            await obj.transfer()
            mock_write_csv.assert_called_once_with(
                'gs://test-bucket/study_data.tsv', separator='\t', include_header=True
            )
