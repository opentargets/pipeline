"""Test cases for the Curation task."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest
from otter.task.model import State, TaskContext

from pps.errors import PpsError
from pps.gwas_catalog.tasks.curation import Curation, CurationSpec


class TestCurationSpec:
    """Test cases for the CurationSpec."""

    def test_curation_spec_initialization(self):
        """Test CurationSpec initialization."""
        curation_spec = CurationSpec(
            name='test curation',
            previous_curation='gs://test-bucket/previous_curation.json',
            studies='gs://test-bucket/studies.json',
            destination_template='gs://test-bucket/{release_date}/curation.json',
            summary_statistics_glob='gs://test-bucket/summary_statistics/*.txt',
            promote=True,
        )
        assert curation_spec.name == 'test curation'
        assert curation_spec.previous_curation == 'gs://test-bucket/previous_curation.json'
        assert curation_spec.studies == 'gs://test-bucket/studies.json'
        assert curation_spec.destination_template == 'gs://test-bucket/{release_date}/curation.json'
        assert curation_spec.promote is True
        destinations = curation_spec.destinations()
        assert destinations[0].destination == 'gs://test-bucket/{release_date}/curation.json'
        assert destinations[0].is_substituted is False
        assert destinations[1].destination == 'gs://test-bucket/latest/curation.json'
        assert destinations[1].is_substituted is True

    def test_curation_spec_requires_release_date_template(self):
        """Test that CurationSpec validates release date template."""
        with pytest.raises(PpsError, match='must contain a template for the release date'):
            CurationSpec(
                name='invalid_curation',
                previous_curation='gs://test-bucket/previous_curation.json',
                studies='gs://test-bucket/studies.json',
                destination_template='gs://test-bucket/curation.json',  # Missing {release_date}
                summary_statistics_glob='gs://test-bucket/summary_statistics/*.txt',
            )

    def test_curation_spec_substituted_destinations(self):
        """Test substituted destinations method."""
        curation_spec = CurationSpec(
            name='test curation',
            previous_curation='gs://test-bucket/previous_curation.json',
            studies='gs://test-bucket/studies.json',
            destination_template='gs://test-bucket/{release_date}/curation.json',
            summary_statistics_glob='gs://test-bucket/summary_statistics/*.txt',
            promote=True,
        )
        mock_release_info = MagicMock()
        mock_release_info.strftime = MagicMock(return_value='20231001')
        mock_release_info.date = '2023-10-01'
        substituted_destinations = curation_spec.substituted_destinations(mock_release_info)
        assert len(substituted_destinations) == 2
        assert 'gs://test-bucket/20231001/curation.json' in substituted_destinations
        assert 'gs://test-bucket/latest/curation.json' in substituted_destinations


class TestCurationTask:
    """Test cases for the Curation task."""

    @pytest.mark.asyncio
    @patch('pps.gwas_catalog.tasks.curation.date')
    @patch('pps.gwas_catalog.tasks.curation.GWASCatalogCuration')
    @patch('pps.gwas_catalog.tasks.curation.PolarsDataFrameToGCSTransferableObject')
    @patch('pps.gwas_catalog.tasks.curation.TransferManager')
    async def test_curation_run(
        self, mock_transfer_manager, mock_transferable_object, mock_gwas_catalog_curation, mock_date
    ):
        """Test Curation task run method with mocked dataframes."""
        # Setup mocks
        mock_today = date(2023, 10, 1)
        mock_date.today.return_value = mock_today

        # Mock the curation result dataframe
        mock_result_df = pl.DataFrame({
            'studyId': ['GCST001', 'GCST002'],
            'studyType': ['GWAS', 'GWAS'],
            'analysisFlag': ['', ''],
            'qualityControl': [None, None],
            'isCurated': [True, False],
            'pubmedId': [12345678, 87654321],
            'publicationTitle': ['Test Study 1', 'Test Study 2'],
            'traitFromSource': ['Trait 1', 'Trait 2'],
            'status': ['curated', 'new'],
        })

        # Mock GWASCatalogCuration instance and its result property
        mock_curation_instance = MagicMock()
        mock_curation_instance.result = mock_result_df
        mock_gwas_catalog_curation.from_prev_curation.return_value = mock_curation_instance

        # Mock transfer objects
        mock_transfer_obj1 = MagicMock()
        mock_transfer_obj2 = MagicMock()
        mock_transferable_object.side_effect = [mock_transfer_obj1, mock_transfer_obj2]

        # Mock transfer manager
        mock_transfer_manager_instance = MagicMock()
        mock_transfer_manager_instance.transfer = AsyncMock()
        mock_transfer_manager.return_value = mock_transfer_manager_instance

        # Create spec and task
        curation_spec = CurationSpec(
            name='test curation',
            previous_curation='gs://test-bucket/previous_curation.tsv',
            studies='gs://test-bucket/studies.tsv',
            destination_template='gs://test-bucket/{release_date}/curation.tsv',
            summary_statistics_glob='gs://test-bucket/summary_statistics/*.txt',
            promote=True,
        )

        mock_context = MagicMock(spec=TaskContext)
        # Set up required attributes that the otter framework expects
        mock_context.state = State.PENDING_RUN
        mock_context.abort = MagicMock()
        mock_context.abort.set = MagicMock()
        curation_task = Curation(curation_spec, mock_context)

        # Run the task
        result = await curation_task.run()

        # Assertions
        assert result == curation_task

        # Verify date.today was called
        mock_date.today.assert_called_once()

        # Verify GWASCatalogCuration.from_prev_curation was called correctly
        mock_gwas_catalog_curation.from_prev_curation.assert_called_once_with(
            'gs://test-bucket/previous_curation.tsv',
            'gs://test-bucket/studies.tsv',
            'gs://test-bucket/summary_statistics/*.txt',
        )

        # Verify substituted destinations are correct
        expected_destinations = ['gs://test-bucket/20231001/curation.tsv', 'gs://test-bucket/latest/curation.tsv']

        # Verify PolarsDataFrameToGCSTransferableObject was called for each destination
        assert mock_transferable_object.call_count == 2
        call_args = mock_transferable_object.call_args_list

        # Check first call
        assert call_args[0][1]['source'] is mock_result_df
        assert call_args[0][1]['destination'] == expected_destinations[0]

        # Check second call
        assert call_args[1][1]['source'] is mock_result_df
        assert call_args[1][1]['destination'] == expected_destinations[1]

        # Verify TransferManager was instantiated and transfer was called
        mock_transfer_manager.assert_called_once()
        mock_transfer_manager_instance.transfer.assert_called_once_with([mock_transfer_obj1, mock_transfer_obj2])

    @pytest.mark.asyncio
    @patch('pps.gwas_catalog.tasks.curation.date')
    @patch('pps.gwas_catalog.tasks.curation.GWASCatalogCuration')
    @patch('pps.gwas_catalog.tasks.curation.PolarsDataFrameToGCSTransferableObject')
    @patch('pps.gwas_catalog.tasks.curation.TransferManager')
    async def test_curation_run_without_promote(
        self, mock_transfer_manager, mock_transferable_object, mock_gwas_catalog_curation, mock_date
    ):
        """Test Curation task run method without promote flag."""
        # Setup mocks
        mock_today = date(2023, 10, 1)
        mock_date.today.return_value = mock_today

        # Mock the curation result dataframe
        mock_result_df = pl.DataFrame({
            'studyId': ['GCST001'],
            'studyType': ['GWAS'],
            'analysisFlag': [''],
            'qualityControl': [None],
            'isCurated': [True],
            'pubmedId': [12345678],
            'publicationTitle': ['Test Study'],
            'traitFromSource': ['Test Trait'],
            'status': ['curated'],
        })

        # Mock GWASCatalogCuration instance
        mock_curation_instance = MagicMock()
        mock_curation_instance.result = mock_result_df
        mock_gwas_catalog_curation.from_prev_curation.return_value = mock_curation_instance

        # Mock transfer objects
        mock_transfer_obj = MagicMock()
        mock_transferable_object.return_value = mock_transfer_obj

        # Mock transfer manager
        mock_transfer_manager_instance = MagicMock()
        mock_transfer_manager_instance.transfer = AsyncMock()
        mock_transfer_manager.return_value = mock_transfer_manager_instance

        # Create spec without promote
        curation_spec = CurationSpec(
            name='test curation',
            previous_curation='gs://test-bucket/previous_curation.tsv',
            studies='gs://test-bucket/studies.tsv',
            destination_template='gs://test-bucket/{release_date}/curation.tsv',
            summary_statistics_glob='gs://test-bucket/summary_statistics/*.txt',
            promote=False,
        )

        mock_context = MagicMock(spec=TaskContext)
        # Set up required attributes that the otter framework expects
        mock_context.state = State.PENDING_RUN
        mock_context.abort = MagicMock()
        mock_context.abort.set = MagicMock()
        curation_task = Curation(curation_spec, mock_context)

        # Run the task
        result = await curation_task.run()

        # Assertions
        assert result == curation_task

        # Verify only one destination (no promotion)
        assert mock_transferable_object.call_count == 1
        call_args = mock_transferable_object.call_args_list[0]
        assert call_args[1]['source'] is mock_result_df
        assert call_args[1]['destination'] == 'gs://test-bucket/20231001/curation.tsv'

        # Verify transfer was called with single object
        mock_transfer_manager_instance.transfer.assert_called_once_with([mock_transfer_obj])
