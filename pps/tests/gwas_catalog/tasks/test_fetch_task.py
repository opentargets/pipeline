from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from otter.task.model import State, TaskContext

from pps.errors import PpsError
from pps.gwas_catalog.release import GwasCatalogReleaseInfo
from pps.gwas_catalog.tasks.fetch import Fetch, FetchSpec


@pytest.fixture
def mock_gwas_catalog_release_info_data():
    """Return a mock dict with GWAS Catalog release information using Field aliases."""
    return {
        'date': '2023-10-01',  # alias for release_date
        'associations': 150000,  # alias for number_of_associations
        'studies': 5000,  # alias for number_of_studies
        'sumstats': 3000,  # alias for number_of_sumstats
        'snps': 2000000,  # alias for number_of_snps
        'ensemblbuild': '112.0',  # alias for ensembl_build
        'dbsnpbuild': '156',  # alias for dbsnp_build
        'efoversion': '3.60.0',  # alias for efo_version
        'genebuild': 'GRCh38.p14',  # alias for gene_build
    }


@pytest.fixture
def mock_gwas_catalog_release_info(mock_gwas_catalog_release_info_data):
    """Return a GwasCatalogReleaseInfo instance with test data."""
    return GwasCatalogReleaseInfo(**mock_gwas_catalog_release_info_data)


class TestFetchSpec:
    """Test cases for the FetchSpec."""

    def test_initialization(self):
        """Test FetchSpec initialization."""
        fetch_spec = FetchSpec(
            name='test fetch',
            stats_uri='https://example.com/data.json',
            source_template='https://example.com/{release_date}/data.json',
            destination_template='gs://test-bucket/{release_date}/data.json',
            promote=True,
        )
        assert fetch_spec.name == 'test fetch'
        assert fetch_spec.source_template == 'https://example.com/{release_date}/data.json'
        assert fetch_spec.destination_template == 'gs://test-bucket/{release_date}/data.json'
        assert fetch_spec.promote is True
        destinations = fetch_spec.destinations()
        assert len(destinations) == 2
        assert destinations[0].destination == 'gs://test-bucket/{release_date}/data.json'
        assert destinations[0].is_substituted is False
        assert destinations[1].destination == 'gs://test-bucket/latest/data.json'
        assert destinations[1].is_substituted is True

    def test_initialization_no_promote(self):
        """Test FetchSpec initialization with promote = False."""
        fetch_spec = FetchSpec(
            name='test fetch',
            stats_uri='https://example.com/data.json',
            source_template='https://example.com/{release_date}/data.json',
            destination_template='gs://test-bucket/{release_date}/data.json',
            promote=False,
        )
        assert fetch_spec.name == 'test fetch'
        assert fetch_spec.source_template == 'https://example.com/{release_date}/data.json'
        assert fetch_spec.destination_template == 'gs://test-bucket/{release_date}/data.json'
        assert fetch_spec.promote is False

        destinations = fetch_spec.destinations()
        assert len(destinations) == 1
        assert destinations[0].destination == 'gs://test-bucket/{release_date}/data.json'
        assert destinations[0].is_substituted is False

    def test_requires_release_date_template(self):
        """Test that FetchSpec validates release date template."""
        with pytest.raises(PpsError, match='must contain a template for the release date'):
            FetchSpec(
                name='invalid_fetch',
                stats_uri='https://example.com/data.json',
                source_template='https://example.com/{release_date}/data.json',
                destination_template='gs://test-bucket/data.json',  # Missing {release_date}
            )
        with pytest.raises(PpsError, match='must contain a template for the release date'):
            FetchSpec(
                name='invalid_fetch',
                stats_uri='https://example.com/data.json',
                source_template='https://example.com/data.json',  # Missing {release_date}
                destination_template='gs://test-bucket/{release_date}/data.json',
            )

    def test_substituted_destinations(self):
        """Test substituted destinations method."""
        fetch_spec = FetchSpec(
            name='test fetch',
            stats_uri='https://example.com/data.json',
            source_template='https://example.com/{release_date}/data.json',
            destination_template='gs://test-bucket/{release_date}/data.json',
            promote=True,
        )
        mock_release_info = MagicMock()
        mock_release_info.strfmt = MagicMock(return_value='20231001')
        mock_release_info.date = '2023-10-01'
        substituted_destinations = fetch_spec.substituted_destinations(mock_release_info)
        assert len(substituted_destinations) == 2
        assert 'gs://test-bucket/20231001/data.json' in substituted_destinations
        assert 'gs://test-bucket/latest/data.json' in substituted_destinations

    def test_substituted_destinations_no_promote(self):
        """Test substituted destinations method with promote = False."""
        fetch_spec = FetchSpec(
            name='test fetch',
            stats_uri='https://example.com/data.json',
            source_template='https://example.com/{release_date}/data.json',
            destination_template='gs://test-bucket/{release_date}/data.json',
            promote=True,
        )
        mock_release_info = MagicMock()
        mock_release_info.strfmt = MagicMock(return_value='20231001')
        mock_release_info.date = '2023-10-01'
        substituted_destinations = fetch_spec.substituted_destinations(mock_release_info)
        assert len(substituted_destinations) == 2
        assert 'gs://test-bucket/20231001/data.json' in substituted_destinations
        assert 'gs://test-bucket/latest/data.json' in substituted_destinations

    def test_substituted_sources(self):
        """Test substituted sources method."""
        fetch_spec = FetchSpec(
            name='test fetch',
            stats_uri='https://example.com/data.json',
            source_template='https://example.com/{release_date}/data.json',
            destination_template='gs://test-bucket/{release_date}/data.json',
            promote=True,
        )
        mock_release_info = MagicMock()
        mock_release_info.strfmt = MagicMock(return_value='2023/10/01')
        substituted_sources = fetch_spec.substituted_sources(mock_release_info)
        assert len(substituted_sources) == 2
        assert all(i == 'https://example.com/2023/10/01/data.json' for i in substituted_sources)

    def test_substituted_sources_no_promote(self):
        """Test substituted sources method with promote = False."""
        fetch_spec = FetchSpec(
            name='test fetch',
            stats_uri='https://example.com/data.json',
            source_template='https://example.com/{release_date}/data.json',
            destination_template='gs://test-bucket/{release_date}/data.json',
            promote=False,
        )
        mock_release_info = MagicMock()
        mock_release_info.strfmt = MagicMock(return_value='2023/10/01')
        substituted_sources = fetch_spec.substituted_sources(mock_release_info)
        assert len(substituted_sources) == 1
        assert all(i == 'https://example.com/2023/10/01/data.json' for i in substituted_sources)


class TestFetchTask:
    """Test cases for the Fetch task."""

    @pytest.mark.asyncio
    @patch('pps.gwas_catalog.tasks.fetch.GwasCatalogReleaseInfo.from_uri')
    @patch('pps.gwas_catalog.tasks.fetch.TransferManager')
    async def test_fetch_run(self, mock_tf_manager, mock_from_uri, mock_gwas_catalog_release_info):
        fetch_spec = FetchSpec(
            name='test fetch',
            stats_uri='https://www.ebi.ac.uk/gwas/api/search/stats',
            source_template='ftp://example.com/{release_date}/data.json',
            destination_template='gs://test-bucket/{release_date}/data.json',
            promote=True,
        )

        # Ensure that GWASCatalogRelease.from_uri returns object with release info
        mock_from_uri.return_value = mock_gwas_catalog_release_info
        mock_tf_manager_instance = MagicMock()
        mock_tf_manager_instance_transfer = AsyncMock()
        mock_tf_manager.return_value = mock_tf_manager_instance
        mock_tf_manager_instance.transfer = mock_tf_manager_instance_transfer

        mock_context = MagicMock(spec=TaskContext)
        # Set up required attributes that the otter framework expects
        mock_context.state = State.PENDING_RUN
        mock_context.abort = MagicMock()
        mock_context.abort.set = MagicMock()

        # Create the task
        task = Fetch(fetch_spec, mock_context)

        # Run the task
        result = await task.run()

        # Assert the `GwasCatalogReleaseInfo.from_uri` was called once with the endpoint
        mock_from_uri.assert_called_once_with('https://www.ebi.ac.uk/gwas/api/search/stats')

        # Assert transfer was called once
        mock_tf_manager_instance.transfer.assert_called_once()

        # Get the transferable objects that were passed to transfer
        call_args = mock_tf_manager_instance.transfer.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0].source == 'ftp://example.com/2023/10/01/data.json'
        assert call_args[0].destination == 'gs://test-bucket/20231001/data.json'
        assert call_args[1].source == 'ftp://example.com/2023/10/01/data.json'
        assert call_args[1].destination == 'gs://test-bucket/latest/data.json'

        assert result == task  # Should return self
        assert isinstance(result, Fetch)
