"""Test cases for the Crawl task."""

from datetime import date
from unittest.mock import Mock, patch

import pytest
from otter.task.model import State, TaskContext

from pps.errors import PpsError
from pps.gwas_catalog.release import GwasCatalogReleaseInfo
from pps.gwas_catalog.tasks.crawl import Crawl, CrawlSpec


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


@pytest.fixture
def crawl_spec():
    """Return a CrawlSpec instance for testing."""
    return CrawlSpec(
        name='test crawl task',
        stats_uri='https://www.ebi.ac.uk/gwas/api/search/stats',
        destination_template='gs://test-bucket/gwas/{release_date}/stats.json',
        promote=True,
    )


@pytest.fixture
def crawl_spec_no_promote():
    """Return a CrawlSpec instance without promotion for testing."""
    return CrawlSpec(
        name='test crawl task no promote',
        stats_uri='https://www.ebi.ac.uk/gwas/api/search/stats',
        destination_template='gs://test-bucket/gwas/{release_date}/stats.json',
        promote=False,
    )


@pytest.fixture
def mock_task_context():
    """Return a mock TaskContext."""
    context = Mock(spec=TaskContext)
    # Set up required attributes that the otter framework expects
    context.state = State.PENDING_RUN
    context.abort = Mock()
    context.abort.set = Mock()
    context.config = None
    return context


class TestCrawlSpec:
    """Test crawl specification."""

    def test_crawl_spec_initialization(self, crawl_spec):
        """Test CrawlSpec initialization with valid parameters."""
        assert crawl_spec.name == 'test crawl task'
        assert crawl_spec.stats_uri == 'https://www.ebi.ac.uk/gwas/api/search/stats'
        assert crawl_spec.destination_template == 'gs://test-bucket/gwas/{release_date}/stats.json'
        assert crawl_spec.promote is True

    def test_crawl_spec_requires_release_date_template(self):
        """Test that CrawlSpec validates destination_template contains release_date."""
        with pytest.raises(PpsError):
            CrawlSpec(
                name='test',
                stats_uri='https://example.com',
                destination_template='gs://test-bucket/gwas/stats.json',  # Missing {release_date}
                promote=True,
            )

    def test_crawl_spec_destinations_with_promote(self, crawl_spec):
        """Test destinations property when promote=True."""
        destinations = crawl_spec.destinations()
        assert len(destinations) == 2
        assert destinations[0].destination == 'gs://test-bucket/gwas/{release_date}/stats.json'
        assert destinations[0].is_substituted is False
        assert destinations[1].destination == 'gs://test-bucket/gwas/latest/stats.json'
        assert destinations[1].is_substituted is True

    def test_crawl_spec_destinations_without_promote(self, crawl_spec_no_promote):
        """Test destinations property when promote=False."""
        destinations = crawl_spec_no_promote.destinations()
        assert len(destinations) == 1
        assert destinations[0].destination == 'gs://test-bucket/gwas/{release_date}/stats.json'
        assert destinations[0].is_substituted is False

    def test_crawl_spec_substituted_destinations(self, crawl_spec, mock_gwas_catalog_release_info):
        """Test substituted_destinations method."""
        destinations = crawl_spec.substituted_destinations(mock_gwas_catalog_release_info)
        assert len(destinations) == 2
        assert 'gs://test-bucket/gwas/20231001/stats.json' in destinations
        assert 'gs://test-bucket/gwas/latest/stats.json' in destinations

    def test_crawl_task_initialization(self, crawl_spec, mock_task_context):
        """Test Crawl task initialization."""
        task = Crawl(crawl_spec, mock_task_context)
        assert task.spec == crawl_spec
        assert isinstance(task.spec, CrawlSpec)


class TestCrawlTask:
    """Test cases for the Crawl task."""

    @pytest.mark.asyncio
    @patch('pps.gwas_catalog.tasks.crawl.GwasCatalogReleaseInfo.from_uri')
    @patch('pps.gwas_catalog.tasks.crawl.StorageHandle')
    async def test_crawl_task_run_success(
        self,
        mock_storage_handle,
        mock_from_uri,
        crawl_spec,
        mock_task_context,
        mock_gwas_catalog_release_info,
    ):
        """Test successful execution of Crawl task."""
        mock_from_uri.return_value = mock_gwas_catalog_release_info

        # Create and run task
        task = Crawl(crawl_spec, mock_task_context)
        result = await task.run()

        # Assertions
        assert result == task  # Should return self
        mock_from_uri.assert_called_once_with('https://www.ebi.ac.uk/gwas/api/search/stats')

        # Verify storage writes: two destinations when promote=True
        assert mock_storage_handle.call_count == 2
        assert mock_storage_handle.return_value.write_text.call_count == 2

    @patch('pps.gwas_catalog.tasks.crawl.StorageHandle')
    def test_write_release_info(
        self,
        mock_storage_handle,
        crawl_spec,
        mock_task_context,
        mock_gwas_catalog_release_info,
    ):
        """Test _write_release_info method."""
        # Create task and call method
        task = Crawl(crawl_spec, mock_task_context)
        result = task._write_release_info(mock_gwas_catalog_release_info)

        # Assertions
        assert result == task  # Should return self
        destinations = [c.args[0] for c in mock_storage_handle.call_args_list]
        assert destinations == crawl_spec.substituted_destinations(mock_gwas_catalog_release_info)

        # Verify JSON content contains expected data
        written_content = mock_storage_handle.return_value.write_text.call_args[0][0]
        assert 'release_date' in written_content
        assert '2023-10-01' in written_content

    def test_gwas_catalog_release_info_from_fixture(self, mock_gwas_catalog_release_info):
        """Test that the fixture creates a valid GwasCatalogReleaseInfo instance."""
        assert isinstance(mock_gwas_catalog_release_info, GwasCatalogReleaseInfo)
        assert mock_gwas_catalog_release_info.release_date == date(2023, 10, 1)
        assert mock_gwas_catalog_release_info.number_of_associations == 150000
        assert mock_gwas_catalog_release_info.number_of_studies == 5000
        assert mock_gwas_catalog_release_info.number_of_sumstats == 3000
        assert mock_gwas_catalog_release_info.number_of_snps == 2000000
        assert mock_gwas_catalog_release_info.ensembl_build == '112.0'
        assert mock_gwas_catalog_release_info.dbsnp_build == '156'
        assert mock_gwas_catalog_release_info.efo_version == '3.60.0'
        assert mock_gwas_catalog_release_info.gene_build == 'GRCh38.p14'

    def test_gwas_catalog_release_info_strfmt(self, mock_gwas_catalog_release_info):
        """Test the strfmt method of GwasCatalogReleaseInfo."""
        assert mock_gwas_catalog_release_info.strfmt('%Y%m%d') == '20231001'
        assert mock_gwas_catalog_release_info.strfmt('%Y-%m-%d') == '2023-10-01'
