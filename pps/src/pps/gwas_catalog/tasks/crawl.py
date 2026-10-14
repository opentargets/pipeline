"""Module to handle the crawling of GWAS Catalog release information."""

from typing import Annotated, Any, Self

from loguru import logger
from otter.storage.synchronous.handle import StorageHandle
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from pydantic import AfterValidator

from pps.gwas_catalog.release import GwasCatalogReleaseInfo, TemplateDestination, destination_validator


class CrawlSpec(Spec):
    """Configuration fields for the release crawler task.

    The `CrawlSpec` defines the parameters needed to crawl the GWAS Catalog release information.
    It includes the `stats_uri` that provides the release statistics and the `destination_prefix` where the
    release information will be stored.

    Examples:
    ---
    >>> cs = CrawlSpec(
    ...     name="crawl gwas catalog release information",
    ...     stats_uri="https://www.ebi.ac.uk/gwas/api/search/stats",
    ...     destination_template="gs://gwas_catalog_inputs/gentroutils/{release_date}/stats.json",
    ...     promote=True
    ... )
    >>> cs.name
    'crawl gwas catalog release information'
    >>> cs.stats_uri
    'https://www.ebi.ac.uk/gwas/api/search/stats'
    >>> cs.destination_template
    'gs://gwas_catalog_inputs/gentroutils/{release_date}/stats.json'
    >>> cs.promote
    True
    >>> rs = GwasCatalogReleaseInfo(
    ...     date="2023-10-01",
    ...     associations=1000,
    ...     studies=200,
    ...     sumstats=300,
    ...     snps=400,
    ...     ensemblbuild="114.0",
    ...     dbsnpbuild="1.0.0",
    ...     efoversion="1.0.0",
    ...     genebuild="GRCh38",
    ... )
    >>> cs.substituted_destinations(rs)
    ['gs://gwas_catalog_inputs/gentroutils/20231001/stats.json', 'gs://gwas_catalog_inputs/gentroutils/latest/stats.json']


    ### Example configuration for the crawl task in a YAML file.
    .. code-block:: yaml

        steps:
            - crawl gwas catalog release information:
                destination: gs://gwas_catalog_inputs/gentroutils/{release_date}/stats.json
                stats_uri: https://www.ebi.ac.uk/gwas/api/search/stats
                promote: true
    """

    name: str = 'crawl gwas catalog release information'
    """The name of the crawl task."""

    stats_uri: str = 'https://www.ebi.ac.uk/gwas/api/search/stats'
    """The URI to crawl the release statistics information from."""

    destination_template: Annotated[str, AfterValidator(destination_validator)]
    """The destination path to save the release information.
       This path should always be a template string that includes `{release_date}`.
       For example, `gs://gwas_catalog_inputs/gentroutils/{release_date}/stats.json`.

       The `release_date` will be substituted with the actual release date or `latest` literal
       from the stats_uri endpoint.
    """

    promote: bool = True
    """Whether to promote the release information as the latest release.

    Given the destination: `gs://gwas_catalog_inputs/gentroutils/{release_date}/stats.json`

       * If set to `False` the task will upload the release information
       only to the specified destination with `release_date` substituted by the value from the stats_uri endpoint.
       Resulting in following destinations:
            * `gs://gwas_catalog_inputs/gentroutils/20231001/stats.json`

       * If set to `True`, the task will also upload the release information
        to the destination with `release_date` substituted to `latest` literal, effectively
        promoting the release as the latest release.
    """

    def destinations(self) -> list[TemplateDestination]:
        """Get the list of destinations templates where the release information will be saved.

        Returns:
            list[TemplateDestination]: A list of TemplateDestination objects with the formatted destination paths.

        Depending on the `promote` flag this property will return:
            * If `promote` is `False`, it will return a single destination template.
            * If `promote` is `True`, it will return two destinations:
                1. The destination template with the release date substituted.
                2. The destination with the release date substituted to `latest`.
        """
        d1 = TemplateDestination(self.destination_template, False)
        if self.promote:
            d2 = d1.format({'release_date': 'latest'})
            return [d1, d2]
        return [d1]

    def substituted_destinations(self, release_info: GwasCatalogReleaseInfo) -> list[str]:
        """Safely parse the destination name to ensure it is valid."""
        substitutions = {'release_date': release_info.strfmt('%Y%m%d')}
        return [
            d.format(substitutions).destination if not d.is_substituted else d.destination for d in self.destinations()
        ]

    def model_post_init(self, __context: Any) -> None:
        """Method to ensure the scratchpad is set to ignore missing replacements."""
        self.scratchpad_ignore_missing = True


class Crawl(Task):
    """Task to crawl the GWAS Catalog release information."""

    def __init__(self, spec: CrawlSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: CrawlSpec

    def _write_release_info(self, release_info: GwasCatalogReleaseInfo) -> Self:
        """Write the release information to the specified GCP blob."""
        content = release_info.model_dump_json(indent=2, by_alias=False)
        destinations = self.spec.substituted_destinations(release_info)
        logger.info(f'Destinations for release information: {destinations}')
        for destination in destinations:
            assert 'gs://' in destination, f'Invalid GCS path in destination template: {destination}'
            StorageHandle(destination, config=self.context.config).write_text(content)
            logger.info(f'Release information written to {destination}')
        return self

    @report
    async def run(self) -> Self:
        """Crawl the release information."""
        logger.info(f'Crawling release information from {self.spec.stats_uri}')
        release_info = await GwasCatalogReleaseInfo.from_uri(self.spec.stats_uri)
        logger.info('Crawling completed successfully.')
        self._write_release_info(release_info)
        logger.info('Writing release information completed successfully.')
        return self
