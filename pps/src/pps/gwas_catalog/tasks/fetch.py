"""Module to handle the fetching of GWAS Catalog release files."""

from typing import Annotated, Any, Self

from loguru import logger
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from pydantic import AfterValidator

from pps.gwas_catalog.release import GwasCatalogReleaseInfo, TemplateDestination, destination_validator
from pps.io.transfer import FTPtoGCPTransferableObject
from pps.io.transfer.manager import TransferManager

MAX_CONCURRENT_CONNECTIONS = 10


class FetchSpec(Spec):
    """Configuration fields for the fetch task.

    The task downloads single file based on the `source_template` and uploads it to the `destination_template`.

    The `FetchSpec` defines the parameters needed to fetch the GWAS Catalog release files.
    These should be files that reside in the `https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/` directory.

    To make sure that we download the latest release and persist the release date,
    we need to make a single request to the `stats_uri` endpoint, which returns the latest release date.
    (We are not using the https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/` endpoint, rather
    the endpoint with `https://ftp.ebi.ac.uk/pub/databases/gwas/releases/{release_date}/` format to
    download the files.


    Examples:
    ---
    >>> fs = FetchSpec(
    ...     name="fetch associations",
    ...     stats_uri="https://www.ebi.ac.uk/gwas/api/search/stats",
    ...     source_template="ftp://ftp.ebi.ac.uk/pub/databases/gwas/releases/{release_date}/gwas-catalog-associations_ontology-annotated.tsv",
    ...     destination_template="gs://gwas_catalog_inputs/gentroutils/{release_date}/gwas_catalog_associations_ontology_annotated.tsv",
    ...     promote=True
    ... )
    >>> fs.name
    'fetch associations'
    >>> fs.stats_uri
    'https://www.ebi.ac.uk/gwas/api/search/stats'
    >>> fs.source_template
    'ftp://ftp.ebi.ac.uk/pub/databases/gwas/releases/{release_date}/gwas-catalog-associations_ontology-annotated.tsv'
    >>> fs.destination_template
    'gs://gwas_catalog_inputs/gentroutils/{release_date}/gwas_catalog_associations_ontology_annotated.tsv'
    >>> fs.promote
    True
    """

    name: str = 'fetch gwas catalog data'
    """The name of the task."""

    stats_uri: str = 'https://www.ebi.ac.uk/gwas/api/search/stats'
    """The URI to crawl the release statistics information from."""

    source_template: Annotated[str, AfterValidator(destination_validator)]
    """The template URI of the file to download."""

    destination_template: Annotated[str, AfterValidator(destination_validator)]
    """The template URI to upload the file to."""

    promote: bool = False
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

    def substituted_sources(self, release_info: GwasCatalogReleaseInfo) -> list[str]:
        """Safely parse the source name to ensure it is valid."""
        substitutions = {'release_date': release_info.strfmt('%Y/%m/%d')}
        if self.promote:
            return [self.source_template.format(**substitutions)] * 2
        return [self.source_template.format(**substitutions)]

    def model_post_init(self, __context: Any) -> None:
        """Method to ensure the scratchpad is set to ignore missing replacements."""
        self.scratchpad_ignore_missing = True


class Fetch(Task):
    """Task to fetch files from the GWAS Catalog release directory file."""

    def __init__(self, spec: FetchSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: FetchSpec

    @report
    async def run(self) -> Self:
        """Fetch the file from the remote to local."""
        logger.info(f'Fetching file from {self.spec.source_template}')
        release_info = await GwasCatalogReleaseInfo.from_uri(self.spec.stats_uri)
        logger.info(f'Release information: {release_info}')
        destinations = self.spec.substituted_destinations(release_info)
        sources = self.spec.substituted_sources(release_info)
        transferable_objects = [
            FTPtoGCPTransferableObject(source=s, destination=d) for s, d in zip(sources, destinations, strict=True)
        ]
        logger.info(f'Transferable objects: {transferable_objects}')
        await TransferManager().transfer(transferable_objects)
        logger.success('File transferred successfully.')
        return self
