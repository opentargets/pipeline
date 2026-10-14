"""Module to handle the business logic for the GWAS Catalog curation task."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Self

from loguru import logger
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from pydantic import AfterValidator

from pps.gwas_catalog.curation import GWASCatalogCuration
from pps.gwas_catalog.release import TemplateDestination, destination_validator
from pps.io.transfer.manager import TransferManager
from pps.io.transfer.polars_to_gcs import PolarsDataFrameToGCSTransferableObject


class CurationSpec(Spec):
    """Configuration fields for the curation task.

    The `CurationSpec` defines the parameters needed to curate GWAS Catalog data.
    It includes the `previous_curation`, which is the path to the previous curation data,
    and the `studies`, which is the path to the studies data.

    Examples:
    ---
    >>> cs = CurationSpec(
    ...     name="curate gwas catalog data",
    ...     previous_curation="gs://gwas_catalog_inputs/curation/latest/curated/GWAS_Catalog_study_curation.tsv",
    ...     studies="gs://gwas_catalog_inputs/gentroutils/latest/gwas_catalog_download_studies.tsv",
    ...     summary_statistics_glob="gs://gwas_catalog_inputs/raw_summary_statistics/**/*.tsv.gz",
    ...     destination_template="gs://gwas_catalog_inputs/{release_date}/pending/curation.tsv"
    ... )
    >>> cs.name
    'curate gwas catalog data'
    >>> cs.previous_curation
    'gs://gwas_catalog_inputs/curation/latest/curated/GWAS_Catalog_study_curation.tsv'
    >>> cs.studies
    'gs://gwas_catalog_inputs/gentroutils/latest/gwas_catalog_download_studies.tsv'
    >>> cs.destination_template
    'gs://gwas_catalog_inputs/{release_date}/pending/curation.tsv'
    >>> cs.summary_statistics_glob
    'gs://gwas_catalog_inputs/raw_summary_statistics/**/*.tsv.gz'
    """

    name: str = 'curate gwas catalog data'
    """The name of the curation task."""

    previous_curation: str
    """The path to the previous curation data."""

    studies: str
    """The path to the studies data."""

    destination_template: Annotated[str, AfterValidator(destination_validator)]
    """The destination path for the curation data."""

    summary_statistics_glob: str
    """The glob pattern to locate summary statistics files."""

    promote: bool = False
    """Whether to promote the curation data to the latest version."""

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

    def substituted_destinations(self, release_date: date) -> list[str]:
        """Safely parse the destination name to ensure it is valid."""
        substitutions = {'release_date': release_date.strftime('%Y%m%d')}
        return [
            d.format(substitutions).destination if not d.is_substituted else d.destination for d in self.destinations()
        ]

    def model_post_init(self, __context: Any) -> None:
        """Method to ensure the scratchpad is set to ignore missing replacements."""
        self.scratchpad_ignore_missing = True


class Curation(Task):
    """Task for curating GWAS Catalog data."""

    def __init__(self, spec: CurationSpec, context: TaskContext) -> None:
        """Initialize the Curation task with the given specification and context."""
        super().__init__(spec, context)
        self.spec: CurationSpec

    @report
    async def run(self) -> Self:
        """Run the curation task."""
        logger.info('Starting curation task.')
        release_date = date.today()
        logger.debug(f'Using release date: {release_date}')
        destinations = self.spec.substituted_destinations(release_date)
        logger.debug(f'Destinations for curation data: {destinations}')
        curation = GWASCatalogCuration.from_prev_curation(
            self.spec.previous_curation, self.spec.studies, self.spec.summary_statistics_glob
        )
        logger.debug(f'Curation result preview:\n{curation.result.head()}')
        transfer_objects = [
            PolarsDataFrameToGCSTransferableObject(source=curation.result, destination=d) for d in destinations
        ]
        await TransferManager().transfer(transfer_objects)

        return self
