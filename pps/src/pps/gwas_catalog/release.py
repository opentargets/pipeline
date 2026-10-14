"""GWAS Catalog release information and destination templates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import aiohttp
from loguru import logger
from pydantic import AliasPath, BaseModel, Field

from pps.errors import PpsError, PpsErrorMessage


class KeepMissing(defaultdict[str, str]):
    def __missing__(self, key):
        return '{' + key + '}'


def destination_validator(path: str) -> str:
    """Ensure that the destination path contains a template for the release date."""
    if '{release_date}' not in path:
        raise PpsError(PpsErrorMessage.MISSING_RELEASE_DATE_TEMPLATE, release_date='{release_date}')
    return path


@dataclass
class TemplateDestination:
    """A destination that can be formatted with a release date."""

    destination: str
    """The destination path that can be formatted with a release date."""
    is_substituted: bool = False
    """Whether the destination template has already been substituted."""

    def format(self, substitutions: dict[str, str]) -> TemplateDestination:
        """Format the destination with the given substitutions.

        This method returns a new TemplateDestination object (not a copy of the current one)
        with the formatted destination.
        """
        return TemplateDestination(self.destination.format_map(KeepMissing(**substitutions)), True)


class GwasCatalogReleaseInfo(BaseModel):
    """Model to hold GWAS Catalog release information."""

    release_date: date = Field(validation_alias=AliasPath('date'))
    """Release date of the GWAS Catalog."""

    number_of_associations: int = Field(validation_alias=AliasPath('associations'))
    """Number of associations in the GWAS Catalog."""

    number_of_studies: int = Field(validation_alias=AliasPath('studies'))
    """Number of studies in the GWAS Catalog."""

    number_of_sumstats: int = Field(validation_alias=AliasPath('sumstats'))
    """Number of summary statistics in the GWAS Catalog."""

    number_of_snps: int = Field(validation_alias=AliasPath('snps'))
    """Number of SNPs in the GWAS Catalog."""

    ensembl_build: str = Field(validation_alias=AliasPath('ensemblbuild'))
    """Ensembl version used in the GWAS Catalog."""

    dbsnp_build: str = Field(validation_alias=AliasPath('dbsnpbuild'))
    """dbSNP version used in the GWAS Catalog."""

    efo_version: str = Field(validation_alias=AliasPath('efoversion'))
    """EFO version used in the GWAS Catalog."""

    gene_build: str = Field(validation_alias=AliasPath('genebuild'))
    """Gene build version used in the GWAS Catalog."""

    def strfmt(self, format: str = '%Y%m%d') -> str:
        """Return a string representation of the release information."""
        return self.release_date.strftime(format)

    @staticmethod
    async def _get_release_info(uri: str) -> GwasCatalogReleaseInfo:
        """Get the release information from the specified URI."""
        headers = {'Accept': 'application/json'}
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(uri) as response:
                release_info = await response.json()
                return GwasCatalogReleaseInfo(**release_info)

    @classmethod
    async def from_uri(cls, uri: str) -> GwasCatalogReleaseInfo:
        """Fetch the release information from the specified URI."""
        logger.debug(f'Fetching release info from {uri}')
        try:
            return await cls._get_release_info(uri)
        except aiohttp.ClientError as e:
            logger.error(f'Error fetching release info: {e}')
            raise PpsError(PpsErrorMessage.FAILED_TO_FETCH, uri=uri)
