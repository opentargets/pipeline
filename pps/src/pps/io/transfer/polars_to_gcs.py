"""Module for transferring Polars DataFrames to Google Cloud Storage (GCS)."""

import polars as pl
from loguru import logger

from pps.io.transfer.model import TransferableObject


class PolarsDataFrameToGCSTransferableObject(TransferableObject):
    """A TransferableObject for transferring Polars DataFrames to Google Cloud Storage."""

    source: pl.DataFrame
    destination: str

    async def transfer(self) -> None:
        """Transfer the Polars DataFrame to the specified GCS destination."""
        # Convert Polars DataFrame to CSV and upload to GCS
        logger.info(f'Transferring Polars DataFrame to {self.destination}.')
        self.source.write_csv(self.destination, separator='\t', include_header=True)
        logger.info(f'Uploading DataFrame to {self.destination}')
