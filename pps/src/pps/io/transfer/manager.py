"""Transfer module."""

import asyncio
from collections.abc import Sequence
from typing import cast

import tqdm
from loguru import logger

from pps.errors import PpsError, PpsErrorMessage
from pps.io.transfer import FTPtoGCPTransferableObject, PolarsDataFrameToGCSTransferableObject
from pps.io.transfer.model import TransferableObject


class TransferManager:
    """Manager class for handling the transfer of various transferable objects.

    This class provides static methods to `transfer` to move files or objects.
    Currently it supports:

        - FTP to Google Cloud Storage (GCP) transfers using `FTPtoGCPTransferableObject`.
        - Polars DataFrame to GCS transfers using `PolarsDataFrameToGCSTransferableObject`.

    """

    @staticmethod
    async def transfer_ftp_to_gcp(transferable_objects: Sequence[FTPtoGCPTransferableObject]) -> None:
        """Update GWAS Catalog metadata directly to cloud bucket.

        This method transfers files from FTP to Google Cloud Storage (GCS) using the provided
        FTPtoGCPTransferableObject instances.
        It fetches the data for the file provided in the local FTP path, collects the
        data asynchronously to buffer, and uploads it to the provided GCP bucket blob.

        Args:
            transferable_objects (Sequence[FTPtoGCPTransferableObject]): The objects to transfer.

        """
        # we always want to have the logs from this command uploaded to the target bucket
        transfer_tasks = [asyncio.create_task(x.transfer()) for x in transferable_objects]
        for f in tqdm.tqdm(asyncio.as_completed(transfer_tasks), total=len(transfer_tasks), desc='Downloading'):
            await f
        logger.info('gwas_curation_update step completed.')

    @staticmethod
    async def transfer_polars_to_gcs(transferable_objects: Sequence[PolarsDataFrameToGCSTransferableObject]) -> None:
        """Transfer Polars DataFrames to Google Cloud Storage.

        This method transfers Polars DataFrames to GCS using the provided
        PolarsDataFrameToGCSTransferableObject instances.

        Args:
            transferable_objects (Sequence[PolarsDataFrameToGCSTransferableObject]): The objects to transfer.

        """
        transfer_tasks = [asyncio.create_task(x.transfer()) for x in transferable_objects]
        for f in tqdm.tqdm(asyncio.as_completed(transfer_tasks), total=len(transfer_tasks), desc='Uploading'):
            await f
        logger.info('Polars DataFrame transfer to GCS completed.')

    async def transfer(self, transferable_objects: Sequence[TransferableObject]) -> None:
        """Transfer method that handles different types of transferable objects.

        Main method to manage the transfer of various transferable objects.

        Args:
            transferable_objects (Sequence[TransferableObject]): A sequence of TransferableObject instances.

        Raises:
            PpsError: If the list of transferable objects is empty or if the objects are not instances
                of the expected types.
        """
        if not transferable_objects:
            raise PpsError(PpsErrorMessage.EMPTY_TRANSFERABLE_OBJECTS)
        elif all(isinstance(c, FTPtoGCPTransferableObject) for c in transferable_objects):
            ftp_objects = cast(Sequence[FTPtoGCPTransferableObject], transferable_objects)
            await self.transfer_ftp_to_gcp(ftp_objects)
        elif all(isinstance(c, PolarsDataFrameToGCSTransferableObject) for c in transferable_objects):
            polars_objects = cast(Sequence[PolarsDataFrameToGCSTransferableObject], transferable_objects)
            await self.transfer_polars_to_gcs(polars_objects)
        else:
            raise PpsError(PpsErrorMessage.INVALID_TRANSFERABLE_OBJECTS)
