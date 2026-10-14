"""Transfer files from FTP to Google Cloud Storage (GCS)."""

import asyncio
import io
import re
from typing import Annotated

import aioftp
from google.cloud import storage
from loguru import logger
from pydantic import AfterValidator

from pps.io.path import FTPPath, GCSPath
from pps.io.transfer.model import TransferableObject


class FTPtoGCPTransferableObject(TransferableObject):
    """A class to represent an object that can be transferred from FTP to GCP."""

    source: Annotated[str, AfterValidator(lambda x: str(FTPPath(x)))]
    destination: Annotated[str, AfterValidator(lambda x: str(GCSPath(x)))]

    async def transfer(self) -> None:
        """Transfer files from FTP to GCP.

        This function fetches the data for the file provided in the local FTP path, collects the
        data asynchronously to buffer, and uploads it to the provided GCP bucket blob.

        Implements retry logic with exponential backoff for handling transient network errors.
        """
        max_retries = 3
        retry_delay = 1  # Initial delay in seconds

        for attempt in range(max_retries):
            try:
                await self._perform_transfer()
                return  # Success, exit the retry loop
            except (ConnectionResetError, OSError, aioftp.errors.AIOFTPException) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f'Transfer attempt {attempt + 1}/{max_retries} failed for {self.source}: {e}. '
                        f'Retrying in {wait_time}s...'
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f'Transfer failed after {max_retries} attempts for {self.source}: {e}')
                    raise
            except Exception as e:
                # For non-retryable exceptions, log and raise immediately
                logger.error(f'Non-retryable error during transfer from {self.source} to {self.destination}: {e}')
                raise

    async def _perform_transfer(self) -> None:
        """Perform the actual transfer operation.

        This is separated from the transfer method to allow for retry logic.
        """
        logger.info(f'Attempting to transfer data from {self.source} to {self.destination}.')
        gcs_obj = GCSPath(self.destination)
        ftp_obj = FTPPath(self.source)

        async with aioftp.Client.context(ftp_obj.server, user='anonymous', password='anonymous') as ftp:  # noqa: S106
            bucket = storage.Client().bucket(gcs_obj.bucket)
            blob = bucket.blob(gcs_obj.object)
            logger.info(f'Searching for the release date in the provided ftp path: {ftp_obj.base_dir}.')
            dir_match = re.match(r'^.*(?P<release_date>\d{4}\/\d{2}\/\d{2}){1}$', str(ftp_obj.base_dir))

            if dir_match:
                logger.info(f'Found release date to search in the ftp {dir_match.group("release_date")}.')
                release_date = dir_match.group('release_date')
                try:
                    logger.debug(f'We are in the directory: {await ftp.get_current_directory()}')
                    logger.debug(f'Changing directory to: {ftp_obj.base_dir}')
                    await ftp.change_directory(ftp_obj.base_dir)
                    logger.success(f'Successfully changed directory to: {ftp_obj.base_dir}')
                except aioftp.StatusCodeError as e:
                    logger.warning(f'Failed to change directory to {ftp_obj.base_dir}: {e}')
                    logger.warning(f'Probably the release date {release_date} is out of sync with the api endpoint.')
                    try:
                        logger.warning('Attempting to load the `latest` release.')
                        ftp_obj = FTPPath(self.source.replace(release_date, 'latest'))
                        await ftp.change_directory(ftp_obj.base_dir)
                        logger.success(f'Successfully changed directory to: {ftp_obj.base_dir}')

                    except aioftp.StatusCodeError as e:
                        logger.error(f'Failed to find the latest release under {ftp_obj}')
                        raise

                logger.debug('Creating in-memory buffer to store downloaded data.')
                buffer = io.BytesIO()
                logger.debug(f'Downloading data from FTP path: {ftp_obj.filename}')
                stream = await ftp.download_stream(ftp_obj.filename)
                logger.info('Successfully connected to the FTP stream, beginning data transfer to buffer.')
                # aioftp annotates __aexit__'s traceback as non-optional, which ty rejects
                async with stream:  # ty: ignore[invalid-context-manager]
                    async for block in stream.iter_by_block():
                        buffer.write(block)
                buffer.seek(0)
                if ftp_obj.filename.endswith('.zip'):
                    logger.info('Uploading zipped content to GCS blob.')
                    logger.info('Unzipping content before upload.')
                    content = unzip_buffer(buffer)
                    blob.upload_from_string(content)
                else:
                    content = buffer.getvalue()
                    buffer.close()
                    blob.upload_from_string(content)

            else:
                logger.error(f'Failed to extract release date from the provided ftp path: {ftp_obj.base_dir}.')
                raise ValueError('Release date could not be extracted from the FTP path.')


def unzip_buffer(buffer: io.BytesIO) -> bytes:
    """Unzip a BytesIO buffer and return a dictionary of file names to their content.

    Args:
        buffer (io.BytesIO): The in-memory buffer containing zipped data.

    Returns:
        bytes: The unzipped content of the single file.

    Raises:
        ValueError: If multiple files are found in the zipped buffer or if no files are found.
    """
    import zipfile

    unzipped_files: dict[str, bytes] = {}
    with zipfile.ZipFile(buffer) as z:
        for file_info in z.infolist():
            with z.open(file_info) as unzipped_file:
                unzipped_files[file_info.filename] = unzipped_file.read()

    if len(unzipped_files) == 0:
        logger.error('No files were found in the zipped buffer.')
        raise ValueError('No files were found in the zipped buffer.')
    if len(unzipped_files) != 1:
        logger.error('Multiple files were found in the zipped buffer.')
        raise ValueError('Multiple files were found in the zipped buffer.')
    keys = list(unzipped_files.keys())
    logger.info(f'Unzipped file: {keys[0]} with size {len(unzipped_files[keys[0]])} bytes.')

    return unzipped_files[keys[0]]
