"""Module for handling transfer operations in pps."""

from pps.io.transfer.ftp_to_gcs import FTPtoGCPTransferableObject
from pps.io.transfer.polars_to_gcs import PolarsDataFrameToGCSTransferableObject

__all__ = ['FTPtoGCPTransferableObject', 'PolarsDataFrameToGCSTransferableObject']
