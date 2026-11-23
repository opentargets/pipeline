"""Croissant distribution metadata for an Open Targets Platform data release."""

from mlcroissant import Context, FileObject, FileSet
from ot_datamodels.spec import DatasetSpec

from pos.croissant.schema import partition_columns


class PlatformOutputDistribution:
    """List of FileSets or FileObjects in the Open Targets Platform data."""

    distribution: list[FileSet | FileObject]
    contained_in: list[str]

    def __init__(self, ctx: Context | None = None) -> None:
        self.ctx = ctx or Context()
        self.distribution = []
        self.contained_in = []

    def get_metadata(self):
        """Return the distribution metadata."""
        return self.distribution

    def add_ftp_location(self, ftp_location: str | None, data_integrity_hash: str | None):
        """Add the FTP location of the distribution IF ftp location is not None.

        Args:
            ftp_location: The FTP location of the distribution.
            data_integrity_hash: The sha256 of the distribution, or None if it has none.

        Returns:
            The PlatformOutputDistribution object.
        """
        if ftp_location:
            self.distribution.append(
                FileObject(
                    ctx=self.ctx,
                    id='ftp-location',
                    name='FTP location',
                    description='FTP location of the Open Targets Platform data.',
                    encoding_formats=['application/x-ftp-directory'],
                    content_url=ftp_location,
                    sha256=data_integrity_hash,
                )
            )
            self.contained_in.append('ftp-location')

        return self

    def add_gcp_location(self, gcp_location: str, data_integrity_hash: str | None):
        """Add the GCP location of the distribution."""
        self.distribution.append(
            FileObject(
                ctx=self.ctx,
                id='gcp-location',
                name='GCP location',
                description=('Location of the Open Targets Platform data in Google Cloud Storage.'),
                encoding_formats=['application/x-gcp-directory'],
                content_url=gcp_location,
                sha256=data_integrity_hash,
            )
        )
        self.contained_in.append('gcp-location')
        return self

    def add_aws_location(self, aws_location: str | None, data_integrity_hash: str | None):
        """Add the AWS location of the distribution IF aws_location is not None.

        Args:
            aws_location: The AWS location of the distribution.
            data_integrity_hash: The sha256 of the distribution, or None if it has none.

        Returns:
            The PlatformOutputDistribution object.
        """
        if aws_location:
            self.distribution.append(
                FileObject(
                    ctx=self.ctx,
                    id='aws-location',
                    name='AWS location',
                    description=('Location of the Open Targets Platform data in Amazon Web Services.'),
                    encoding_formats=['application/x-aws-directory'],
                    content_url=aws_location,
                    sha256=data_integrity_hash,
                )
            )
            self.contained_in.append('aws-location')

        return self

    def add_datasets(self, datasets: list[tuple[str, DatasetSpec]]):
        """Add one file set per dataset folder, named by its model."""
        for path, spec in datasets:
            dataset_id = spec.name

            # Get columns the dataset is partitioned by:
            partitioned_by = partition_columns(path)

            # The includes depends on if the dataset has hyve partition:
            includes = [f'{dataset_id}/**/*.parquet'] if len(partitioned_by) > 0 else [f'{dataset_id}/*.parquet']

            # Description:
            description = f'Files containing all partitions of the {dataset_id} dataset'

            # If the dataset is partitioned by any field, add to description:
            if len(partitioned_by) > 0:
                description += f' partitioned by {",".join(partitioned_by)}'

            fileset = FileSet(
                id=dataset_id + '-fileset',
                name=spec.title,
                description=description,
                encoding_formats=['application/vnd.apache.parquet'],
                includes=includes,
            )

            if len(self.contained_in) > 0:
                # A new list per fileset: mlcroissant types this as list[str | Source],
                # which a list[str] is not assignable to.
                fileset.contained_in = [*self.contained_in]

            self.distribution.append(fileset)
        return self
