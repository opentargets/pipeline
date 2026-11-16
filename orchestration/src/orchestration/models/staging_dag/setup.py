"""Setup of the staging DAGs: the files copied into the `bin/` folder of a run before the steps."""

from __future__ import annotations

import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SetupCopy(BaseModel):
    """A file copied from a URL into the `bin/` folder of a run, before the steps.

    Examples of such files are a cluster init action, or the main python file of a
    Dataproc job.

    Examples:
        >>> from pydantic import ValidationError
        >>> copy = SetupCopy.model_validate({
        ...     'from': 'https://raw.githubusercontent.com/opentargets/gentropy/v4.0.0-dev.1/src/gentropy/cli.py',
        ...     'to': 'gs://gwas_catalog_top_hits/prod/20261026T153052/bin/cli.py',
        ... })
        >>> copy.file_name
        'cli.py'

        The source is an http(s) URL:

        >>> try:
        ...     SetupCopy.model_validate({'from': 'gs://bucket/cli.py', 'to': 'gs://bucket/prod/20261026T153052/bin/cli.py'})
        ... except ValidationError as e:
        ...     print(e.errors()[0]['msg'])
        Value error, the setup source is not an http(s) URL: 'gs://bucket/cli.py'
    """

    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    source: str = Field(alias='from')
    """The URL of the file."""
    to: str
    """The destination, a file of the `bin/` folder of the run."""

    @model_validator(mode='after')
    def _validate(self) -> Self:
        if not re.fullmatch(r'https?://\S+', self.source):
            raise ValueError(f'the setup source is not an http(s) URL: {self.source!r}')
        return self

    @property
    def file_name(self) -> str:
        """The name of the file in `bin/`."""
        return self.to.rsplit('/', 1)[-1]
