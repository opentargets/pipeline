"""The run a release_uri names, and the values fixed by its product."""

import re
from dataclasses import dataclass
from typing import Literal, cast

RUN_NAME = re.compile(r'^(?P<product>platform|ppp)-(?P<yy>\d{2})(?P<mm>\d{2})-\d+(-.+)?$')

PRODUCTS: dict[str, dict[str, str]] = {
    'platform': {
        'release_gcs_bucket': 'gs://open-targets-data-releases',
        'aws_address': 's3://open-targets-public-data-releases/platform',
        'bigquery_project': 'open-targets-prod',
        'bigquery_dataset': 'platform',
        'bigquery_staging_dataset': 'platform_staging',
        'bigquery_location': 'eu',
    },
    'ppp': {
        'release_gcs_bucket': 'gs://open-targets-ppp-data-releases',
    },
}


@dataclass(frozen=True)
class Run:
    product: Literal['platform', 'ppp']
    release: str
    database_namespace: str
    release_uri: str

    def scratchpad(self) -> dict[str, str]:
        """The scratchpad values derived from the run."""
        return {
            'release': self.release,
            'database_namespace': self.database_namespace,
            'data_source': self.release_uri,
            'product': self.product,
            **PRODUCTS[self.product],
        }


def parse_release_uri(uri: str) -> Run:
    """Parse a release_uri whose last segment is `<product>-YYMM-N[-suffix]`."""
    release_uri = uri.rstrip('/')
    m = RUN_NAME.match(release_uri.rsplit('/', 1)[-1])
    if m is None:
        raise ValueError(f'release_uri {uri!r} must end in <product>-YYMM-N[-suffix], product in {sorted(PRODUCTS)}')
    product = cast("Literal['platform', 'ppp']", m['product'])
    return Run(
        product=product,
        release=f'{m["yy"]}.{m["mm"]}',
        database_namespace=f'{product}{m["yy"]}{m["mm"]}',
        release_uri=release_uri,
    )
