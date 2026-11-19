"""The run a release_uri names, and the values fixed by its product."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

from otter.task.model import Spec

RUN_NAME = re.compile(r'^(?P<product>platform|ppp)-(?P<yy>\d{2})(?P<mm>\d{2})-\d+(-.+)?$')

PRODUCTS: dict[str, dict[str, str]] = {
    'platform': {
        'release_gcs_bucket': 'gs://open-targets-data-releases',
        'ftp_address': 'https://ftp.ebi.ac.uk/pub/databases/opentargets/platform',
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
        fixed = PRODUCTS[self.product]
        return {
            'release': self.release,
            'database_namespace': self.database_namespace,
            'data_source': self.release_uri,
            'product': self.product,
            **fixed,
            # empty when the product is not published there
            'croissant_ftp_address': self._output_at(fixed.get('ftp_address')),
            'croissant_aws_address': self._output_at(fixed.get('aws_address')),
        }

    def _output_at(self, base: str | None) -> str:
        return f'{base}/{self.release}/output/' if base else ''


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


def for_product(specs: Sequence[Spec], product: str, step: str) -> list[Spec]:
    """The step's specs that run for the product: those without `products:`, or listing it."""
    kept = [s for s in specs if product in (s.model_extra or {}).get('products', [product])]
    if specs and not kept:
        raise ValueError(f'step {step} does not run for {product}')
    dropped = {s.name for s in specs} - {s.name for s in kept}
    for s in kept:
        if missing := dropped.intersection(s.requires):
            raise ValueError(f'{s.name} requires {", ".join(sorted(missing))}, which does not run for {product}')
    return kept
