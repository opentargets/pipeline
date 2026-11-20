"""The run a release_uri names, and the values fixed by its product."""

import os
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

SCRATCH_BUCKET = 'gs://open-targets-pos-scratch'

# PRODUCTS in scratch mode, without FTP and AWS
SCRATCH_PRODUCTS: dict[str, dict[str, str]] = {
    'platform': {
        'release_gcs_bucket': f'{SCRATCH_BUCKET}/platform',
        'bigquery_project': 'open-targets-eu-dev',
        'bigquery_dataset': 'platform_scratch',
        'bigquery_staging_dataset': 'platform_scratch_staging',
        'bigquery_location': 'eu',
    },
    'ppp': {
        'release_gcs_bucket': f'{SCRATCH_BUCKET}/ppp',
    },
}

SCRATCH_VALUES = ('skip', 'only')

# steps that succeed doing nothing when no task runs for the product
OPTIONAL_STEPS = {'scratch_cleanup'}


@dataclass(frozen=True)
class Run:
    product: Literal['platform', 'ppp']
    release: str
    database_namespace: str
    release_uri: str

    @property
    def name(self) -> str:
        """The run segment of release_uri, e.g. `platform-2612-3`."""
        return self.release_uri.rsplit('/', 1)[-1]

    @property
    def prefix(self) -> str:
        """The release_uri segment before the run name, not the bucket; '' if none."""
        scheme, sep, rest = self.release_uri.partition('://')
        path = rest.partition('/')[2] if sep else scheme
        parts = [p for p in path.split('/') if p]
        return parts[-2] if len(parts) > 1 else ''

    def scratch_release_uri(self) -> str:
        """Where a scratch run writes in place of release_uri."""
        name = f'{self.prefix}-{self.name}' if self.prefix else self.name
        return f'{SCRATCH_BUCKET}/runs/{name}'

    def scratchpad(self, scratch: bool = False) -> dict[str, str]:
        """The scratchpad values derived from the run."""
        real = PRODUCTS[self.product]
        fixed = SCRATCH_PRODUCTS[self.product] if scratch else real
        return {
            'release': self.release,
            'database_namespace': self.database_namespace,
            'data_source': self.release_uri,
            'product': self.product,
            **fixed,
            # empty when the product is not published there; the real addresses in scratch mode too
            'croissant_ftp_address': self._output_at(real.get('ftp_address')),
            'croissant_aws_address': self._output_at(real.get('aws_address')),
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


def scratch_enabled() -> bool:
    """Whether `POS_SCRATCH=1` turns scratch mode on."""
    value = os.environ.get('POS_SCRATCH')
    if value and value != '1':
        raise ValueError(f'POS_SCRATCH must be 1 or unset, not {value!r}')
    return value == '1'


def for_product(specs: Sequence[Spec], product: str, step: str) -> list[Spec]:
    """The step's specs that run for the product: those without `products:`, or listing it.

    None for a step in OPTIONAL_STEPS is no error.
    """
    for s in specs:
        extra = s.model_extra or {}
        if unknown := set(extra.get('products', [])) - set(PRODUCTS):
            raise ValueError(f'{s.name}: unknown product {", ".join(sorted(unknown))}, products are {sorted(PRODUCTS)}')
        _refuse_gated_children(s)
    kept = [s for s in specs if product in (s.model_extra or {}).get('products', [product])]
    if specs and not kept and step not in OPTIONAL_STEPS:
        raise ValueError(f'step {step} does not run for {product}')
    _refuse_dropped_requirements(specs, kept, f'does not run for {product}')
    return kept


def for_mode(specs: Sequence[Spec], scratch: bool, step: str) -> list[Spec]:
    """The step's specs that run in the mode: `scratch: skip` drops a task in scratch mode, `only` outside it."""
    for s in specs:
        value = (s.model_extra or {}).get('scratch')
        if value is not None and value not in SCRATCH_VALUES:
            raise ValueError(f'{s.name}: unknown scratch: {value}, values are {list(SCRATCH_VALUES)}')
        _refuse_gated_children(s)
    mode = 'scratch' if scratch else 'normal'
    dropped_by = 'skip' if scratch else 'only'
    kept = [s for s in specs if (s.model_extra or {}).get('scratch') != dropped_by]
    if specs and not kept:
        raise ValueError(f'step {step} does not run in {mode} mode')
    _refuse_dropped_requirements(specs, kept, f'does not run in {mode} mode')
    return kept


def _refuse_dropped_requirements(specs: Sequence[Spec], kept: list[Spec], reason: str) -> None:
    dropped = {s.name for s in specs} - {s.name for s in kept}
    for s in kept:
        if missing := dropped.intersection(s.requires):
            raise ValueError(f'{s.name} requires {", ".join(sorted(missing))}, which {reason}')


def _refuse_gated_children(spec: Spec) -> None:
    if gated := _gated_children((spec.model_extra or {}).get('do', [])):
        raise ValueError(f'{spec.name}: only top-level tasks take products: or scratch:, not {", ".join(gated)}')


def _gated_children(children: list[dict]) -> list[str]:
    """Names of `do:` children, at any depth, that carry `products:` or `scratch:`."""
    gated = []
    for child in children:
        if 'products' in child or 'scratch' in child:
            gated.append(str(child.get('name')))
        gated.extend(_gated_children(child.get('do', [])))
    return gated
