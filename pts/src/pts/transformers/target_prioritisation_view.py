"""Target prioritisation view dataset generation: the served `target_prioritisation`.

Each `target_prioritisation` row as (name, value) string pairs, in ClickHouse `toString` form, then the
target's DepMap essentiality. Python per value: the input is one row per target.
"""

from __future__ import annotations

import math
from typing import Any, get_args

import polars as pl
from loguru import logger
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.views.target_prioritisation_view import PrioritisationKey, TargetPrioritisationView
from otter.config.model import Config

from pts.schemas.dataset.conform import conform
from pts.task_outputs import writes
from pts.transformers.utils.dataset import scan_dataset, write_dataset

ESSENTIALITY = 'geneEssentiality'
#: the target_prioritisation columns, in served order
COLUMNS = [key for key in get_args(PrioritisationKey) if key != ESSENTIALITY]
VIEW_SCHEMA = to_polars_schema(TargetPrioritisationView)


def ch_float(x: float) -> str:
    """`x` as ClickHouse `toString(Float64)` writes it.

    The shortest round-trip digits; fixed notation when 1e-6 <= |x| < 1e21, otherwise `d.ddde<exp>` with no `+`;
    no trailing `.0`; `-0`, `nan`, `inf`, `-inf`.
    """
    if math.isnan(x):
        return 'nan'
    if math.isinf(x):
        return 'inf' if x > 0 else '-inf'
    if x == 0:
        return '-0' if math.copysign(1.0, x) < 0 else '0'
    sign = '-' if x < 0 else ''
    mantissa, _, exponent = repr(abs(x)).partition('e')
    whole, _, fraction = mantissa.partition('.')
    padded = whole + fraction
    digits = padded.lstrip('0')
    # x = 0.<digits> * 10**point
    point = len(whole) + int(exponent or 0) - (len(padded) - len(digits))
    digits = digits.rstrip('0')
    k = len(digits)
    if k <= point <= 21:
        text = digits + '0' * (point - k)
    elif 0 < point <= 21:
        text = f'{digits[:point]}.{digits[point:]}'
    elif -6 < point <= 0:
        text = f'0.{"0" * -point}{digits}'
    else:
        text = f'{digits[0]}{"." + digits[1:] if k > 1 else ""}e{point - 1}'
    return sign + text


def ch_string(value: int | float) -> str:
    """A target_prioritisation value as ClickHouse `toString` writes it."""
    return ch_float(value) if isinstance(value, float) else str(value)


def build_target_prioritisation_view(
    target_prioritisation: pl.DataFrame, target_essentiality: pl.DataFrame
) -> pl.DataFrame:
    """One row per target_prioritisation row, in its order, conformed to `TargetPrioritisationView`.

    `items` holds the non-null values in `COLUMNS` order; when there is one, `geneEssentiality` follows for a target
    in `target_essentiality`: `-1` if essential, `0` if not.
    """
    essential = dict(target_essentiality.select('targetId', 'isEssential').iter_rows())
    rows = []
    for row in target_prioritisation.select('targetId', *COLUMNS).iter_rows(named=True):
        items = [{'key': key, 'value': ch_string(row[key])} for key in COLUMNS if row[key] is not None]
        if items and (is_essential := essential.get(row['targetId'])) is not None:
            items.append({'key': ESSENTIALITY, 'value': '-1' if is_essential else '0'})
        rows.append({'targetId': row['targetId'], 'items': items})
    return conform(pl.DataFrame(rows, schema=VIEW_SCHEMA), TargetPrioritisationView)


@writes(view=TargetPrioritisationView)
def target_prioritisation_view(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build the target prioritisation view and write it.

    Args:
        source: `target_prioritisation` and `target_essentiality` locations.
        destination: `view`, the target prioritisation view directory.
        settings: unused.
        config: unused.
    """
    logger.info('load target prioritisation view inputs')
    target_prioritisation, target_essentiality = pl.collect_all([
        scan_dataset(source['target_prioritisation']),
        scan_dataset(source['target_essentiality']).select('targetId', 'isEssential'),
    ])
    result = build_target_prioritisation_view(target_prioritisation, target_essentiality)
    logger.info(f'{result.height} targets in the target prioritisation view')
    write_dataset(result, str(destination['view']))
