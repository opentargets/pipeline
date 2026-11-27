"""clickhouse_load_views adds one clickhouse_load per served view of the product."""

from unittest.mock import MagicMock

import pytest
from otter.scratchpad.model import Scratchpad
from otter.task.model import TaskContext

from pos.clickhouse_ddl import served_entries
from pos.tasks.clickhouse_load import ClickhouseLoadSpec
from pos.tasks.clickhouse_load_views import (
    ClickhouseLoadViews,
    ClickhouseLoadViewsError,
    ClickhouseLoadViewsSpec,
    view_loads,
)


def _spec(product: str, exclude: list[str]) -> ClickhouseLoadViewsSpec:
    return ClickhouseLoadViewsSpec(
        name='clickhouse_load_views load served views',
        requires=['clickhouse_create_database release database'],
        clickhouse_database='platform2612',
        data_dir_parent='release_data',
        product=product,
        exclude=exclude,
    )


@pytest.mark.parametrize('product', ['platform', 'ppp'])
def test_one_load_per_served_view_of_the_product(product):
    task = ClickhouseLoadViews(_spec(product, ['drug_view']), TaskContext(MagicMock(), Scratchpad({})))
    ClickhouseLoadViews.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]
    specs = [s for s in task.context.specs if isinstance(s, ClickhouseLoadSpec)]
    assert len(specs) == len(task.context.specs)
    assert [s.dataset for s in specs] == [n for n in served_entries(product) if n != 'drug_view']
    assert [s.name for s in specs] == [f'clickhouse_load clickhouse load {s.dataset}' for s in specs]
    for s in specs:
        assert (s.product, s.clickhouse_database, s.data_dir_parent) == (product, 'platform2612', 'release_data')
        assert s.dataset_config_path == 'config/datasets.yaml'
        assert s.requires == []


def test_a_ppp_only_view_loads_for_ppp_alone():
    assert 'association_timeseries_view' in view_loads('ppp', [])
    assert 'association_timeseries_view' not in view_loads('platform', [])


def test_excluding_a_view_that_is_not_served_fails_at_spec_time():
    with pytest.raises(ClickhouseLoadViewsError, match=r"not served: \['targets'\]"):
        ClickhouseLoadViews(_spec('platform', ['targets']), TaskContext(MagicMock(), Scratchpad({})))
