"""Generate one `clickhouse_load` per served view of the product: the models declare them (`pos.clickhouse_ddl`)."""

from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import OtterError

from pos.clickhouse_ddl import served_views
from pos.tasks.clickhouse_load import ClickhouseLoadSpec


class ClickhouseLoadViewsError(OtterError):
    """Base class for exceptions in this module."""


class ClickhouseLoadViewsSpec(Spec):
    """Configuration fields for the load served views task: those of `clickhouse_load` but `dataset`."""

    clickhouse_host: str = 'localhost'
    clickhouse_username: str = 'default'
    clickhouse_password: str = ''
    clickhouse_database: str = 'ot'
    clickhouse_port: int = 8123
    data_dir_parent: str
    product: str
    exclude: list[str] = []
    """Served views loaded elsewhere in the step."""


def view_loads(product: str, exclude: list[str]) -> list[str]:
    """The served views the task loads for `product`, by name; `exclude` must name served views.

    Raises:
        ClickhouseLoadViewsError: if `exclude` names a view that is not served.
    """
    if unknown := sorted(set(exclude) - set(served_views())):
        raise ClickhouseLoadViewsError(f'exclude names views that are not served: {unknown}')
    return [name for name in served_views(product) if name not in exclude]


def load_spec(spec: ClickhouseLoadViewsSpec, dataset: str) -> ClickhouseLoadSpec:
    """The `clickhouse_load` of one served view."""
    fields = spec.model_dump(include=set(ClickhouseLoadSpec.model_fields) - {'name', 'requires', 'dataset'})
    return ClickhouseLoadSpec(name=f'clickhouse_load clickhouse load {dataset}', dataset=dataset, **fields)


class ClickhouseLoadViews(Task):
    def __init__(self, spec: ClickhouseLoadViewsSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: ClickhouseLoadViewsSpec
        self._loads = view_loads(self.spec.product, self.spec.exclude)

    @report
    def run(self) -> Task:
        self.context.add_specs([load_spec(self.spec, name) for name in self._loads])
        return self
