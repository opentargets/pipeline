"""How a view is served in ClickHouse, declared on its model as `clickhouse = Serving(...)`.

A served view is one table named after its `dataset_name`, built from the whole view. The rest is derived by pos:
the RocksDB key is the model's one `PrimaryKey`, the input folder `view/<dataset_name>`, the landing table
`<dataset_name>_log`, and the products the model's.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ClickHouseDictionary:
    """A dictionary over a served view's own table.

    Attributes:
        name: the dictionary's name.
        key: the lookup key column.
        attributes: the dictionary's value columns.
        source_columns: dictionary column to source column, where they differ.
    """

    name: str
    key: str
    attributes: tuple[str, ...]
    source_columns: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Serving:
    """How a view is served: its table's engine, a MergeTree's sort order, and the dictionaries built from it.

    Attributes:
        engine: `rocksdb` (point lookups on the model's one `PrimaryKey`) or `mergetree` (ordered by `order_by`).
        order_by: the MergeTree sort order, also its primary key; MergeTree only, and required there.
        dictionaries: the dictionaries built from the table.
    """

    engine: Literal['rocksdb', 'mergetree']
    order_by: tuple[str, ...] = ()
    dictionaries: tuple[ClickHouseDictionary, ...] = ()

    def __post_init__(self) -> None:
        if self.engine not in ('rocksdb', 'mergetree'):
            raise ValueError(f'unknown engine {self.engine!r}')
        if self.engine == 'mergetree' and not self.order_by:
            raise ValueError('a mergetree table needs an order_by')
        if self.engine == 'rocksdb' and self.order_by:
            raise ValueError('a rocksdb table takes no order_by')
