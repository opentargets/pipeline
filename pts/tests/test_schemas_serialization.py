import pandera.polars as pa
import polars as pl
from pandera.polars import PolarsData

from pts.schemas.serialization import schema_to_dict


class _Widget(pa.DataFrameModel):
    kind: str = pa.Field(isin=['a', 'b'], description='widget kind', metadata={'primary_key': True})
    tags: list[str] = pa.Field(description='widget tags')

    class Config:
        name = 'widget'
        metadata = {'owner': 'test'}

    @pa.check('tags')
    def tags_are_short(cls, data: PolarsData) -> pl.LazyFrame:  # noqa: N805 -- pandera's own @pa.check convention
        """Every tag must be under 5 characters."""
        return data.lazyframe.select(pl.col(data.key).list.eval(pl.element().str.len_chars() < 5).list.all())


def test_schema_to_dict_keeps_builtin_checks() -> None:
    dumped = schema_to_dict(_Widget)

    assert dumped['columns']['kind']['isin'] == ['a', 'b']


def test_schema_to_dict_describes_custom_checks_to_json_would_drop() -> None:
    dumped = schema_to_dict(_Widget)

    assert dumped['columns']['tags'].get('checks') is None
    assert dumped['columns']['tags']['custom_checks'] == [
        {'name': 'tags_are_short', 'description': 'Every tag must be under 5 characters.'}
    ]


def test_schema_to_dict_merges_column_and_dataframe_metadata() -> None:
    dumped = schema_to_dict(_Widget)

    assert dumped['columns']['kind']['metadata'] == {'primary_key': True}
    assert 'metadata' not in dumped['columns']['tags']
    assert dumped['metadata'] == {'owner': 'test'}
