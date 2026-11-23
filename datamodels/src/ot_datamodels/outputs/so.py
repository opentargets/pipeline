"""Sequence Ontology: the `so` output.

Generated from the 26.12 release output (2,620 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from ot_datamodels import Bioregistry, ClickHouseDictionary, ClickHouseTable, ClickHouseType, DatasetModel, PrimaryKey


class So(DatasetModel):
    """Sequence Ontology."""

    dataset_name: ClassVar[str] = 'so'
    dataset_title: ClassVar[str] = 'Sequence Ontology'
    dataset_description: ClassVar[str] = 'Sequence ontology lookup table'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Ontology',)
    clickhouse: ClassVar[tuple[ClickHouseTable | ClickHouseDictionary, ...]] = (
        ClickHouseTable('sequence_ontology_source', 'rocksdb', key='id'),
        ClickHouseDictionary('sequence_ontology', 'sequence_ontology_source', key='id', attributes=('label',)),
    )
    example: ClassVar[dict] = {'id': 'SO:0005854', 'label': 'gene_cassette_array'}

    id: Annotated[str, Bioregistry('so'), PrimaryKey(), ClickHouseType('LowCardinality(String)')] = Field(
        min_length=1, description='Sequence ontology term identifier'
    )
    label: Annotated[str, ClickHouseType('LowCardinality(String)')] = Field(
        min_length=1, description='Sequence ontology term label'
    )
