"""Class to create the croissant recordset metadata for the Open Targets Platform."""

from __future__ import annotations

import mlcroissant as mlc
from loguru import logger
from ot_datamodels.spec import DatasetSpec, FieldSpec, Node

from pos.croissant.constants import TYPE_DICT

SNAKE_CASE_WARNING = "column '{field_name}' is snake_case, camelCase is expected"


def _warn_if_snake_case(field_name: str) -> None:
    """Warn if a field name uses snake_case instead of camelCase."""
    if '_' in field_name:
        logger.warning(SNAKE_CASE_WARNING.format(field_name=field_name))


def data_type(node: Node):
    """The Croissant data type of a primitive model value."""
    for dtype, croissant_type in TYPE_DICT.items():
        if isinstance(node.dtype, dtype):
            return croissant_type
    raise ValueError(f'no croissant data type for polars dtype {node.dtype}')


def dataset_description(spec: DatasetSpec) -> str:
    """The dataset's description, followed by its tags."""
    if not spec.tags:
        return spec.description
    return f'{spec.description} [{", ".join(spec.tags)}]'


class PlatformOutputRecordSets:
    """The record sets of the Open Targets Platform data: one per dataset, from its model."""

    record_sets: list[mlc.RecordSet]

    def __init__(self: PlatformOutputRecordSets) -> None:
        self.record_sets = []

    def get_metadata(self: PlatformOutputRecordSets) -> list[mlc.RecordSet]:
        """Return the record sets."""
        return self.record_sets

    def add_datasets(self: PlatformOutputRecordSets, specs: list[DatasetSpec]):
        """Add one record set per dataset spec."""
        for spec in specs:
            self.record_sets.append(self.get_fileset_recordset(spec))
        return self

    def get_fileset_recordset(self: PlatformOutputRecordSets, spec: DatasetSpec) -> mlc.RecordSet:
        """Returns the recordset for a dataset."""
        logger.info(f"processing '{spec.name}'")
        record_set = mlc.RecordSet(
            id=spec.name,
            name=spec.name,
            fields=[self.parse_field(spec.name, field) for field in spec.fields],
        )
        if spec.primary_key:
            record_set.key = [f'{spec.name}/{name}' for name in spec.primary_key]
        record_set.description = dataset_description(spec)
        return record_set

    def parse_field(
        self: PlatformOutputRecordSets,
        dataset: str,
        field: FieldSpec,
        parent: str | None = None,
    ) -> mlc.Field:
        """Parse a model field into a Croissant Field, recursing into lists and structs."""
        _warn_if_snake_case(field.name)
        column = f'{parent}/{field.name}' if parent else field.name
        field_id = f'{dataset}/{column}'
        node = field.node
        # Markers on a list's values sit on its element.
        value = node
        while value.element is not None:
            value = value.element

        if not field.description:
            raise ValueError(f'{field_id} has no description in its dataset model')
        description = field.description
        if value.bioregistry:
            description += f' [bioregistry:{value.bioregistry}]'

        croissant_field = mlc.Field(
            id=field_id,
            name=field.name,
            description=description,
            source=mlc.Source(
                file_set=f'{dataset}-fileset',
                extract=mlc.Extract(column=column),
            ),
        )
        croissant_field.data_types = []

        if value.reference is not None:
            croissant_field.references = mlc.Source(field=f'{value.reference.dataset}/{value.reference.column}')

        if node.kind == 'primitive':
            croissant_field.data_types.append(data_type(node))
        elif node.kind == 'struct':
            croissant_field.sub_fields = [self.parse_field(dataset, sub, column) for sub in node.fields]
        else:
            element = node.element
            assert element is not None
            croissant_field.repeated = True
            if element.kind == 'struct':
                croissant_field.sub_fields = [self.parse_field(dataset, sub, column) for sub in element.fields]
            elif element.kind == 'primitive':
                croissant_field.data_types.append(data_type(element))
            else:
                logger.warning(
                    f'field {field.name} is of type array of array, this is not yet supported by croissant, flattening'
                )
                assert element.element is not None
                if element.element.kind != 'primitive':
                    kind = element.element.kind
                    raise ValueError(f'{field_id} is a list of lists of {kind}, which croissant cannot type')
                croissant_field.data_types.append(data_type(element.element))

        return croissant_field
