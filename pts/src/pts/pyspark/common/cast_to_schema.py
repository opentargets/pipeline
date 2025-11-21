"""Methods for handling schemas."""

from __future__ import annotations

from loguru import logger
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import ArrayType, DataType, StructType


def harmonise_to_schema(df: DataFrame, target_schema: StructType) -> DataFrame:
    """Cast DataFrame columns to match the target schema.

    Args:
        df: The DataFrame to cast
        target_schema: The target schema to cast to

    Returns:
        DataFrame with columns cast to match the target schema
    """
    schema_fields = {field.name: field for field in target_schema.fields}

    for col_name in df.columns:
        if col_name not in schema_fields:
            logger.warning(f'Column {col_name} not found in target schema')
            continue

        target_field = schema_fields[col_name]
        source_field = df.schema[col_name]

        if source_field.dataType == target_field.dataType:
            continue

        df = df.withColumn(
            col_name,
            cast_column_to_target_type(
                f.col(col_name),
                source_field.dataType,
                target_field.dataType,
            ),
        )

    return df


def cast_column_to_target_type(
    col: Column,
    source_type: DataType,
    target_type: DataType,
) -> Column:
    """Cast a column to match the target type.

    Handles arrays, and nested structs recursively.

    Args:
        col: The column expression to cast
        source_type: The source data type
        target_type: The target data type

    Returns:
        The cast column expression
    """
    if isinstance(source_type, ArrayType) and isinstance(target_type, ArrayType):
        if isinstance(target_type.elementType, StructType):
            # Array of structs
            return f.transform(
                col,
                lambda x: cast_struct(x, source_type.elementType, target_type.elementType),
            )
        else:
            # Simple array
            return f.transform(col, lambda x: x.cast(target_type.elementType))

    elif isinstance(source_type, StructType) and isinstance(target_type, StructType):
        return cast_struct(col, source_type, target_type)

    # Simple casts
    return col.cast(target_type)


def cast_struct(
    struct_col: Column,
    source_struct_type: StructType,
    target_struct_type: StructType,
) -> Column:
    """Cast a struct field to match the target struct schema.

    Handles missing fields, type conversions, and field reordering.

    Args:
        struct_col: The struct column to cast
        source_struct_type: The source struct type
        target_struct_type: The target struct type

    Returns:
        A struct with fields matching the target schema
    """
    source_fields = {field.name: field for field in source_struct_type.fields}

    field_exprs = []
    for target_field in target_struct_type.fields:
        field_name = target_field.name

        if field_name in source_fields:
            # Field exists - get it and cast to target type
            field_expr = struct_col.getField(field_name)
            source_field_type = source_fields[field_name].dataType

            # Recursively handle nested types
            field_expr = cast_column_to_target_type(
                field_expr,
                source_field_type,
                target_field.dataType,
            )

            field_exprs.append(field_expr.alias(field_name))
        else:
            # Field missing - add as null
            field_exprs.append(f.lit(None).cast(target_field.dataType).alias(field_name))

    return f.struct(*field_exprs)
