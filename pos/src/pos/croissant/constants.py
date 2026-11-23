"""Constants for the OT-Croissant package."""

import mlcroissant as mlc
import polars as pl

# Maps the polars dtype of a model value to its Croissant DataType. Each unsigned integer maps to the next
# wider signed type (UInt64 to FLOAT64).
TYPE_DICT = {
    # Text / boolean
    pl.String: mlc.DataType.TEXT,
    pl.Enum: mlc.DataType.TEXT,
    pl.Boolean: mlc.DataType.BOOL,
    # Integers
    pl.Int8: mlc.DataType.INT8,
    pl.Int16: mlc.DataType.INT16,
    pl.Int32: mlc.DataType.INT32,
    pl.Int64: mlc.DataType.INT64,
    pl.UInt8: mlc.DataType.INT16,
    pl.UInt16: mlc.DataType.INT32,
    pl.UInt32: mlc.DataType.INT64,
    pl.UInt64: mlc.DataType.FLOAT64,
    # Floating point
    pl.Float32: mlc.DataType.FLOAT32,
    pl.Float64: mlc.DataType.FLOAT64,
    # Date / time
    pl.Date: mlc.DataType.DATE,
    pl.Datetime: mlc.DataType.DATETIME,
}
