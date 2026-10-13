"""Locus-to-gene training and prediction, ported from gentropy onto polars."""

# import xgboost before skops and sklearn: the reverse order segfaults on macos
import xgboost  # noqa: F401
