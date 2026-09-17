"""Locus-to-gene training and prediction, ported from gentropy onto polars."""

# xgboost MUST be imported before skops/scikit-learn anywhere in this package.
# sklearn ships its own LLVM OpenMP runtime (sklearn/.dylibs/libomp.dylib) and
# libxgboost.dylib links another; loading skops' first and xgboost's second kills
# the process with SIGSEGV on macOS, at the first fit or predict rather than at
# import. Linux is unaffected -- both link the one system libgomp -- so this
# protects the dev machine, not production. Importing the package here means every
# submodule inherits the ordering instead of each repeating it.
import xgboost  # noqa: F401
