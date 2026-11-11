"""The VIDRA Stan models: where they are, how they are loaded on a worker, and how posteriors are summarised.

The models live in `pts/src/pts/stan/vidra/`. On a Dataproc worker they are not compiled: the
cluster's init action installs CmdStan and the two compiled executables from a prebuilt tarball
(`pts/tools/build_vidra_stan.sh`), and `load_models` points cmdstanpy at them. Without an
executable directory (local runs, tests) the models are compiled once into a cache directory.
"""

from __future__ import annotations

import os
import tempfile
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cmdstanpy import CmdStanModel

STAN_DIR = Path(__file__).resolve().parents[2] / 'stan' / 'vidra'
"""The directory of the `.stan` sources inside the package."""

MULTI_VARIANT = 'VIDRA'
SINGLE_VARIANT = 'VIDRA_single_variant'
MODEL_NAMES = (MULTI_VARIANT, SINGLE_VARIANT)

ADVI_SEED = 412
ADVI_MAX_ITER = 10_000
ADVI_GRAD_SAMPLES = 20
ADVI_DRAWS = 1_000
FULLRANK_MAX_N = 200
"""Variants above which the multi-variant model uses meanfield ADVI: fullrank's covariance is O((4N)^2)."""

PERCENTILES = (1, 2.5, 5, 10, 25, 40, 50, 60, 75, 90, 95, 97.5, 99)
POSTERIOR_COLUMNS = (
    'mean',
    'median',
    *[f'percentile{str(p).replace(".", "_")}' for p in PERCENTILES],
    'probabilityPositive',
    'probabilityNegative',
)


def scratch_dir() -> str:
    """Where Stan writes its output files: the Spark local directory on a cluster, else the system temp.

    `/tmp` on Dataproc is a size-capped tmpfs that fails for large gene-disease pairs. `TMPDIR` is
    set too so that cmdstanpy's data files go to the same disk.
    """
    scratch = os.environ.get('SPARK_LOCAL_DIRS', tempfile.gettempdir()).split(',')[0]
    os.environ['TMPDIR'] = scratch
    return scratch


@cache
def load_models(exe_dir: str | None = None, cmdstan_dir: str | None = None) -> dict[str, CmdStanModel]:
    """The two models, once per process.

    Args:
        exe_dir: directory holding the prebuilt executables `VIDRA` and `VIDRA_single_variant`.
            Without it, the sources are compiled into a cache directory under `scratch_dir`.
        cmdstan_dir: the CmdStan installation to use; otherwise cmdstanpy's default lookup.
    """
    import cmdstanpy
    from cmdstanpy import CmdStanModel

    if cmdstan_dir:
        cmdstanpy.set_cmdstan_path(cmdstan_dir)
    models = {}
    for name in MODEL_NAMES:
        if exe_dir:
            models[name] = CmdStanModel(exe_file=str(Path(exe_dir) / name))
        else:
            cache_dir = Path(scratch_dir()) / 'vidra_stan_cache'
            cache_dir.mkdir(parents=True, exist_ok=True)
            source = cache_dir / f'{name}.stan'
            if not source.exists() or source.read_text() != (STAN_DIR / f'{name}.stan').read_text():
                source.write_text((STAN_DIR / f'{name}.stan').read_text())
            models[name] = CmdStanModel(stan_file=str(source))
    return models


def summarise(draws: np.ndarray) -> dict[str, float]:
    """Posterior summary of one parameter's draws, keyed by `POSTERIOR_COLUMNS`.

    Examples:
        >>> s = summarise(np.array([-1.0, 0.0, 1.0, 2.0]))
        >>> (s['mean'], s['median'], s['probabilityPositive'], s['probabilityNegative'])
        (0.5, 0.5, 0.5, 0.25)
    """
    values = np.asarray(draws, dtype=float)
    summary = {'mean': float(values.mean()), 'median': float(np.median(values))}
    for percentile in PERCENTILES:
        summary[f'percentile{str(percentile).replace(".", "_")}'] = float(np.percentile(values, percentile))
    summary['probabilityPositive'] = float((values > 0).mean())
    summary['probabilityNegative'] = float((values < 0).mean())
    return summary
