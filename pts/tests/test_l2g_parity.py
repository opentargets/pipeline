"""Fixed-model parity against a published release.

The sharpest available test of the migration: take the release's OWN classifier, run it through
this polars path over the release's own feature matrix, and require the scores back. Because the
model is held fixed, any difference is the migration's doing and not the retraining's.

Row-level SHAP parity is deliberately NOT asserted. Each of gentropy's 1000 Batch tasks drew its
own unseeded background, so `shapBaseValue` varies across the baseline's own partitions -- 0.0381
to 0.0668 in 26.09-1. Re-running gentropy would not reproduce it either.

Gated on an environment variable because it reads a multi-gigabyte GCS dataset:

    L2G_PARITY_RUN=do/platform-2609-1 uv run --frozen pytest tests/test_l2g_parity.py -rxs
"""

import os

import polars as pl
import pytest

from pts.transformers.l2g.features import FEATURES, impute_and_cast
from pts.transformers.l2g.model import load_model, to_matrix

RUN = os.environ.get('L2G_PARITY_RUN')
BUCKET = os.environ.get('L2G_PARITY_BUCKET', 'gs://open-targets-pipeline-runs')

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not RUN, reason='set L2G_PARITY_RUN to a run prefix to enable'),
]


@pytest.fixture(scope='module')
def scored(tmp_path_factory) -> pl.DataFrame:
    """Score the release's feature matrix with the release's own model."""
    base = f'{BUCKET}/{RUN}'
    local_model = tmp_path_factory.mktemp('model') / 'classifier.skops'
    import gcsfs

    gcsfs.GCSFileSystem().get(
        f'{base}/etc/model/locus_to_gene_model/classifier.skops'.removeprefix('gs://'),
        str(local_model),
    )
    model = load_model(str(local_model))

    gwas = (
        pl.scan_parquet(f'{base}/output/credible_set/*.parquet')
        .filter(pl.col('studyType') == 'gwas')
        .select('studyLocusId')
        .unique()
    )
    prepared = impute_and_cast(
        pl.scan_parquet(f'{base}/intermediate/l2g_feature_matrix/*.parquet')
        .filter(pl.col('isProteinCoding') == 1)
        .join(gwas, on='studyLocusId', how='semi'),
        list(FEATURES),
    ).collect()

    matrix = to_matrix(prepared, list(FEATURES))
    return prepared.select('studyLocusId', 'geneId').with_columns(
        pl.Series('score', model.predict_proba(matrix)[:, 1])
    )


@pytest.fixture(scope='module')
def baseline() -> pl.DataFrame:
    return pl.read_parquet(
        f'{BUCKET}/{RUN}/output/l2g_prediction/*.parquet', columns=['studyLocusId', 'geneId', 'score']
    )


def test_scores_match_the_published_column(scored: pl.DataFrame, baseline: pl.DataFrame) -> None:
    joined = baseline.join(scored, on=['studyLocusId', 'geneId'], how='inner', suffix='_new')
    assert joined.height == baseline.height, 'a published prediction has no row in the new path'
    delta = (joined['score'] - joined['score_new']).abs()
    assert float(delta.max()) < 1e-6, f'max |delta| = {delta.max()}'


def test_the_row_set_at_threshold_matches(scored: pl.DataFrame, baseline: pl.DataFrame) -> None:
    mine = scored.filter(pl.col('score') >= 0.05).select('studyLocusId', 'geneId')
    theirs = baseline.select('studyLocusId', 'geneId')
    assert mine.height == theirs.height
    assert mine.join(theirs, on=['studyLocusId', 'geneId'], how='anti').height == 0


def test_the_baselines_own_base_value_is_not_constant() -> None:
    """Documents why row-level SHAP parity is not asserted anywhere in this file."""
    values = pl.read_parquet(
        f'{BUCKET}/{RUN}/output/l2g_prediction/*.parquet', columns=['shapBaseValue']
    )['shapBaseValue'].unique()
    assert values.len() > 1, 'baseline base value is constant; revisit the SHAP parity decision'
    assert float(values.max()) / float(values.min()) > 1.5
