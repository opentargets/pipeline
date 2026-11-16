"""Fixed-model parity against a published release.

The sharpest available test of the migration: take the release's OWN classifier, run the SHIPPED
`l2g_predict` over the release's own feature matrix, and require the scores back. Because the model
is held fixed, any difference is the migration's doing and not the retraining's.

The fixture calls `l2g_predict` rather than re-implementing its preparation. An earlier version
hand-rolled the scan, the `isProteinCoding` filter and the GWAS semi-join, and so certified a copy
of the code rather than the code.

What this gate certifies, precisely: the shipped `l2g_predict`'s filters, its scoring and its
output assembly, against a dataset a real release published. Drop the `isProteinCoding` filter or
the GWAS restriction and the row-set test fails on the extra rows.

What it cannot certify: the boundary comparison at the threshold. No released row scores exactly
0.05 in float64, so `>=` and `>` produce byte-identical output here and no gate over released data
could tell them apart. That mutation is covered instead by
`test_l2g_predict.py::test_l2g_predict_keeps_a_row_scoring_exactly_at_the_threshold`, which derives
its threshold from a score the model actually produced.

Row-level SHAP parity is deliberately NOT asserted, and `explain_predictions` is off, which is
also what keeps this affordable. Releases made by gentropy Batch cannot serve as a SHAP reference:
each of its 1000 tasks drew its own unseeded background, so `shapBaseValue` varied across the
release's own partitions (200 distinct values on `do/platform-2609-1`, a 4.91x spread).
`l2g_predict` writes one base value for the whole run, so a v4 release has a constant
`shapBaseValue` and nothing about it is asserted here.

The run must be a gentropy v4 release: its feature matrix, credible set and predictions keyed by
`credibleSetId`, so pre-v4 runs such as `do/platform-2609-1` no longer qualify.

Gated on an environment variable because it reads a multi-gigabyte GCS dataset:

    L2G_PARITY_RUN=<v4 run prefix> uv run --frozen pytest tests/test_l2g_parity.py -rxs
"""

import os

import polars as pl
import pytest

from pts.transformers.l2g.features import FEATURES
from pts.transformers.l2g_predict import l2g_predict

RUN = os.environ.get('L2G_PARITY_RUN')
BUCKET = os.environ.get('L2G_PARITY_BUCKET', 'gs://open-targets-pipeline-runs')

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not RUN, reason='set L2G_PARITY_RUN to a run prefix to enable'),
]


@pytest.fixture(scope='module')
def scored(tmp_path_factory) -> pl.DataFrame:
    """Run the shipped `l2g_predict` over the release's feature matrix and its own model.

    No local staging: `load_model` reads its bytes through otter's storage abstraction, so the
    model is loaded straight from the release bucket. The threshold is the production 0.05, so the
    row set this produces is exactly what the released run should have produced.
    """
    base = f'{BUCKET}/{RUN}'
    destination = str(tmp_path_factory.mktemp('parity') / 'l2g_prediction')
    l2g_predict(
        {
            'feature_matrix': f'{base}/intermediate/l2g_feature_matrix',
            'credible_set': f'{base}/output/credible_set',
            'model': f'{base}/etc/model/locus_to_gene_model/classifier.skops',
            'background': f'{base}/etc/model/locus_to_gene_model/shap_background.parquet',
        },
        destination,
        {
            'features_list': list(FEATURES),
            'l2g_threshold': 0.05,
            'explain_predictions': False,
            'shap_background_size': 100,
        },
        None,
    )
    return pl.read_parquet(
        f'{destination}/*.parquet', columns=['credibleSetId', 'geneId', 'score']
    )


@pytest.fixture(scope='module')
def baseline() -> pl.DataFrame:
    return pl.read_parquet(
        f'{BUCKET}/{RUN}/output/l2g_prediction/*.parquet', columns=['credibleSetId', 'geneId', 'score']
    )


def test_scores_match_the_published_column(scored: pl.DataFrame, baseline: pl.DataFrame) -> None:
    joined = baseline.join(scored, on=['credibleSetId', 'geneId'], how='inner', suffix='_new')
    assert joined.height == baseline.height, 'a published prediction has no row in the new path'
    delta = (joined['score'] - joined['score_new']).abs()
    assert float(delta.max()) < 1e-6, f'max |delta| = {delta.max()}'


def test_the_row_set_at_threshold_matches(scored: pl.DataFrame, baseline: pl.DataFrame) -> None:
    mine = scored.filter(pl.col('score') >= 0.05).select('credibleSetId', 'geneId')
    theirs = baseline.select('credibleSetId', 'geneId')
    assert mine.height == theirs.height
    assert mine.join(theirs, on=['credibleSetId', 'geneId'], how='anti').height == 0

