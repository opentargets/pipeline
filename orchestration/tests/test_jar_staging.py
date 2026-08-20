"""Tests for Spark ``spark.jars`` staging resolution.

``resolve_jar_staging`` turns a cluster's rendered ``spark.jars`` value into the
list of ``(src_url, dst_uri)`` pairs orchestration must stage before the cluster
is created. Every GCS jar must have a registered source; one that does not is a
misconfiguration and fails loudly, so a mistyped path cannot slip through
unstaged and leave the cluster pointing at an object that does not exist. Jars
that are not on GCS are provided elsewhere and are left alone.
"""

import pytest

from orchestration.utils import resolve_jar_staging

PREFIX = 'gs://opentargets-pipelines/up/pts/jars/'
NLP_DST = f'{PREFIX}spark-nlp-assembly-6.1.5.jar'
NLP_SRC = 'https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-6.1.5.jar'
REGISTRY = {NLP_DST: NLP_SRC}

# A jar baked into the Dataproc image, referenced by an absolute local path —
# the shape gentropy uses for hail in clusters.yaml `step_job_properties`.
IMAGE_JAR = '/opt/conda/miniconda3/lib/python3.11/site-packages/hail/backend/hail-all-spark.jar'


def test_registered_jar_is_staged():
    """A jar present in the registry resolves to its (src, dst) pair."""
    assert resolve_jar_staging(NLP_DST, REGISTRY, PREFIX) == [(NLP_SRC, NLP_DST)]


def test_non_gcs_jar_is_ignored():
    """A jar that is not on GCS comes from the image: nothing to stage."""
    assert resolve_jar_staging(IMAGE_JAR, REGISTRY, PREFIX) == []


def test_unregistered_jar_under_prefix_raises():
    """A jar under the managed prefix with no registered source is an error."""
    orphan = f'{PREFIX}mystery-1.0.jar'
    with pytest.raises(ValueError, match='no registered source'):
        resolve_jar_staging(orphan, REGISTRY, PREFIX)


def test_unregistered_gcs_jar_outside_prefix_raises():
    """An unregistered GCS jar anywhere is an error, not something to ignore.

    Nothing would stage it, so the cluster would be created pointing at an object
    that may not exist and every job on it would fail at submit.
    """
    external = 'gs://some-other-bucket/foo.jar'
    with pytest.raises(ValueError, match='no registered source'):
        resolve_jar_staging(external, REGISTRY, PREFIX)


def test_mistyped_prefix_raises_and_names_the_expected_prefix():
    """A near-miss path is the likeliest mistake, so it must not fall through."""
    typo = 'gs://opentargets-pipelines/up/pts/jar/spark-nlp-assembly-6.1.5.jar'
    with pytest.raises(ValueError, match='managed jar prefix'):
        resolve_jar_staging(typo, REGISTRY, PREFIX)


def test_comma_separated_mixed_list():
    """Multiple jars: each resolved independently; image-local ones drop out."""
    spark_jars = f'{NLP_DST}, {IMAGE_JAR}'
    assert resolve_jar_staging(spark_jars, REGISTRY, PREFIX) == [(NLP_SRC, NLP_DST)]


def test_empty_and_whitespace_yield_nothing():
    """No jars declared (empty or blank) stages nothing."""
    assert resolve_jar_staging('', REGISTRY, PREFIX) == []
    assert resolve_jar_staging('  , ', REGISTRY, PREFIX) == []
