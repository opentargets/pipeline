"""Regenerate `normalisation.parquet`: OnToma's own spark expressions and spark-nlp pipeline over a label sample.

The test suite cannot run the real spark-nlp pipeline, which needs its jar from Maven and Java
17+, so the polars port of OnToma is checked against this recorded output instead. Rerun after
bumping ontoma or spark-nlp, then rerun the tests:

    cd pts && JAVA_HOME=<java 17+> PYSPARK_PYTHON=$PWD/.venv/bin/python \\
        uv run python tests/test_data/disease_mapping/generate_normalisation_fixture.py <disease index parquet>

The disease index is sampled for real names, synonyms and cross references; the edge cases are
fixed below. spark-nlp is resolved from Maven by `spark.jars.packages`.
"""

# the edge cases are look-alike characters on purpose
# ruff: noqa: RUF001

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pyspark.sql.functions as f
from ontoma import spark_nlp_coordinate
from ontoma.common.utils import clean_disease_label, determine_track, format_identifier, get_alternative_translations
from ontoma.nlp_pipeline import NLPPipeline
from pyspark.sql import SparkSession

OUTPUT = Path(__file__).with_name('normalisation.parquet')
SAMPLE_SIZE = 1500
SEED = 2609

EDGE_CASES = [
    '',
    ' ',
    '  padded  ',
    'the',
    'The',
    'THE',
    "aren't",
    "Aren't",
    '(the)',
    '"of"',
    "Alzheimer's disease",
    'type-2 diabetes',
    'a/b',
    '--',
    '-',
    '((',
    'x--y',
    '-(of)-',
    'of.-x',
    'agreed',
    'feed',
    'replacement',
    'hopping',
    'filing',
    'falling',
    'generalization',
    's',
    "'s",
    'ss',
    'ies',
    'sses',
    'y',
    'by',
    'sky',
    'ed',
    'ing',
    'eed',
    'bled',
    'Papillon-Lefèvre syndrome',
    'small, protruding, ‚Äúcup-shaped‚Äù ears',
    'α-thalassemia',
    'ΑΒΓ disease',
    'β-ketothiolase deficiency',
    'tab\tsep',
    'new\nline',
    'nb space',
    'vt\x0bsep',
    'ls sep',
    'nel\u0085sep',
    # the tokenizer's break marker, and its protect marker, which does not break
    'theↇasthma',
    'fibrosisↇcystic',
    'aↈb c',
    'ﬁbrosis',
    'İstanbul',
    'ſyndrome',
    'straße',
    'snake_case_label',
    'C3/C5',
    '12.5',
    'e.g. foo',
    'U.S.A.',
    'MS',
    'ms',
    'ALS',
    'Als',
    'HbSS',
    'CF-1',
    'T2D',
    'a b',
    'ABCDEFGHIJK',
    'ABCDEFGHIJKL',
    'AbCdEfGhIj',
    'abcde',
    'ABcde',
    '1234',
    '12345',
    '#',
    'abc#',
    'a#b#c',
    '##',
    'OMIM:1#Name',
    '#Q12.3 Name',
    '#A01 Name',
    '#Chapter IV Neoplasms',
    '#chapter iv neoplasms',
    '#Name\n',
    '#Name\r\n',
    '#Name\r',
    '#a Name',
    '#Name tail',
    'ORDO:558',
    'Orphanet_558',
    'ORPHA:558',
    'orpha:558',
    'Orphanet:558',
    'MONDO_0007739',
    'mondo:0007739',
    'OMIM:154700',
    'http://www.orpha.net/ORDO/Orphanet_558',
    ' omim:154700\t',
    'x:a_b\n',
    'x:a_b\r',
    'x:a_b ',
    'ab_',
    '_ab',
    'a:_',
    'A:B_C:D_E',
    'ﬁ:1',
    'ß:x_y',
]


def _labels(disease_index: str) -> list[str]:
    index = pl.read_parquet(disease_index)
    synonyms = index['synonyms'].struct.unnest()
    pools = {
        'names': index['name'],
        'synonyms': pl.concat([synonyms[c].explode() for c in synonyms.columns]),
        'ids': pl.concat([index['id'], index['dbXRefs'].explode(), index['obsoleteTerms'].explode()]),
    }
    sampled = {
        label
        for pool in pools.values()
        for label in pool.drop_nulls().unique().sort().sample(SAMPLE_SIZE, seed=SEED).to_list()
    }
    return sorted(sampled | set(EDGE_CASES))


def main(disease_index: str) -> None:
    spark = SparkSession.builder.master('local[2]').config('spark.jars.packages', spark_nlp_coordinate()).getOrCreate()
    labels = spark.createDataFrame([(label,) for label in _labels(disease_index)], 'label string')
    queried = labels.select(
        'label',
        determine_track(f.col('label')).alias('track'),
        get_alternative_translations(f.trim(f.col('label'))).alias('translations'),
        clean_disease_label(f.col('label')).alias('cleanedDiseaseLabel'),
        format_identifier(f.upper(f.trim(f.col('label')))).alias('formattedIdentifier'),
    )
    normalised = NLPPipeline.apply_pipeline(queried, 'label').select(
        'label',
        'track',
        'translations',
        'cleanedDiseaseLabel',
        'formattedIdentifier',
        # as OnToma._normalise_entities finishes each track
        f.array_join(
            f.array_sort(f.filter(f.array_distinct('finished_term'), lambda c: c.isNotNull() & (c != ''))), ''
        ).alias('term'),
        f.array_join(f.filter('finished_symbol', lambda c: c.isNotNull() & (c != '')), '').alias('symbol'),
    )
    pl.DataFrame(normalised.toArrow()).sort('label').write_parquet(OUTPUT)
    spark.stop()


if __name__ == '__main__':
    main(sys.argv[1])
