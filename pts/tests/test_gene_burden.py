"""Tests for the gene_burden pyspark module."""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from pts.pyspark.common.session import Session
from pts.pyspark.gene_burden import _brava_sample_size_lookup, _excel_sheet_to_spark, process_cvdi_gene_burden

# Every cell of the BRaVa workbook is read as a string (see `_excel_sheet_to_spark`), so the
# fixtures below mirror that and leave the casting to the code under test.
S4_COLUMNS = ['Phenotype ID', 'Ancestry', 'Biobank', 'N cases', 'N controls']
S5_COLUMNS = ['Phenotype ID', 'Ancestry', 'Biobank', 'N']

# Binary contributions: two biobanks for EUR, one each for AFR and MID.
S4_ROWS = [
    ('T2Diab', 'EUR', 'uk-biobank', '100', '900'),
    ('T2Diab', 'EUR', 'all-of-us', '200', '800'),
    ('T2Diab', 'AFR', 'all-of-us', '30', '70'),
    ('T2Diab', 'MID', 'ccpm', '5', '15'),
]

# Quantitative contributions: 'all-of-us' is listed twice, identically, exactly as Table S5 does.
S5_ROWS = [
    ('Height', 'EUR', 'uk-biobank', '5000'),
    ('Height', 'EUR', 'all-of-us', '1000'),
    ('Height', 'EUR', 'all-of-us', '1000'),
    ('Height', 'AFR', 'all-of-us', '400'),
]


def _lookup(spark: SparkSession) -> dict[tuple[str, str], tuple[int | None, int | None]]:
    """Run the lookup over the fixtures and index it by (Phenotype ID, Ancestry Group)."""
    rows = _brava_sample_size_lookup(
        spark.createDataFrame(S4_ROWS, S4_COLUMNS),
        spark.createDataFrame(S5_ROWS, S5_COLUMNS),
    ).collect()
    return {(r['Phenotype ID'], r['Ancestry Group']): (r['studySampleSize'], r['studyCases']) for r in rows}


def test_identical_quantitative_contributions_are_counted_once(spark: SparkSession):
    """Table S5 repeats some biobank rows verbatim; the repeat is the same participants, not a second cohort."""
    lookup = _lookup(spark)

    # 5000 + 1000, NOT 5000 + 1000 + 1000.
    assert lookup[('Height', 'EUR')][0] == 6000
    assert lookup[('Height', 'ALL')][0] == 6400


def test_pooled_and_non_eur_groups_are_consistent_with_the_ancestry_groups(spark: SparkSession):
    """'ALL' - 'EUR' == 'non_EUR' has to hold by construction, for both trait types."""
    lookup = _lookup(spark)

    for phenotype in ('T2Diab', 'Height'):
        pooled, eur, non_eur = (lookup[(phenotype, group)][0] for group in ('ALL', 'EUR', 'non_EUR'))
        assert pooled is not None
        assert eur is not None
        assert pooled - eur == non_eur


def test_binary_totals_sum_cases_and_controls_across_biobanks(spark: SparkSession):
    """The sample size is cases + controls, while studyCases counts the cases alone."""
    lookup = _lookup(spark)

    assert lookup[('T2Diab', 'EUR')] == (2000, 300)
    assert lookup[('T2Diab', 'AFR')] == (100, 30)
    assert lookup[('T2Diab', 'ALL')] == (2120, 335)


def test_an_ancestry_that_is_never_an_evidence_group_still_counts_towards_the_pooled_totals(spark: SparkSession):
    """'MID' contributes to S4 but never appears as an evidence group, yet it is part of ALL and non_EUR."""
    lookup = _lookup(spark)

    assert lookup[('T2Diab', 'MID')] == (20, 5)
    # non_EUR is AFR (100) + MID (20), so the unlabelled ancestry is not silently dropped.
    assert lookup[('T2Diab', 'non_EUR')] == (120, 35)


def test_quantitative_rows_report_no_case_count(spark: SparkSession):
    """Quantitative traits have no cases, so studyCases stays null rather than becoming zero."""
    lookup = _lookup(spark)

    assert lookup[('Height', 'EUR')][1] is None
    assert lookup[('Height', 'ALL')][1] is None


def test_the_lookup_key_is_unique(spark: SparkSession):
    """A duplicate key would fan out the left join that attaches sample sizes to evidence."""
    df = _brava_sample_size_lookup(
        spark.createDataFrame(S4_ROWS, S4_COLUMNS),
        spark.createDataFrame(S5_ROWS, S5_COLUMNS),
    )

    assert df.count() == df.select('Phenotype ID', 'Ancestry Group').distinct().count()


@pytest.fixture
def session_without_arrow(pts_session: Session) -> Iterator[Session]:
    """The shared session as production runs it, with Arrow off.

    The `spark` fixture enables Arrow and both fixtures share one SparkSession. Production sets no
    Arrow config, and it is the row-wise pandas-to-Spark path that turns a NaN into the string "nan".
    """
    arrow = pts_session.spark.conf.get('spark.sql.execution.arrow.pyspark.enabled')
    pts_session.spark.conf.set('spark.sql.execution.arrow.pyspark.enabled', 'false')
    try:
        yield pts_session
    finally:
        pts_session.spark.conf.set('spark.sql.execution.arrow.pyspark.enabled', arrow)


def test_blank_excel_cells_become_null(session_without_arrow: Session, tmp_path: Path):
    """Under pandas 3, a `str` column cannot hold None; the blanks must still reach Spark as null, not "nan"."""
    path = tmp_path / 'sheet.xlsx'
    pd.DataFrame({'gene': ['A', None], 'value': [None, 1.5]}).to_excel(path, sheet_name='S', index=False)

    rows = _excel_sheet_to_spark(session_without_arrow, str(path), 'S').collect()

    assert [r.asDict() for r in rows] == [{'gene': 'A', 'value': None}, {'gene': None, 'value': '1.5'}]


def _cvdi_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror the column hierarchy of CVDI tables ST6 and ST3: one LOF mask plus the Cauchy combined test."""
    associations = pd.DataFrame(
        [['Asthma', 'ENSG00000001', 'GENE1', 10.0, '2.5 [1.5; 3.5]', 1e-10, 1e-11]],
        columns=pd.MultiIndex.from_tuples([
            ('phenotype', 'Unnamed: 0_level_1', 'Unnamed: 0_level_2'),
            ('Gene ID Ensembl', 'Unnamed: 1_level_1', 'Unnamed: 1_level_2'),
            ('Gene', 'Unnamed: 2_level_1', 'Unnamed: 2_level_2'),
            ('ALL ancestry', 'LOF (MAF<0.1%)', 'cMAC'),
            ('ALL ancestry', 'LOF (MAF<0.1%)', 'OR [95%CI]'),
            ('ALL ancestry', 'LOF (MAF<0.1%)', 'Meta P-value'),
            ('ALL ancestry', 'Cauchy', 'Cauchy P-value'),
        ]),
    )
    cutoffs = pd.DataFrame(
        [['FDR1%', 'LOF (MAF<0.1%)', 1, 1, 1, 1, 1e-5], [None, 'Cauchy', 1, 1, 1, 1, 1e-5]],
        columns=pd.MultiIndex.from_tuples([
            ('Unnamed: 0_level_0', 'Significance cutoff'),
            ('Unnamed: 1_level_0', 'Mask'),
            ('UKB', 'P cutoff'),
            ('AoU', 'P cutoff'),
            ('MGB', 'P cutoff'),
            ('META (no correction)', 'P cutoff'),
            ('META (overlap corrected)', 'P cutoff'),
        ]),
    )
    return associations, cutoffs


def test_cvdi_cauchy_associations_are_kept_without_an_odds_ratio(session_without_arrow: Session):
    """The Cauchy combined test reports no odds ratio; its significant rows are evidence all the same."""
    associations, cutoffs = _cvdi_frames()

    df = process_cvdi_gene_burden(session_without_arrow, associations, cutoffs)
    rows = {r.statisticalMethod: r for r in df.collect()}

    assert set(rows) == {'LOF (MAF<0.1%)', 'Cauchy'}
    assert rows['Cauchy'].oddsRatio is None
    assert rows['LOF (MAF<0.1%)'].oddsRatio == 2.5
