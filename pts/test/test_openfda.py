"""Tests for the openfda pyspark module.

Ported from platform-etl-backend OpenFDA step.
"""

from pyspark.sql import Row
from pyspark.sql.types import (
    ArrayType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.openfda import (
    _prepare_adverse_event_data,
    _prepare_blacklist,
    _prepare_drug_list,
    _prepare_for_montecarlo,
    _prepare_summary_statistics,
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

CHEMBL_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('name', StringType()),
    StructField('synonyms', ArrayType(StringType())),
    StructField('tradeNames', ArrayType(StringType())),
])

BLACKLIST_SCHEMA = StructType([
    StructField('reactions', StringType()),
])


# ---------------------------------------------------------------------------
# 1. _prepare_drug_list
# ---------------------------------------------------------------------------


def test_prepare_drug_list_includes_pref_name(spark):
    """_prepare_drug_list includes the preferred name as a drug_name."""
    data = [Row(id='CHEMBL1', name='Aspirin', synonyms=[], tradeNames=[])]
    df = spark.createDataFrame(data, CHEMBL_SCHEMA)
    result = _prepare_drug_list(df)
    names = {r.drug_name for r in result.collect()}
    assert 'aspirin' in names


def test_prepare_drug_list_includes_synonyms(spark):
    """_prepare_drug_list includes synonyms as drug_names."""
    data = [Row(id='CHEMBL1', name='Drug', synonyms=['SynA', 'SynB'], tradeNames=[])]
    df = spark.createDataFrame(data, CHEMBL_SCHEMA)
    result = _prepare_drug_list(df)
    names = {r.drug_name for r in result.collect()}
    assert 'syna' in names
    assert 'synb' in names


def test_prepare_drug_list_includes_trade_names(spark):
    """_prepare_drug_list includes trade names as drug_names."""
    data = [Row(id='CHEMBL1', name='Drug', synonyms=[], tradeNames=['BrandX'])]
    df = spark.createDataFrame(data, CHEMBL_SCHEMA)
    result = _prepare_drug_list(df)
    names = {r.drug_name for r in result.collect()}
    assert 'brandx' in names


def test_prepare_drug_list_output_columns(spark):
    """_prepare_drug_list output has exactly chembl_id and drug_name."""
    data = [Row(id='CHEMBL1', name='Drug', synonyms=[], tradeNames=[])]
    df = spark.createDataFrame(data, CHEMBL_SCHEMA)
    result = _prepare_drug_list(df)
    assert set(result.columns) == {'chembl_id', 'drug_name'}


def test_prepare_drug_list_deduplicates(spark):
    """_prepare_drug_list deduplicates (chembl_id, drug_name) pairs."""
    data = [
        Row(id='CHEMBL1', name='Drug', synonyms=['Drug'], tradeNames=[]),
    ]
    df = spark.createDataFrame(data, CHEMBL_SCHEMA)
    result = _prepare_drug_list(df)
    rows = result.filter(result.chembl_id == 'CHEMBL1').filter(result.drug_name == 'drug').collect()
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# 2. _prepare_blacklist
# ---------------------------------------------------------------------------


def test_prepare_blacklist_lowercases(spark):
    """_prepare_blacklist lowercases the reaction names."""
    data = [Row(reactions='DEATH')]
    df = spark.createDataFrame(data, BLACKLIST_SCHEMA)
    result = _prepare_blacklist(df)
    rows = result.collect()
    assert rows[0].reactions == 'death'


def test_prepare_blacklist_trims(spark):
    """_prepare_blacklist trims whitespace."""
    data = [Row(reactions='  headache  ')]
    df = spark.createDataFrame(data, BLACKLIST_SCHEMA)
    result = _prepare_blacklist(df)
    rows = result.collect()
    assert rows[0].reactions == 'headache'


def test_prepare_blacklist_output_column(spark):
    """_prepare_blacklist output has a reactions column."""
    data = [Row(reactions='death')]
    df = spark.createDataFrame(data, BLACKLIST_SCHEMA)
    result = _prepare_blacklist(df)
    assert 'reactions' in result.columns


# ---------------------------------------------------------------------------
# 3. _prepare_adverse_event_data
# ---------------------------------------------------------------------------

_PATIENT_SCHEMA = StructType([
    StructField('safetyreportid', StringType()),
    StructField('serious', StringType()),
    StructField('seriousnessdeath', StringType()),
    StructField('receivedate', StringType()),
    StructField('qualification', StringType()),
    StructField('reaction_reactionmeddrapt', StringType()),
    StructField('drug_medicinalproduct', StringType()),
    StructField('drug_generic_name_list', ArrayType(StringType())),
    StructField('drug_brand_name_list', ArrayType(StringType())),
    StructField('drug_substance_name_list', ArrayType(StringType())),
    StructField('drugcharacterization', StringType()),
])


def _make_pre_prepped_row(**kwargs):
    defaults = {
        'safetyreportid': 'R1',
        'serious': '1',
        'seriousnessdeath': '0',
        'receivedate': '20200101',
        'qualification': '1',
        'reaction_reactionmeddrapt': 'headache',
        'drug_medicinalproduct': 'aspirin',
        'drug_generic_name_list': [],
        'drug_brand_name_list': [],
        'drug_substance_name_list': [],
        'drugcharacterization': '1',
    }
    defaults.update(kwargs)
    return Row(**defaults)


def test_adverse_event_qualification_filter(spark):
    """_prepare_adverse_event_data keeps qualification 1,2,3 and drops others."""
    data = [
        _make_pre_prepped_row(safetyreportid='R1', qualification='1'),
        _make_pre_prepped_row(safetyreportid='R2', qualification='4'),
        _make_pre_prepped_row(safetyreportid='R3', qualification='2'),
    ]
    df = spark.createDataFrame(data, _PATIENT_SCHEMA)
    drug_df = spark.createDataFrame(
        [Row(chembl_id='CHEMBL1', drug_name='aspirin')],
        StructType([StructField('chembl_id', StringType()), StructField('drug_name', StringType())]),
    )
    result = _prepare_adverse_event_data(df, drug_df, spark.createDataFrame([], BLACKLIST_SCHEMA))
    ids = {r.safetyreportid for r in result.collect()}
    assert 'R1' in ids
    assert 'R3' in ids
    assert 'R2' not in ids


def test_adverse_event_excludes_serious_death(spark):
    """_prepare_adverse_event_data excludes rows with seriousness_death != '0'."""
    data = [
        _make_pre_prepped_row(safetyreportid='R1', seriousnessdeath='0'),
        _make_pre_prepped_row(safetyreportid='R2', seriousnessdeath='1'),
    ]
    df = spark.createDataFrame(data, _PATIENT_SCHEMA)
    drug_df = spark.createDataFrame(
        [Row(chembl_id='CHEMBL1', drug_name='aspirin')],
        StructType([StructField('chembl_id', StringType()), StructField('drug_name', StringType())]),
    )
    result = _prepare_adverse_event_data(df, drug_df, spark.createDataFrame([], BLACKLIST_SCHEMA))
    ids = {r.safetyreportid for r in result.collect()}
    assert 'R1' in ids
    assert 'R2' not in ids


def test_adverse_event_filters_blacklisted(spark):
    """_prepare_adverse_event_data removes blacklisted reactions."""
    data = [
        _make_pre_prepped_row(safetyreportid='R1', reaction_reactionmeddrapt='headache'),
        _make_pre_prepped_row(safetyreportid='R2', reaction_reactionmeddrapt='death'),
    ]
    df = spark.createDataFrame(data, _PATIENT_SCHEMA)
    drug_df = spark.createDataFrame(
        [Row(chembl_id='CHEMBL1', drug_name='aspirin')],
        StructType([StructField('chembl_id', StringType()), StructField('drug_name', StringType())]),
    )
    blacklist = spark.createDataFrame([Row(reactions='death')], BLACKLIST_SCHEMA)
    result = _prepare_adverse_event_data(df, drug_df, blacklist)
    reactions = {r.reaction_reactionmeddrapt for r in result.collect()}
    assert 'headache' in reactions
    assert 'death' not in reactions


def test_adverse_event_joins_chembl(spark):
    """_prepare_adverse_event_data links drug names to chembl_id."""
    data = [_make_pre_prepped_row(drug_medicinalproduct='aspirin')]
    df = spark.createDataFrame(data, _PATIENT_SCHEMA)
    drug_df = spark.createDataFrame(
        [Row(chembl_id='CHEMBL1', drug_name='aspirin')],
        StructType([StructField('chembl_id', StringType()), StructField('drug_name', StringType())]),
    )
    result = _prepare_adverse_event_data(df, drug_df, spark.createDataFrame([], BLACKLIST_SCHEMA))
    rows = result.collect()
    assert len(rows) >= 1
    assert rows[0].chembl_id == 'CHEMBL1'


# ---------------------------------------------------------------------------
# 4. _prepare_summary_statistics
# ---------------------------------------------------------------------------

_COOKED_SCHEMA = StructType([
    StructField('safetyreportid', StringType()),
    StructField('reaction_reactionmeddrapt', StringType()),
    StructField('chembl_id', StringType()),
])


def test_summary_statistics_output_columns(spark):
    """_prepare_summary_statistics produces expected output columns."""
    data = [
        Row(safetyreportid='R1', reaction_reactionmeddrapt='headache', chembl_id='CHEMBL1'),
        Row(safetyreportid='R2', reaction_reactionmeddrapt='headache', chembl_id='CHEMBL1'),
        Row(safetyreportid='R3', reaction_reactionmeddrapt='nausea', chembl_id='CHEMBL2'),
    ]
    df = spark.createDataFrame(data, _COOKED_SCHEMA)
    result = _prepare_summary_statistics(df, 'chembl_id', 'chembl_id_stats')
    assert set(result.columns) == {
        'safetyreportid',
        'reaction_reactionmeddrapt',
        'uniq_report_ids_by_reaction',
        'chembl_id_stats',
        'uniq_report_ids',
        'chembl_id',
    }


def test_summary_statistics_counts_reports_by_reaction(spark):
    """_prepare_summary_statistics counts distinct reports per reaction."""
    data = [
        Row(safetyreportid='R1', reaction_reactionmeddrapt='headache', chembl_id='CHEMBL1'),
        Row(safetyreportid='R2', reaction_reactionmeddrapt='headache', chembl_id='CHEMBL2'),
        Row(safetyreportid='R3', reaction_reactionmeddrapt='nausea', chembl_id='CHEMBL1'),
    ]
    df = spark.createDataFrame(data, _COOKED_SCHEMA)
    result = _prepare_summary_statistics(df, 'chembl_id', 'chembl_id_stats')
    headache_rows = [r for r in result.collect() if r.reaction_reactionmeddrapt == 'headache']
    # Both headache rows should report 2 distinct reports for that reaction
    assert all(r.uniq_report_ids_by_reaction >= 2 for r in headache_rows)


# ---------------------------------------------------------------------------
# 5. _prepare_for_montecarlo
# ---------------------------------------------------------------------------

_STATS_SCHEMA = StructType([
    StructField('safetyreportid', StringType()),
    StructField('reaction_reactionmeddrapt', StringType()),
    StructField('chembl_id', StringType()),
    StructField('chembl_id_stats', StringType()),
    StructField('uniq_report_ids_by_reaction', StringType()),
    StructField('uniq_report_ids', StringType()),
])


_STATS_SCHEMA_LONG = StructType([
    StructField('safetyreportid', StringType()),
    StructField('reaction_reactionmeddrapt', StringType()),
    StructField('chembl_id', StringType()),
    StructField('chembl_id_stats', LongType()),
    StructField('uniq_report_ids_by_reaction', LongType()),
    StructField('uniq_report_ids', LongType()),
])


def test_montecarlo_output_has_llr(spark):
    """_prepare_for_montecarlo output includes an llr column."""
    data = [
        Row(
            safetyreportid='R1',
            reaction_reactionmeddrapt='headache',
            chembl_id='CHEMBL1',
            chembl_id_stats=10,
            uniq_report_ids_by_reaction=5,
            uniq_report_ids=3,
        ),
    ]
    df = spark.createDataFrame(data, _STATS_SCHEMA_LONG)
    result = _prepare_for_montecarlo(df, 'chembl_id_stats')
    assert 'llr' in result.columns


def test_montecarlo_drops_safetyreportid(spark):
    """_prepare_for_montecarlo drops safetyreportid column."""
    data = [
        Row(
            safetyreportid='R1',
            reaction_reactionmeddrapt='headache',
            chembl_id='CHEMBL1',
            chembl_id_stats=10,
            uniq_report_ids_by_reaction=5,
            uniq_report_ids=3,
        ),
    ]
    df = spark.createDataFrame(data, _STATS_SCHEMA_LONG)
    result = _prepare_for_montecarlo(df, 'chembl_id_stats')
    assert 'safetyreportid' not in result.columns


def test_montecarlo_filters_null_llr(spark):
    """_prepare_for_montecarlo removes rows where llr is null or NaN."""
    # A=0, which would cause log(0) → NaN/null llr
    data = [
        Row(
            safetyreportid='R1',
            reaction_reactionmeddrapt='headache',
            chembl_id='CHEMBL1',
            chembl_id_stats=10,
            uniq_report_ids_by_reaction=5,
            uniq_report_ids=0,
        ),
    ]
    df = spark.createDataFrame(data, _STATS_SCHEMA_LONG)
    result = _prepare_for_montecarlo(df, 'chembl_id_stats')
    # Rows with A=0 produce log(0) → NaN, should be filtered out
    rows = result.collect()
    for r in rows:
        assert r.llr is not None
