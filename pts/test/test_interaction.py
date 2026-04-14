"""Tests for the interaction pyspark module.

Ported from platform-etl-backend Interaction step.
"""

from pyspark.sql import Row
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.interaction import (
    _transform_human_mapping,
    _transform_rnacentral,
    _transform_string_proteins,
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

RNACENTRAL_SCHEMA = StructType([
    StructField('_c0', StringType()),
    StructField('_c1', StringType()),
    StructField('_c2', StringType()),
    StructField('_c3', StringType()),
    StructField('_c4', StringType()),
    StructField('_c5', StringType()),
])

HUMAN_MAPPING_SCHEMA = StructType([
    StructField('_c0', StringType()),
    StructField('_c1', StringType()),
    StructField('_c2', StringType()),
])

STRINGS_SCHEMA = StructType([
    StructField('protein1', StringType()),
    StructField('protein2', StringType()),
    StructField('combined_score', StringType()),
    StructField('coexpression', StringType()),
    StructField('cooccurence', StringType()),
    StructField('neighborhood', StringType()),
    StructField('fusion', StringType()),
    StructField('homology', StringType()),
    StructField('experimental', StringType()),
    StructField('database', StringType()),
    StructField('textmining', StringType()),
])


# ---------------------------------------------------------------------------
# 1. _transform_rnacentral
# ---------------------------------------------------------------------------


def test_transform_rnacentral_maps_c0_to_mapped_id(spark):
    """_transform_rnacentral maps _c0 to mapped_id."""
    data = [Row(_c0='URS0000001', _c1='9606', _c2='x', _c3='y', _c4='z', _c5='ENSG00000001')]
    df = spark.createDataFrame(data, RNACENTRAL_SCHEMA)
    result = _transform_rnacentral(df)
    row = result.collect()[0]
    assert row.mapped_id == 'URS0000001'


def test_transform_rnacentral_maps_c5_to_gene_id(spark):
    """_transform_rnacentral maps _c5 to gene_id."""
    data = [Row(_c0='URS0000001', _c1='9606', _c2='x', _c3='y', _c4='z', _c5='ENSG00000001')]
    df = spark.createDataFrame(data, RNACENTRAL_SCHEMA)
    result = _transform_rnacentral(df)
    row = result.collect()[0]
    assert row.gene_id == 'ENSG00000001'


def test_transform_rnacentral_output_columns(spark):
    """_transform_rnacentral output has exactly gene_id and mapped_id columns."""
    data = [Row(_c0='URS0000001', _c1='9606', _c2='x', _c3='y', _c4='z', _c5='ENSG00000002')]
    df = spark.createDataFrame(data, RNACENTRAL_SCHEMA)
    result = _transform_rnacentral(df)
    assert set(result.columns) == {'gene_id', 'mapped_id'}


# ---------------------------------------------------------------------------
# 2. _transform_human_mapping
# ---------------------------------------------------------------------------


def test_transform_human_mapping_filters_ensembl(spark):
    """_transform_human_mapping keeps only rows where _c1 == 'Ensembl'."""
    data = [
        Row(_c0='P12345', _c1='Ensembl', _c2='ENSG00000001'),
        Row(_c0='Q98765', _c1='Gene_Name', _c2='ENSG00000001'),
        Row(_c0='BRCA1', _c1='Ensembl', _c2='ENSG00000002'),
    ]
    df = spark.createDataFrame(data, HUMAN_MAPPING_SCHEMA)
    result = _transform_human_mapping(df)
    rows = result.collect()
    ids = {r.id for r in rows}
    # only Ensembl rows contribute, but Gene_Name row is excluded
    assert 'ENSG00000001' in ids
    assert 'ENSG00000002' in ids


def test_transform_human_mapping_groups_by_c2(spark):
    """_transform_human_mapping groups by _c2 to produce mapping_list."""
    data = [
        Row(_c0='P12345', _c1='Ensembl', _c2='ENSG00000001'),
        Row(_c0='Q11111', _c1='Ensembl', _c2='ENSG00000001'),
    ]
    df = spark.createDataFrame(data, HUMAN_MAPPING_SCHEMA)
    result = _transform_human_mapping(df)
    rows = result.collect()
    assert len(rows) == 1
    row = rows[0]
    assert row.id == 'ENSG00000001'
    assert set(row.mapping_list) == {'P12345', 'Q11111'}


def test_transform_human_mapping_output_columns(spark):
    """_transform_human_mapping output has id and mapping_list columns."""
    data = [Row(_c0='P12345', _c1='Ensembl', _c2='ENSG00000001')]
    df = spark.createDataFrame(data, HUMAN_MAPPING_SCHEMA)
    result = _transform_human_mapping(df)
    assert set(result.columns) == {'id', 'mapping_list'}


def test_transform_human_mapping_mapping_list_is_array(spark):
    """_transform_human_mapping produces mapping_list as an array type."""
    data = [Row(_c0='P12345', _c1='Ensembl', _c2='ENSG00000001')]
    df = spark.createDataFrame(data, HUMAN_MAPPING_SCHEMA)
    result = _transform_human_mapping(df)
    field = [f for f in result.schema.fields if f.name == 'mapping_list'][0]
    assert isinstance(field.dataType, ArrayType)


# ---------------------------------------------------------------------------
# 3. STRING score threshold filtering
# ---------------------------------------------------------------------------


def test_strings_score_threshold_filters_low_scores(spark):
    """STRING interactions below the score threshold are excluded."""
    data = [
        Row(
            protein1='9606.ENSP00000001',
            protein2='9606.ENSP00000002',
            combined_score='500',
            coexpression='100',
            cooccurence='0',
            neighborhood='0',
            fusion='0',
            homology='0',
            experimental='200',
            database='0',
            textmining='100',
        ),
        Row(
            protein1='9606.ENSP00000003',
            protein2='9606.ENSP00000004',
            combined_score='200',
            coexpression='50',
            cooccurence='0',
            neighborhood='0',
            fusion='0',
            homology='0',
            experimental='100',
            database='0',
            textmining='50',
        ),
    ]
    df = spark.createDataFrame(data, STRINGS_SCHEMA)
    result = _transform_string_proteins(df, score_threshold=400)
    rows = result.collect()
    # only the row with combined_score=500 should survive
    assert len(rows) == 1


def test_strings_score_threshold_zero_keeps_all(spark):
    """STRING interactions all pass when threshold is 0."""
    data = [
        Row(
            protein1='9606.ENSP00000001',
            protein2='9606.ENSP00000002',
            combined_score='1',
            coexpression='0',
            cooccurence='0',
            neighborhood='0',
            fusion='0',
            homology='0',
            experimental='0',
            database='0',
            textmining='0',
        ),
    ]
    df = spark.createDataFrame(data, STRINGS_SCHEMA)
    result = _transform_string_proteins(df, score_threshold=0)
    rows = result.collect()
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# 4. STRING output schema fields
# ---------------------------------------------------------------------------


def test_string_proteins_output_schema_fields(spark):
    """_transform_string_proteins output has expected nested schema fields."""
    data = [
        Row(
            protein1='9606.ENSP00000001',
            protein2='9606.ENSP00000002',
            combined_score='500',
            coexpression='100',
            cooccurence='0',
            neighborhood='0',
            fusion='0',
            homology='0',
            experimental='200',
            database='0',
            textmining='100',
        ),
    ]
    df = spark.createDataFrame(data, STRINGS_SCHEMA)
    result = _transform_string_proteins(df, score_threshold=0)
    col_names = set(result.columns)
    assert 'interactorA' in col_names
    assert 'interactorB' in col_names
    assert 'interaction' in col_names
    assert 'source_info' in col_names


def test_string_proteins_source_database_is_string(spark):
    """_transform_string_proteins sets source_database to 'string'."""
    data = [
        Row(
            protein1='9606.ENSP00000001',
            protein2='9606.ENSP00000002',
            combined_score='500',
            coexpression='100',
            cooccurence='0',
            neighborhood='0',
            fusion='0',
            homology='0',
            experimental='200',
            database='0',
            textmining='100',
        ),
    ]
    df = spark.createDataFrame(data, STRINGS_SCHEMA)
    result = _transform_string_proteins(df, score_threshold=0)
    row = result.collect()[0]
    assert row.source_info.source_database == 'string'
