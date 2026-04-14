"""Tests for the search_facet pyspark module.

Ported from platform-etl-backend searchFacet step.
"""

from pyspark.sql import Row
from pyspark.sql.types import ArrayType, BooleanType, LongType, StringType, StructField, StructType

from pts.pyspark.search_facet import (
    _compute_disease_name_facets,
    _compute_go_facets,
    _compute_pathway_facets,
    _compute_simple_facet,
    _compute_subcellular_location_facets,
    _compute_target_class_facets,
    _compute_therapeutic_areas_facets,
    _compute_tractability_facets,
)

# ---------------------------------------------------------------------------
# Category label constants (mirrors reference.conf categories)
# ---------------------------------------------------------------------------

CATEGORIES = {
    'disease_name': 'Disease',
    'therapeutic_area': 'Therapeutic Area',
    'sm': 'Tractability Small Molecule',
    'ab': 'Tractability Antibody',
    'pr': 'Tractability PROTAC',
    'oc': 'Tractability Other Modalities',
    'target_id': 'Target ID',
    'approved_symbol': 'Approved Symbol',
    'approved_name': 'Approved Name',
    'subcellular_location': 'Subcellular Location',
    'target_class': 'ChEMBL Target Class',
    'pathways': 'Reactome',
    'go_f': 'GO:MF',
    'go_p': 'GO:BP',
    'go_c': 'GO:CC',
}

# ---------------------------------------------------------------------------
# Disease schemas
# ---------------------------------------------------------------------------

DISEASE_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('name', StringType()),
    StructField('therapeuticAreas', ArrayType(StringType())),
])


# ---------------------------------------------------------------------------
# 1. Disease name facet
# ---------------------------------------------------------------------------


def test_disease_name_facet_produces_name_category(spark):
    """Disease name facet produces rows with 'Disease' category."""
    data = [
        Row(id='EFO_0000001', name='breast cancer', therapeuticAreas=['EFO_0010285']),
        Row(id='EFO_0000002', name='lung cancer', therapeuticAreas=[]),
    ]
    df = spark.createDataFrame(data, DISEASE_SCHEMA)
    result = _compute_disease_name_facets(df, CATEGORIES)
    rows = result.collect()
    categories = {r.category for r in rows}
    assert 'Disease' in categories
    labels = {r.label for r in rows}
    assert 'breast cancer' in labels
    assert 'lung cancer' in labels


def test_disease_name_facet_entry_structure(spark):
    """Disease name facet entries have label, category, entityIds, datasourceId."""
    data = [Row(id='EFO_0000001', name='breast cancer', therapeuticAreas=[])]
    df = spark.createDataFrame(data, DISEASE_SCHEMA)
    result = _compute_disease_name_facets(df, CATEGORIES)
    row = result.collect()[0]
    assert hasattr(row, 'label')
    assert hasattr(row, 'category')
    assert hasattr(row, 'entityIds')
    assert hasattr(row, 'datasourceId')
    assert row.category == 'Disease'
    assert 'EFO_0000001' in row.entityIds


def test_disease_name_facet_datasource_id_is_disease_id(spark):
    """Disease name facet datasourceId equals the disease id."""
    data = [Row(id='EFO_0000001', name='breast cancer', therapeuticAreas=[])]
    df = spark.createDataFrame(data, DISEASE_SCHEMA)
    result = _compute_disease_name_facets(df, CATEGORIES)
    row = result.collect()[0]
    assert row.datasourceId == 'EFO_0000001'


# ---------------------------------------------------------------------------
# 2. Therapeutic area facet
# ---------------------------------------------------------------------------


def test_therapeutic_area_facet_produces_therapeutic_area_category(spark):
    """Therapeutic areas facet produces rows with 'Therapeutic Area' category."""
    data = [
        Row(id='EFO_0000010', name='cancer', therapeuticAreas=[]),
        Row(id='EFO_0000001', name='breast cancer', therapeuticAreas=['EFO_0000010']),
    ]
    df = spark.createDataFrame(data, DISEASE_SCHEMA)
    result = _compute_therapeutic_areas_facets(df, CATEGORIES)
    rows = result.collect()
    categories = {r.category for r in rows}
    assert 'Therapeutic Area' in categories


def test_therapeutic_area_facet_entry_structure(spark):
    """Therapeutic area facet entries have label, category, entityIds, datasourceId."""
    data = [
        Row(id='EFO_0000010', name='neoplasm', therapeuticAreas=[]),
        Row(id='EFO_0000001', name='breast cancer', therapeuticAreas=['EFO_0000010']),
    ]
    df = spark.createDataFrame(data, DISEASE_SCHEMA)
    result = _compute_therapeutic_areas_facets(df, CATEGORIES)
    rows = result.collect()
    assert len(rows) >= 1
    row = rows[0]
    assert hasattr(row, 'label')
    assert hasattr(row, 'category')
    assert hasattr(row, 'entityIds')
    assert hasattr(row, 'datasourceId')


def test_therapeutic_area_label_is_area_name(spark):
    """Therapeutic area facet label is the name of the therapeutic area disease."""
    data = [
        Row(id='EFO_0000010', name='neoplasm', therapeuticAreas=[]),
        Row(id='EFO_0000001', name='breast cancer', therapeuticAreas=['EFO_0000010']),
    ]
    df = spark.createDataFrame(data, DISEASE_SCHEMA)
    result = _compute_therapeutic_areas_facets(df, CATEGORIES)
    rows = result.collect()
    labels = {r.label for r in rows}
    assert 'neoplasm' in labels


# ---------------------------------------------------------------------------
# 3. Tractability facets
# ---------------------------------------------------------------------------

TRACTABILITY_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField(
        'tractability',
        ArrayType(
            StructType([
                StructField('modality', StringType()),
                StructField('id', StringType()),
                StructField('value', BooleanType()),
            ])
        ),
    ),
])


def test_tractability_facet_produces_modality_categories(spark):
    """Tractability facet produces SM/AB/PR/OC modality categories."""
    data = [
        Row(
            id='ENSG00000001',
            tractability=[
                Row(modality='SM', id='Clinical Precedence', value=True),
                Row(modality='AB', id='Predicted Tractable High', value=True),
            ],
        ),
    ]
    df = spark.createDataFrame(data, TRACTABILITY_SCHEMA)
    result = _compute_tractability_facets(df, CATEGORIES)
    rows = result.collect()
    categories = {r.category for r in rows}
    assert 'Tractability Small Molecule' in categories
    assert 'Tractability Antibody' in categories


def test_tractability_facet_only_true_values(spark):
    """Tractability facet only includes entries where value=True."""
    data = [
        Row(
            id='ENSG00000001',
            tractability=[
                Row(modality='SM', id='Clinical Precedence', value=True),
                Row(modality='SM', id='Discovery Precedence', value=False),
            ],
        ),
    ]
    df = spark.createDataFrame(data, TRACTABILITY_SCHEMA)
    result = _compute_tractability_facets(df, CATEGORIES)
    rows = result.collect()
    labels = {r.label for r in rows}
    assert 'Clinical Precedence' in labels
    assert 'Discovery Precedence' not in labels


def test_tractability_facet_entry_structure(spark):
    """Tractability facet entries have label, category, entityIds, datasourceId."""
    data = [
        Row(
            id='ENSG00000001',
            tractability=[Row(modality='SM', id='Clinical Precedence', value=True)],
        ),
    ]
    df = spark.createDataFrame(data, TRACTABILITY_SCHEMA)
    result = _compute_tractability_facets(df, CATEGORIES)
    row = result.collect()[0]
    assert hasattr(row, 'label')
    assert hasattr(row, 'category')
    assert hasattr(row, 'entityIds')
    assert hasattr(row, 'datasourceId')


# ---------------------------------------------------------------------------
# 4. GO facets
# ---------------------------------------------------------------------------

TARGET_GO_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField(
        'go',
        ArrayType(
            StructType([
                StructField('id', StringType()),
                StructField('aspect', StringType()),
            ])
        ),
    ),
])

GO_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('label', StringType()),
])


def test_go_facet_produces_f_p_c_categories(spark):
    """GO facet maps F/P/C aspects to GO:MF / GO:BP / GO:CC categories."""
    target_data = [
        Row(
            id='ENSG00000001',
            go=[
                Row(id='GO:0003677', aspect='F'),
                Row(id='GO:0008150', aspect='P'),
                Row(id='GO:0005634', aspect='C'),
            ],
        ),
    ]
    go_data = [
        Row(id='GO:0003677', label='DNA binding'),
        Row(id='GO:0008150', label='biological_process'),
        Row(id='GO:0005634', label='nucleus'),
    ]
    target_df = spark.createDataFrame(target_data, TARGET_GO_SCHEMA)
    go_df = spark.createDataFrame(go_data, GO_SCHEMA)
    result = _compute_go_facets(target_df, go_df, CATEGORIES)
    rows = result.collect()
    categories = {r.category for r in rows}
    assert 'GO:MF' in categories
    assert 'GO:BP' in categories
    assert 'GO:CC' in categories


def test_go_facet_entry_structure(spark):
    """GO facet entries have label, category, entityIds, datasourceId."""
    target_data = [Row(id='ENSG00000001', go=[Row(id='GO:0003677', aspect='F')])]
    go_data = [Row(id='GO:0003677', label='DNA binding')]
    target_df = spark.createDataFrame(target_data, TARGET_GO_SCHEMA)
    go_df = spark.createDataFrame(go_data, GO_SCHEMA)
    result = _compute_go_facets(target_df, go_df, CATEGORIES)
    row = result.collect()[0]
    assert hasattr(row, 'label')
    assert hasattr(row, 'category')
    assert hasattr(row, 'entityIds')
    assert hasattr(row, 'datasourceId')
    assert row.datasourceId == 'GO:0003677'


# ---------------------------------------------------------------------------
# 5. Subcellular location facets
# ---------------------------------------------------------------------------

SUBCELLULAR_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField(
        'subcellularLocations',
        ArrayType(
            StructType([
                StructField('location', StringType()),
                StructField('termSl', StringType()),
            ])
        ),
    ),
])


def test_subcellular_location_facet_category(spark):
    """Subcellular location facet produces 'Subcellular Location' category."""
    data = [
        Row(
            id='ENSG00000001',
            subcellularLocations=[Row(location='nucleus', termSl='SL-0191')],
        ),
    ]
    df = spark.createDataFrame(data, SUBCELLULAR_SCHEMA)
    result = _compute_subcellular_location_facets(df, CATEGORIES)
    rows = result.collect()
    categories = {r.category for r in rows}
    assert 'Subcellular Location' in categories


def test_subcellular_location_facet_entry_structure(spark):
    """Subcellular location facet entries have label, category, entityIds, datasourceId."""
    data = [
        Row(
            id='ENSG00000001',
            subcellularLocations=[Row(location='nucleus', termSl='SL-0191')],
        ),
    ]
    df = spark.createDataFrame(data, SUBCELLULAR_SCHEMA)
    result = _compute_subcellular_location_facets(df, CATEGORIES)
    row = result.collect()[0]
    assert hasattr(row, 'label')
    assert hasattr(row, 'category')
    assert hasattr(row, 'entityIds')
    assert hasattr(row, 'datasourceId')
    assert row.label == 'nucleus'
    assert row.datasourceId == 'SL-0191'


# ---------------------------------------------------------------------------
# 6. Target class facets
# ---------------------------------------------------------------------------

TARGET_CLASS_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField(
        'targetClass',
        ArrayType(
            StructType([
                StructField('id', LongType()),
                StructField('label', StringType()),
                StructField('level', StringType()),
            ])
        ),
    ),
])


def test_target_class_facet_category(spark):
    """Target class facet produces 'ChEMBL Target Class' category."""
    data = [
        Row(
            id='ENSG00000001',
            targetClass=[Row(id=1, label='Kinase', level='l1')],
        ),
    ]
    df = spark.createDataFrame(data, TARGET_CLASS_SCHEMA)
    result = _compute_target_class_facets(df, CATEGORIES)
    rows = result.collect()
    categories = {r.category for r in rows}
    assert 'ChEMBL Target Class' in categories


def test_target_class_facet_entry_structure(spark):
    """Target class facet entries have label, category, entityIds, datasourceId (null)."""
    data = [
        Row(
            id='ENSG00000001',
            targetClass=[Row(id=1, label='Kinase', level='l1')],
        ),
    ]
    df = spark.createDataFrame(data, TARGET_CLASS_SCHEMA)
    result = _compute_target_class_facets(df, CATEGORIES)
    row = result.collect()[0]
    assert hasattr(row, 'label')
    assert hasattr(row, 'category')
    assert hasattr(row, 'entityIds')
    assert hasattr(row, 'datasourceId')
    assert row.label == 'Kinase'
    assert row.datasourceId is None


# ---------------------------------------------------------------------------
# 7. Pathway facets
# ---------------------------------------------------------------------------

PATHWAY_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField(
        'pathways',
        ArrayType(
            StructType([
                StructField('pathwayId', StringType()),
                StructField('pathway', StringType()),
                StructField('topLevelTerm', StringType()),
            ])
        ),
    ),
])


def test_pathway_facet_category(spark):
    """Pathway facet produces 'Reactome' category."""
    data = [
        Row(
            id='ENSG00000001',
            pathways=[Row(pathwayId='R-HSA-1', pathway='Cell Cycle', topLevelTerm='Cell Cycle')],
        ),
    ]
    df = spark.createDataFrame(data, PATHWAY_SCHEMA)
    result = _compute_pathway_facets(df, CATEGORIES)
    rows = result.collect()
    categories = {r.category for r in rows}
    assert 'Reactome' in categories


def test_pathway_facet_entry_structure(spark):
    """Pathway facet entries have label, category, entityIds, datasourceId."""
    data = [
        Row(
            id='ENSG00000001',
            pathways=[Row(pathwayId='R-HSA-1', pathway='Cell Cycle', topLevelTerm='Cell Cycle')],
        ),
    ]
    df = spark.createDataFrame(data, PATHWAY_SCHEMA)
    result = _compute_pathway_facets(df, CATEGORIES)
    row = result.collect()[0]
    assert hasattr(row, 'label')
    assert hasattr(row, 'category')
    assert hasattr(row, 'entityIds')
    assert hasattr(row, 'datasourceId')
    assert row.label == 'Cell Cycle'
    assert row.datasourceId == 'R-HSA-1'


# ---------------------------------------------------------------------------
# 8. Simple facet
# ---------------------------------------------------------------------------

SIMPLE_FACET_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('approvedSymbol', StringType()),
])


class TestComputeSimpleFacet:
    """Tests for _compute_simple_facet."""

    def test_simple_facet_produces_expected_columns(self, spark):
        """Simple facet output has label, category, entityIds, datasourceId."""
        data = [
            Row(id='ENSG00000001', approvedSymbol='BRCA1'),
            Row(id='ENSG00000002', approvedSymbol='TP53'),
        ]
        df = spark.createDataFrame(data, SIMPLE_FACET_SCHEMA)
        result = _compute_simple_facet(df, 'approvedSymbol', 'Approved Symbol', 'id')
        row_dict = {r.label: r for r in result.collect()}
        assert 'BRCA1' in row_dict
        assert 'TP53' in row_dict

    def test_simple_facet_category_value(self, spark):
        """Simple facet uses the provided category string."""
        data = [Row(id='ENSG00000001', approvedSymbol='BRCA1')]
        df = spark.createDataFrame(data, SIMPLE_FACET_SCHEMA)
        result = _compute_simple_facet(df, 'approvedSymbol', 'Target ID', 'id')
        row = result.collect()[0]
        assert row.category == 'Target ID'

    def test_simple_facet_entity_ids_collected(self, spark):
        """Simple facet collects entity IDs for the same label."""
        data = [
            Row(id='ENSG00000001', approvedSymbol='BRCA1'),
        ]
        df = spark.createDataFrame(data, SIMPLE_FACET_SCHEMA)
        result = _compute_simple_facet(df, 'approvedSymbol', 'Approved Symbol', 'id')
        row = result.collect()[0]
        assert row.label == 'BRCA1'
        assert 'ENSG00000001' in row.entityIds

    def test_simple_facet_datasource_id_is_null(self, spark):
        """Simple facet datasourceId is null."""
        data = [Row(id='ENSG00000001', approvedSymbol='BRCA1')]
        df = spark.createDataFrame(data, SIMPLE_FACET_SCHEMA)
        result = _compute_simple_facet(df, 'approvedSymbol', 'Approved Symbol', 'id')
        row = result.collect()[0]
        assert row.datasourceId is None
