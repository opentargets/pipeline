"""Tests for the search pyspark module.

Ported from platform-etl-backend Search step.
"""

from pyspark.sql import Row
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.search import (
    _build_disease_index,
    _build_drug_index,
    _build_nct_map,
    _build_study_index,
    _build_target_index,
    _build_variant_index,
    _flatten_cat,
    _resolve_ta_labels,
)

# ---------------------------------------------------------------------------
# 1. _flatten_cat
# ---------------------------------------------------------------------------

_FC_SCHEMA = StructType([
    StructField('a', ArrayType(StringType())),
    StructField('b', ArrayType(StringType())),
    StructField('c', StringType()),
])


def test_flatten_cat_combines_arrays(spark):
    """_flatten_cat merges multiple array columns into one array."""
    data = [Row(a=['x', 'y'], b=['z'], c='w')]
    df = spark.createDataFrame(data, _FC_SCHEMA)
    df = df.withColumn('result', _flatten_cat('a', 'b', 'array(c)'))
    row = df.collect()[0]
    assert set(row.result) == {'x', 'y', 'z', 'w'}


def test_flatten_cat_deduplicates(spark):
    """_flatten_cat removes duplicate values."""
    data = [Row(a=['x', 'x'], b=['x'], c='x')]
    df = spark.createDataFrame(data, _FC_SCHEMA)
    df = df.withColumn('result', _flatten_cat('a', 'b', 'array(c)'))
    row = df.collect()[0]
    assert row.result.count('x') == 1


def test_flatten_cat_removes_nulls(spark):
    """_flatten_cat filters out null values."""
    data = [Row(a=['x', None], b=[None], c=None)]
    df = spark.createDataFrame(data, _FC_SCHEMA)
    df = df.withColumn('result', _flatten_cat('a', 'b', 'array(c)'))
    row = df.collect()[0]
    assert None not in row.result


def test_flatten_cat_strips_commas(spark):
    """_flatten_cat strips commas from values."""
    data = [Row(a=['hello, world'], b=[], c=None)]
    df = spark.createDataFrame(data, _FC_SCHEMA)
    df = df.withColumn('result', _flatten_cat('a', 'b', 'array(c)'))
    row = df.collect()[0]
    assert 'hello world' in row.result


# ---------------------------------------------------------------------------
# 2. _resolve_ta_labels
# ---------------------------------------------------------------------------

_DISEASE_SCHEMA = StructType([
    StructField('diseaseId', StringType()),
    StructField('name', StringType()),
    StructField('therapeuticAreas', ArrayType(StringType())),
])


def test_resolve_ta_labels_adds_column(spark):
    """_resolve_ta_labels adds a therapeutic_labels column."""
    data = [
        Row(diseaseId='D1', name='Disease A', therapeuticAreas=['D2']),
        Row(diseaseId='D2', name='Therapeutic Area', therapeuticAreas=[]),
    ]
    df = spark.createDataFrame(data, _DISEASE_SCHEMA)
    result = _resolve_ta_labels(df, 'diseaseId', 'therapeutic_labels')
    assert 'therapeutic_labels' in result.columns


def test_resolve_ta_labels_resolves_names(spark):
    """_resolve_ta_labels resolves therapeutic area IDs to names."""
    data = [
        Row(diseaseId='D1', name='Disease A', therapeuticAreas=['D2']),
        Row(diseaseId='D2', name='Therapy Area', therapeuticAreas=[]),
    ]
    df = spark.createDataFrame(data, _DISEASE_SCHEMA)
    result = _resolve_ta_labels(df, 'diseaseId', 'therapeutic_labels')
    row = [r for r in result.collect() if r.diseaseId == 'D1'][0]
    assert 'Therapy Area' in row.therapeutic_labels


# ---------------------------------------------------------------------------
# 3. _build_target_index
# ---------------------------------------------------------------------------

_TARGET_SCHEMA = StructType([
    StructField('targetId', StringType()),
    StructField('approvedSymbol', StringType()),
    StructField('approvedName', StringType()),
    StructField('biotype', StringType()),
    StructField('synonyms', ArrayType(StructType([StructField('label', StringType())]))),
    StructField('proteinIds', ArrayType(StructType([StructField('id', StringType())]))),
    StructField(
        'dbXRefs',
        ArrayType(
            StructType([
                StructField('source', StringType()),
                StructField('id', StringType()),
            ])
        ),
    ),
])

_ASSOC_SCHEMA = StructType([
    StructField('associationId', StringType()),
    StructField('targetId', StringType()),
    StructField('diseaseId', StringType()),
    StructField('score', DoubleType()),
])

_DISEASE_LUT_SCHEMA = StructType([
    StructField('diseaseId', StringType()),
    StructField('disease_labels', ArrayType(StringType())),
    StructField('disease_name', StringType()),
    StructField('therapeutic_labels', ArrayType(StringType())),
])

_DRUG_LUT_SCHEMA = StructType([
    StructField('drugId', StringType()),
    StructField('drug_labels', ArrayType(StringType())),
])

_VARIANT_SCHEMA = StructType([
    StructField('variantId', StringType()),
    StructField('chromosome', StringType()),
    StructField('position', StringType()),
    StructField('rsIds', ArrayType(StringType())),
    StructField('hgvsId', StringType()),
    StructField('dbXrefs', ArrayType(StructType([StructField('id', StringType())]))),
    StructField(
        'transcriptConsequences',
        ArrayType(
            StructType([
                StructField('targetId', StringType()),
                StructField('consequenceScore', DoubleType()),
                StructField('distanceFromFootprint', DoubleType()),
            ])
        ),
    ),
])


def test_build_target_index_output_schema(spark):
    """_build_target_index output has expected search index columns."""
    targets = spark.createDataFrame(
        [
            Row(
                targetId='ENSG001',
                approvedSymbol='BRCA1',
                approvedName='Breast cancer 1',
                biotype='protein_coding',
                synonyms=[Row(label='BRCA1 synonym')],
                proteinIds=[Row(id='P12345')],
                dbXRefs=[Row(source='HGNC', id='1100')],
            )
        ],
        _TARGET_SCHEMA,
    )
    assocs = spark.createDataFrame([], _ASSOC_SCHEMA)
    d_lut = spark.createDataFrame([], _DISEASE_LUT_SCHEMA)
    dr_lut = spark.createDataFrame([], _DRUG_LUT_SCHEMA)
    variants = spark.createDataFrame([], _VARIANT_SCHEMA)

    result = _build_target_index(targets, assocs, d_lut, dr_lut, variants)
    cols = set(result.columns)
    assert 'id' in cols
    assert 'name' in cols
    assert 'entity' in cols
    assert 'keywords' in cols
    assert 'multiplier' in cols


def test_build_target_index_entity_is_target(spark):
    """_build_target_index sets entity field to 'target'."""
    targets = spark.createDataFrame(
        [
            Row(
                targetId='ENSG001',
                approvedSymbol='BRCA1',
                approvedName='Breast cancer 1',
                biotype='protein_coding',
                synonyms=[],
                proteinIds=[],
                dbXRefs=[],
            )
        ],
        _TARGET_SCHEMA,
    )
    assocs = spark.createDataFrame([], _ASSOC_SCHEMA)
    d_lut = spark.createDataFrame([], _DISEASE_LUT_SCHEMA)
    dr_lut = spark.createDataFrame([], _DRUG_LUT_SCHEMA)
    variants = spark.createDataFrame([], _VARIANT_SCHEMA)

    result = _build_target_index(targets, assocs, d_lut, dr_lut, variants)
    row = result.collect()[0]
    assert row.entity == 'target'


# ---------------------------------------------------------------------------
# 4. _build_variant_index
# ---------------------------------------------------------------------------


def test_build_variant_index_output_schema(spark):
    """_build_variant_index output has expected search index columns."""
    variants = spark.createDataFrame(
        [
            Row(
                variantId='1_100_A_G',
                chromosome='1',
                position='100',
                rsIds=['rs123'],
                hgvsId='1:100:A:G',
                dbXrefs=[Row(id='rs123')],
                transcriptConsequences=[],
            )
        ],
        _VARIANT_SCHEMA,
    )
    result = _build_variant_index(variants)
    cols = set(result.columns)
    assert 'id' in cols
    assert 'entity' in cols
    assert 'keywords' in cols


def test_build_variant_index_entity_is_variant(spark):
    """_build_variant_index sets entity to 'variant'."""
    variants = spark.createDataFrame(
        [
            Row(
                variantId='1_100_A_G',
                chromosome='1',
                position='100',
                rsIds=[],
                hgvsId=None,
                dbXrefs=[],
                transcriptConsequences=[],
            )
        ],
        _VARIANT_SCHEMA,
    )
    result = _build_variant_index(variants)
    row = result.collect()[0]
    assert row.entity == 'variant'


# ---------------------------------------------------------------------------
# 5. _build_drug_index
# ---------------------------------------------------------------------------

_NCT_BY_DRUG_SCHEMA = StructType([
    StructField('drugId', StringType()),
    StructField('nctIds', ArrayType(StringType())),
])

_DRUG_SCHEMA = StructType([
    StructField('drugId', StringType()),
    StructField('name', StringType()),
    StructField('description', StringType()),
    StructField('drugType', StringType()),
    StructField('synonyms', ArrayType(StringType())),
    StructField('tradeNames', ArrayType(StringType())),
    StructField('childChemblIds', ArrayType(StringType())),
    StructField(
        'crossReferences',
        ArrayType(
            StructType([
                StructField('ids', ArrayType(StringType())),
            ])
        ),
    ),
    StructField(
        'rows',
        ArrayType(
            StructType([
                StructField('mechanismOfAction', StringType()),
            ])
        ),
    ),
    StructField('indications', ArrayType(StringType())),
])

_ASSOC_DRUG_SCHEMA = StructType([
    StructField('drugId', StringType()),
    StructField('targetIds', ArrayType(StringType())),
    StructField('diseaseIds', ArrayType(StringType())),
    StructField('meanScore', DoubleType()),
    StructField('drug_relevance', DoubleType()),
])

_TARGET_LUT_SCHEMA = StructType([
    StructField('targetId', StringType()),
    StructField('target_labels', ArrayType(StringType())),
])

_DISEASE_LUT2_SCHEMA = StructType([
    StructField('diseaseId', StringType()),
    StructField('disease_labels', ArrayType(StringType())),
    StructField('therapeutic_labels', ArrayType(StringType())),
    StructField('disease_name', StringType()),
])


def test_build_drug_index_output_schema(spark):
    """_build_drug_index output has expected search index columns."""
    drugs = spark.createDataFrame(
        [
            Row(
                drugId='CHEMBL1',
                name='Aspirin',
                description='Pain reliever',
                drugType='Small molecule',
                synonyms=['ASA'],
                tradeNames=['Bayer'],
                childChemblIds=[],
                crossReferences=[Row(ids=['CID12345'])],
                rows=[Row(mechanismOfAction='COX inhibitor')],
                indications=['D001'],
            )
        ],
        _DRUG_SCHEMA,
    )
    assoc_drugs = spark.createDataFrame([], _ASSOC_DRUG_SCHEMA)
    t_lut = spark.createDataFrame([], _TARGET_LUT_SCHEMA)
    d_lut = spark.createDataFrame([], _DISEASE_LUT2_SCHEMA)

    result = _build_drug_index(drugs, assoc_drugs, t_lut, d_lut, spark.createDataFrame([], _NCT_BY_DRUG_SCHEMA))
    cols = set(result.columns)
    assert 'id' in cols
    assert 'entity' in cols
    assert 'keywords' in cols
    assert 'multiplier' in cols


def test_build_drug_index_entity_is_drug(spark):
    """_build_drug_index sets entity to 'drug'."""
    drugs = spark.createDataFrame(
        [
            Row(
                drugId='CHEMBL1',
                name='Aspirin',
                description='Pain reliever',
                drugType='Small molecule',
                synonyms=[],
                tradeNames=[],
                childChemblIds=[],
                crossReferences=[],
                rows=[],
                indications=[],
            )
        ],
        _DRUG_SCHEMA,
    )
    assoc_drugs = spark.createDataFrame([], _ASSOC_DRUG_SCHEMA)
    t_lut = spark.createDataFrame([], _TARGET_LUT_SCHEMA)
    d_lut = spark.createDataFrame([], _DISEASE_LUT2_SCHEMA)

    result = _build_drug_index(drugs, assoc_drugs, t_lut, d_lut, spark.createDataFrame([], _NCT_BY_DRUG_SCHEMA))
    row = result.collect()[0]
    assert row.entity == 'drug'


# ---------------------------------------------------------------------------
# 6. _build_study_index
# ---------------------------------------------------------------------------

_STUDY_SCHEMA = StructType([
    StructField('studyId', StringType()),
    StructField('traitFromSource', StringType()),
    StructField('pubmedId', StringType()),
    StructField('publicationFirstAuthor', StringType()),
    StructField('diseaseIds', ArrayType(StringType())),
    StructField('nSamples', DoubleType()),
    StructField('geneId', StringType()),
])

_CREDIBLE_SET_SCHEMA = StructType([
    StructField('studyId', StringType()),
    StructField('credibleSetCount', DoubleType()),
])

_TARGET2_SCHEMA = StructType([
    StructField('targetId', StringType()),
    StructField('approvedSymbol', StringType()),
])


def test_build_study_index_output_schema(spark):
    """_build_study_index output has expected search index columns."""
    studies = spark.createDataFrame(
        [
            Row(
                studyId='GCST001',
                traitFromSource='Blood pressure',
                pubmedId='12345678',
                publicationFirstAuthor='Smith J',
                diseaseIds=['D001'],
                nSamples=1000.0,
                geneId='ENSG001',
            )
        ],
        _STUDY_SCHEMA,
    )
    targets = spark.createDataFrame([Row(targetId='ENSG001', approvedSymbol='BRCA1')], _TARGET2_SCHEMA)
    cred_sets = spark.createDataFrame([], _CREDIBLE_SET_SCHEMA)

    result = _build_study_index(studies, targets, cred_sets)
    cols = set(result.columns)
    assert 'id' in cols
    assert 'entity' in cols
    assert 'keywords' in cols


def test_build_study_index_entity_is_study(spark):
    """_build_study_index sets entity to 'study'."""
    studies = spark.createDataFrame(
        [
            Row(
                studyId='GCST001',
                traitFromSource='Blood pressure',
                pubmedId=None,
                publicationFirstAuthor=None,
                diseaseIds=[],
                nSamples=None,
                geneId=None,
            )
        ],
        _STUDY_SCHEMA,
    )
    targets = spark.createDataFrame([], _TARGET2_SCHEMA)
    cred_sets = spark.createDataFrame([], _CREDIBLE_SET_SCHEMA)

    result = _build_study_index(studies, targets, cred_sets)
    row = result.collect()[0]
    assert row.entity == 'study'


# ---------------------------------------------------------------------------
# 7. _build_nct_map
# ---------------------------------------------------------------------------

_INDICATION_SCHEMA = StructType([
    StructField('drugId', StringType()),
    StructField('diseaseId', StringType()),
    StructField('clinicalReportIds', ArrayType(StringType())),
])

_NCT_BY_DISEASE_SCHEMA = StructType([
    StructField('diseaseId', StringType()),
    StructField('nctIds', ArrayType(StringType())),
])


def test_build_nct_map_drops_rows_without_nct_ids(spark):
    """_build_nct_map excludes rows where no report ID starts with 'nct'."""
    data = [
        Row(drugId='CHEMBL1', diseaseId='D1', clinicalReportIds=['uuid-abc']),
        Row(drugId='CHEMBL2', diseaseId='D2', clinicalReportIds=['nct001']),
    ]
    df = spark.createDataFrame(data, _INDICATION_SCHEMA)
    result = _build_nct_map(df)
    assert result.count() == 1
    assert result.collect()[0].drugId == 'CHEMBL2'


# ---------------------------------------------------------------------------
# 8. NCT IDs in drug keywords
# ---------------------------------------------------------------------------


def test_build_drug_index_nct_ids_appear_in_keywords(spark):
    """_build_drug_index puts NCT IDs from nct_by_drug into keywords."""
    drugs = spark.createDataFrame(
        [
            Row(
                drugId='CHEMBL1',
                name='Aspirin',
                description='Pain reliever',
                drugType='Small molecule',
                synonyms=[],
                tradeNames=[],
                childChemblIds=[],
                crossReferences=[],
                rows=[],
                indications=[],
            )
        ],
        _DRUG_SCHEMA,
    )
    nct_by_drug = spark.createDataFrame(
        [Row(drugId='CHEMBL1', nctIds=['nct00001234', 'nct00005678'])],
        _NCT_BY_DRUG_SCHEMA,
    )

    result = _build_drug_index(
        drugs,
        spark.createDataFrame([], _ASSOC_DRUG_SCHEMA),
        spark.createDataFrame([], _TARGET_LUT_SCHEMA),
        spark.createDataFrame([], _DISEASE_LUT2_SCHEMA),
        nct_by_drug,
    )
    keywords = result.collect()[0].keywords
    assert 'nct00001234' in keywords
    assert 'nct00005678' in keywords


def test_build_drug_index_missing_nct_ids_yields_no_crash(spark):
    """_build_drug_index works correctly when a drug has no NCT IDs."""
    drugs = spark.createDataFrame(
        [
            Row(
                drugId='CHEMBL1',
                name='Aspirin',
                description='Pain reliever',
                drugType='Small molecule',
                synonyms=[],
                tradeNames=[],
                childChemblIds=[],
                crossReferences=[],
                rows=[],
                indications=[],
            )
        ],
        _DRUG_SCHEMA,
    )

    result = _build_drug_index(
        drugs,
        spark.createDataFrame([], _ASSOC_DRUG_SCHEMA),
        spark.createDataFrame([], _TARGET_LUT_SCHEMA),
        spark.createDataFrame([], _DISEASE_LUT2_SCHEMA),
        spark.createDataFrame([], _NCT_BY_DRUG_SCHEMA),
    )
    assert result.count() == 1


# ---------------------------------------------------------------------------
# 9. NCT IDs in disease keywords
# ---------------------------------------------------------------------------

_DISEASE_INDEX_SCHEMA = StructType([
    StructField('diseaseId', StringType()),
    StructField('name', StringType()),
    StructField('description', StringType()),
    StructField('therapeutic_labels', ArrayType(StringType())),
    StructField('therapeuticAreas', ArrayType(StringType())),
    StructField(
        'synonyms',
        StructType([
            StructField('hasBroadSynonym', ArrayType(StringType())),
            StructField('hasExactSynonym', ArrayType(StringType())),
            StructField('hasNarrowSynonym', ArrayType(StringType())),
            StructField('hasRelatedSynonym', ArrayType(StringType())),
        ]),
    ),
])

_PHENOTYPE_SCHEMA = StructType([
    StructField('diseaseId', StringType()),
    StructField('phenotype_labels', ArrayType(StringType())),
])

_ASSOC_DRUG_DISEASE_SCHEMA = StructType([
    StructField('associationId', StringType()),
    StructField('drugId', StringType()),
    StructField('drugIds', ArrayType(StringType())),
    StructField('targetId', StringType()),
    StructField('diseaseId', StringType()),
    StructField('score', DoubleType()),
])

_STUDIES_SCHEMA = StructType([
    StructField('studyId', StringType()),
    StructField('diseaseIds', ArrayType(StringType())),
])


def _empty_disease_index_inputs(spark):
    return (
        spark.createDataFrame([], _PHENOTYPE_SCHEMA),
        spark.createDataFrame([], _ASSOC_SCHEMA),
        spark.createDataFrame([], _ASSOC_DRUG_DISEASE_SCHEMA),
        spark.createDataFrame([], _TARGET_LUT_SCHEMA),
        spark.createDataFrame([], _DRUG_LUT_SCHEMA),
        spark.createDataFrame([], _STUDIES_SCHEMA),
    )


def test_build_disease_index_nct_ids_appear_in_keywords(spark):
    """_build_disease_index puts NCT IDs from nct_by_disease into keywords."""
    diseases = spark.createDataFrame(
        [
            Row(
                diseaseId='EFO_001',
                name='Cancer',
                description='A disease',
                therapeutic_labels=[],
                therapeuticAreas=[],
                synonyms=Row(hasBroadSynonym=[], hasExactSynonym=[], hasNarrowSynonym=[], hasRelatedSynonym=[]),
            )
        ],
        _DISEASE_INDEX_SCHEMA,
    )
    nct_by_disease = spark.createDataFrame(
        [Row(diseaseId='EFO_001', nctIds=['nct00001234'])],
        _NCT_BY_DISEASE_SCHEMA,
    )
    phenotypes, assocs, assoc_drugs, t_lut, dr_lut, studies = _empty_disease_index_inputs(spark)

    result = _build_disease_index(diseases, phenotypes, assocs, assoc_drugs, t_lut, dr_lut, studies, nct_by_disease)
    keywords = result.collect()[0].keywords
    assert 'nct00001234' in keywords


def test_build_disease_index_missing_nct_ids_yields_no_crash(spark):
    """_build_disease_index works correctly when a disease has no NCT IDs."""
    diseases = spark.createDataFrame(
        [
            Row(
                diseaseId='EFO_001',
                name='Cancer',
                description='A disease',
                therapeutic_labels=[],
                therapeuticAreas=[],
                synonyms=Row(hasBroadSynonym=[], hasExactSynonym=[], hasNarrowSynonym=[], hasRelatedSynonym=[]),
            )
        ],
        _DISEASE_INDEX_SCHEMA,
    )
    phenotypes, assocs, assoc_drugs, t_lut, dr_lut, studies = _empty_disease_index_inputs(spark)

    result = _build_disease_index(
        diseases,
        phenotypes,
        assocs,
        assoc_drugs,
        t_lut,
        dr_lut,
        studies,
        spark.createDataFrame([], _NCT_BY_DISEASE_SCHEMA),
    )
    assert result.count() == 1
