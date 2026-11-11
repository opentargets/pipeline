"""Tests of the VIDRA analysis-ready table (`pts.pyspark.vidra_input`)."""

from __future__ import annotations

import math

import polars as pl
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import types as t

from pts.pyspark import vidra_input as vi
from pts.pyspark.vidra_utils.constants import (
    ANNOTATION_COLUMNS,
    DEFAULT_BO,
    DEFAULT_BOSE,
    PROVENANCE_COLUMNS,
    SOURCE_AZ_RARE,
    SOURCE_CLINVAR,
    SOURCE_CODING_GWAS,
    SOURCE_COMMON_QTL,
)

SETTINGS = vi.VidraInputSettings(gpn_star={'a': -4.9, 'b': -0.65})

LEAD_COLUMNS = [
    'credibleSetId',
    'studyId',
    'studyType',
    'leadVariantId',
    'log10PValue',
    'reportedBeta',
    'reportedStandardError',
    'rescaledBeta',
    'rescaledStandardError',
]


@pytest.fixture(scope='module')
def genes(spark: SparkSession):
    return spark.createDataFrame([('g1', 'GENE1'), ('g2', 'GENE2')], ['geneId', 'approvedSymbol'])


@pytest.fixture(scope='module')
def obsolete(spark: SparkSession):
    return spark.createDataFrame([('EFO_OLD', 'EFO_1')], ['obsoleteId', 'diseaseId'])


@pytest.fixture(scope='module')
def leads(spark: SparkSession):
    return spark.createDataFrame(
        [
            ('gwas_a', 's_gwas', 'gwas', '1_10_A_G', -12.0, 0.4, 0.05, 0.3, 0.04),
            ('gwas_b', 's_gwas', 'gwas', '1_20_C_T', -9.0, -0.2, 0.05, -0.15, 0.04),
            ('gwas_weak', 's_gwas2', 'gwas', '1_30_G_A', -5.0, 0.1, 0.05, 0.1, 0.04),
            ('gwas_nobeta', 's_gwas', 'gwas', '1_40_T_C', -10.0, None, None, None, None),
            ('eqtl_a', 's_eqtl', 'eqtl', '1_11_A_G', -20.0, 0.8, 0.1, None, None),
            ('eqtl_b', 's_eqtl_liver', 'eqtl', '1_12_A_G', -20.0, 0.9, 0.1, None, None),
            ('pqtl_a', 's_pqtl', 'pqtl', '1_10_A_G', -15.0, -0.5, 0.1, None, None),
            ('sqtl_a', 's_sqtl', 'sqtl', '1_10_A_G', -15.0, 0.5, 0.1, None, None),
        ],
        LEAD_COLUMNS,
    )


@pytest.fixture(scope='module')
def study(spark: SparkSession):
    schema = t.StructType([
        t.StructField('studyId', t.StringType()),
        t.StructField('studyType', t.StringType()),
        t.StructField('geneId', t.StringType()),
        t.StructField('biosampleId', t.StringType()),
        t.StructField('diseaseIds', t.ArrayType(t.StringType())),
    ])
    return spark.createDataFrame(
        [
            ('s_gwas', 'gwas', None, None, ['EFO_OLD', 'EFO_2']),
            ('s_gwas2', 'gwas', None, None, ['EFO_2']),
            ('s_eqtl', 'eqtl', 'g1', 'UBERON_0000178', []),
            ('s_eqtl_liver', 'eqtl', 'g1', 'UBERON_0002107', []),
            ('s_pqtl', 'pqtl', 'g1', 'UBERON_0001969', []),
            ('s_sqtl', 'sqtl', 'g1', 'UBERON_0000178', []),
        ],
        schema,
    )


@pytest.fixture(scope='module')
def blood(spark: SparkSession):
    return spark.createDataFrame([('UBERON_0000178',), ('UBERON_0001969',)], ['biosampleId'])


def test_common_variants_take_both_orientations_filter_tissue_and_dedup(spark, leads, study, genes, obsolete, blood):
    diseases = vi.study_diseases(study, obsolete)
    qtl = vi.qtl_credible_sets(leads, study, genes, blood, SETTINGS)
    gwas = vi.gwas_leads(leads)
    coloc = spark.createDataFrame(
        [
            ('gwas_a', 'eqtl_a', 0.9, -1.0),  # GWAS on the left
            ('pqtl_a', 'gwas_a', 0.95, 1.0),  # GWAS on the right
            ('gwas_a', 'eqtl_b', 0.9, None),  # liver eQTL: dropped by the tissue filter
            ('gwas_b', 'eqtl_a', 0.5, None),  # H4 too low
            ('gwas_a', 'sqtl_a', 0.99, None),  # sQTL: not a QTL type of VIDRA
            ('gwas_nobeta', 'eqtl_a', 0.99, None),  # GWAS lead without a rescaled effect
        ],
        ['leftLocusId', 'rightLocusId', 'h4', 'betaRatioSignAverage'],
    )
    rows = vi.common_variants(coloc, gwas, qtl, diseases, SETTINGS)
    frame = pl.from_pandas(rows.toPandas()).sort('as_disease', 'GqtlLab')
    assert frame['GsourceLab'].unique().to_list() == [SOURCE_COMMON_QTL]
    assert frame.select('variant', 'as_gene', 'as_disease', 'GqtlLab', 'xc', 'qtlVariantId').rows() == [
        ('1_10_A_G', 'g1', 'EFO_1', 0, 0.8, '1_11_A_G'),
        ('1_10_A_G', 'g1', 'EFO_1', 1, -0.5, '1_10_A_G'),
        ('1_10_A_G', 'g1', 'EFO_2', 0, 0.8, '1_11_A_G'),
        ('1_10_A_G', 'g1', 'EFO_2', 1, -0.5, '1_10_A_G'),
    ]
    assert frame['yc'].unique().to_list() == [0.3]  # the rescaled GWAS effect, not the reported one
    assert frame.filter(pl.col('GqtlLab') == 1)['colocH4'].to_list() == ['0.95', '0.95']


COLOC_SCHEMA = t.StructType([
    t.StructField('leftLocusId', t.StringType()),
    t.StructField('rightLocusId', t.StringType()),
    t.StructField('h4', t.DoubleType()),
    t.StructField('betaRatioSignAverage', t.DoubleType()),
])


def test_common_variants_without_tissue_filter_keep_every_biosample(spark, leads, study, genes, obsolete):
    diseases = vi.study_diseases(study, obsolete)
    qtl = vi.qtl_credible_sets(leads, study, genes, None, SETTINGS)
    coloc = spark.createDataFrame([('gwas_a', 'eqtl_b', 0.9, None)], COLOC_SCHEMA)
    rows = vi.common_variants(coloc, vi.gwas_leads(leads), qtl, diseases, SETTINGS)
    assert rows.count() == 2  # two diseases of the study
    assert rows.select('qtlBiosampleId').distinct().collect()[0][0] == 'UBERON_0002107'


@pytest.fixture(scope='module')
def variant(spark: SparkSession):
    effect = t.StructType([
        t.StructField('method', t.StringType()),
        t.StructField('score', t.FloatType()),
        t.StructField('targetId', t.StringType()),
    ])
    transcript = t.StructType([
        t.StructField('targetId', t.StringType()),
        t.StructField('variantFunctionalConsequenceIds', t.ArrayType(t.StringType())),
        t.StructField('lofteePrediction', t.StringType()),
        t.StructField('isEnsemblCanonical', t.BooleanType()),
        t.StructField('maneSelect', t.StringType()),
        t.StructField('consequenceScore', t.FloatType()),
        t.StructField('biotype', t.StringType()),
    ])
    schema = t.StructType([
        t.StructField('variantId', t.StringType()),
        t.StructField('variantEffect', t.ArrayType(effect)),
        t.StructField('transcriptConsequences', t.ArrayType(transcript)),
    ])
    return spark.createDataFrame(
        [
            (
                '1_10_A_G',
                [('GPN-Star', -6.0, None), ('AlphaMissense', 0.9, 'g1'), ('AlphaMissense', 0.2, 'g2')],
                [
                    ('g1', ['SO_0001583'], None, True, 'NM_1', 0.66, 'protein_coding'),
                    ('g1', ['SO_0001630'], None, False, None, 0.3, 'protein_coding'),
                    ('g2', ['SO_0001627'], None, True, None, 0.1, 'protein_coding'),
                    ('g9', ['SO_0001583'], None, True, None, 0.66, 'lncRNA'),
                ],
            ),
            ('1_20_C_T', [('GPN-Star', 2.0, None)], [('g1', ['SO_0001589'], 'HC', True, None, 1.0, 'protein_coding')]),
            ('1_30_G_A', [], [('g1', ['SO_0001819'], None, False, 'NM_2', 0.2, 'protein_coding')]),
            ('1_99_T_A', None, None),
        ],
        schema,
    )


def test_gene_consequences_prefer_canonical_transcript_and_classify(variant, genes):
    frame = pl.from_pandas(vi.gene_consequences(variant, genes).toPandas()).sort('variant', 'as_gene')
    assert frame.select('variant', 'as_gene', 'consequence_class', 'is_lof').rows() == [
        ('1_10_A_G', 'g1', 'missense', False),
        ('1_10_A_G', 'g2', 'other', False),
        ('1_20_C_T', 'g1', 'lof', True),
        ('1_30_G_A', 'g1', 'other', False),
    ]
    assert frame.filter(pl.col('variant') == '1_10_A_G')['consequences'].to_list()[0] == ['SO_0001583']


def test_coding_gwas_variants_need_a_coding_consequence_significance_and_no_coloc(
    spark, leads, study, genes, obsolete, variant
):
    diseases = vi.study_diseases(study, obsolete)
    consequences = vi.gene_consequences(variant, genes)
    gwas = vi.gwas_leads(leads)
    common = spark.createDataFrame([('1_20_C_T',)], ['variant'])  # the frameshift lead colocalises elsewhere
    rows = pl.from_pandas(vi.coding_gwas_variants(gwas, consequences, diseases, common, SETTINGS).toPandas()).sort(
        'as_disease'
    )
    # 1_10_A_G is missense for g1 (kept) and intronic for g2 (not coding); 1_30_G_A is synonymous but p = 1e-5
    assert rows.select('variant', 'as_gene', 'as_disease', 'GsourceLab', 'yc', 'xc').rows() == [
        ('1_10_A_G', 'g1', 'EFO_1', SOURCE_CODING_GWAS, 0.3, 0.0),
        ('1_10_A_G', 'g1', 'EFO_2', SOURCE_CODING_GWAS, 0.3, 0.0),
    ]


def test_clinvar_variants_score_labels_and_keep_directions(spark, genes, obsolete):
    schema = t.StructType([
        t.StructField('datasourceId', t.StringType()),
        t.StructField('variantId', t.StringType()),
        t.StructField('targetId', t.StringType()),
        t.StructField('diseaseId', t.StringType()),
        t.StructField('confidence', t.StringType()),
        t.StructField('clinicalSignificances', t.ArrayType(t.StringType())),
        t.StructField('directionOnTarget', t.StringType()),
        t.StructField('directionOnTrait', t.StringType()),
    ])
    single = 'criteria provided, single submitter'
    eva = spark.createDataFrame(
        [
            ('eva', '1_1_A_G', 'g1', 'EFO_OLD', single, ['Pathogenic'], 'LoF', 'risk'),
            ('eva', '1_2_A_G', 'g1', 'EFO_2', single, ['conflicting classifications of pathogenicity'], None, None),
            ('eva', '1_3_A_G', 'g1', 'EFO_2', single, ['benign', 'risk factor'], None, None),
            ('eva', '1_4_A_G', 'g1', 'EFO_2', single, ['made up'], None, None),
            ('eva', '1_5_A_G', 'g1', 'EFO_2', 'no assertion criteria provided', ['pathogenic'], None, None),
            ('eva', '1_6_A_G', 'g9', 'EFO_2', single, ['pathogenic'], None, None),
            ('eva_somatic', '1_7_A_G', 'g1', 'EFO_2', single, ['pathogenic'], None, None),
            ('eva', None, 'g1', 'EFO_2', single, ['pathogenic'], None, None),
        ],
        schema,
    )
    rows = pl.from_pandas(vi.clinvar_variants(eva, genes, obsolete).toPandas()).sort('variant')
    assert rows.select(
        'variant', 'as_disease', 'GsourceLab', 'as_clinicalSignificance', 'clinvarDirectionOnTarget'
    ).rows() == [
        ('1_1_A_G', 'EFO_1', SOURCE_CLINVAR, 1.0, 'LoF'),
        ('1_2_A_G', 'EFO_2', SOURCE_CLINVAR, 9 / 16, None),
        ('1_3_A_G', 'EFO_2', SOURCE_CLINVAR, 13 / 16, None),
        ('1_4_A_G', 'EFO_2', SOURCE_CLINVAR, 0.0, None),
    ]


@pytest.fixture(scope='module')
def phenotype_map(spark: SparkSession):
    return spark.createDataFrame([('p1', 'EFO_OLD'), ('p1', 'EFO_3'), ('p2', 'EFO_2')], ['phenotype', 'diseaseId'])


def test_az_rare_variants_map_symbols_and_phenotypes(spark, genes, phenotype_map, obsolete):
    exwas = spark.createDataFrame(
        [
            ('1_5_C_T', 'GENE1', 'p1', 3.0, 2.0, 4.5),
            ('1_5_C_T', 'GENE1', 'p1', 3.0, 2.0, 4.5),  # duplicate row collapses
            ('1_6_C_T', 'GENE1', 'unmapped', 3.0, 2.0, 4.5),
            ('1_7_C_T', 'NOTAGENE', 'p1', 3.0, 2.0, 4.5),
            ('1_8_C_T', 'GENE2', 'p2', 3.0, 0.0, 4.5),  # unusable CI
        ],
        ['variantId', 'geneSymbol', 'phenotype', 'oddsRatio', 'oddsRatioLower', 'oddsRatioUpper'],
    )
    rows = pl.from_pandas(vi.az_rare_variants(exwas, genes, phenotype_map, obsolete).toPandas()).sort('as_disease')
    assert rows.select('variant', 'as_gene', 'as_disease', 'GsourceLab').rows() == [
        ('1_5_C_T', 'g1', 'EFO_1', SOURCE_AZ_RARE),
        ('1_5_C_T', 'g1', 'EFO_3', SOURCE_AZ_RARE),
    ]
    assert rows['yc'][0] == pytest.approx(math.log(3.0))
    assert rows['ycse'][0] == pytest.approx((math.log(4.5) - math.log(2.0)) / 3.92)


def test_burden_tests_pick_lowest_p_then_largest_effect_after_remap(spark, genes, phenotype_map, obsolete):
    text = ['Gene', 'Phenotype', 'CollapsingModel', 'pValue', 'BinOddsRatio', 'BinOddsRatioLCI', 'BinOddsRatioUCI']
    binary = spark.createDataFrame(
        [
            ('GENE1', 'p1', 'ptv', '1e-3', '2.0', '1.5', '2.5'),
            ('GENE1', 'p1', 'ptvraredmg', '1e-5', '1.5', '1.2', '1.9'),  # lowest p for p1 -> EFO_OLD (EFO_1) and EFO_3
            ('GENE1', 'p1', 'syn', '1e-9', '9.0', '8.0', '10.0'),  # model not used
            ('GENE1', 'p2', 'ptv', '1e-2', '0.5', '0.3', '0.8'),
        ],
        text,
    )
    quantitative = spark.createDataFrame(
        [
            ('GENE1', 'p2', 'ptv5pcnt', '1e-4', '0.1', '0.05', '0.15'),
            ('GENE2', 'p2', 'ptv', '1e-4', '-0.3', '-0.5', '-0.1'),
        ],
        ['Gene', 'Phenotype', 'CollapsingModel', 'pValue', 'beta', 'LCI', 'UCI'],
    )
    rows = pl.from_pandas(
        vi.burden_tests(binary, quantitative, genes, phenotype_map, obsolete, SETTINGS).toPandas()
    ).sort('as_gene', 'as_disease')
    assert rows.select('as_gene', 'as_disease').rows() == [
        ('g1', 'EFO_1'),
        ('g1', 'EFO_2'),
        ('g1', 'EFO_3'),
        ('g2', 'EFO_2'),
    ]
    assert rows.filter(pl.col('as_disease') == 'EFO_1')['bO'][0] == pytest.approx(math.log(1.5))
    assert rows.filter((pl.col('as_gene') == 'g1') & (pl.col('as_disease') == 'EFO_2'))['bO'][0] == pytest.approx(0.1)
    assert rows.filter(pl.col('as_gene') == 'g2')['bOse'][0] == pytest.approx(0.4 / 3.92)


def test_annotate_and_assemble_fill_defaults_and_classes(spark, variant, genes):
    consequences = vi.gene_consequences(variant, genes)
    provenance = [None] * len(PROVENANCE_COLUMNS)
    rows = spark.createDataFrame(
        [
            ('1_10_A_G', 'g1', 'EFO_1', SOURCE_CODING_GWAS, 2, 0.3, 0.04, 0.0, 0.1, 0.0, *provenance),
            ('1_10_A_G', 'g2', 'EFO_1', SOURCE_CODING_GWAS, 2, 0.3, 0.04, 0.0, 0.1, 0.0, *provenance),
            ('1_20_C_T', 'g1', 'EFO_1', SOURCE_AZ_RARE, 2, 1.1, 0.2, 0.0, 0.1, 0.0, *provenance),
            ('1_99_T_A', 'g1', 'EFO_1', SOURCE_CLINVAR, 2, 0.0, 0.14, 0.0, 0.1, 1.0, *provenance),
        ],
        t.StructType([
            t.StructField('variant', t.StringType()),
            t.StructField('as_gene', t.StringType()),
            t.StructField('as_disease', t.StringType()),
            t.StructField('GsourceLab', t.IntegerType()),
            t.StructField('GqtlLab', t.IntegerType()),
            *[t.StructField(c, t.DoubleType()) for c in ('yc', 'ycse', 'xc', 'xcse', 'as_clinicalSignificance')],
            *[t.StructField(c, t.StringType()) for c in PROVENANCE_COLUMNS],
        ]),
    )
    burden = spark.createDataFrame([('g1', 'EFO_1', 0.7, 0.2)], ['as_gene', 'as_disease', 'bO', 'bOse'])
    out = vi.annotate(vi.assemble([rows], burden), variant, consequences, SETTINGS).select(*vi.OUTPUT_COLUMNS)
    frame = pl.from_pandas(out.toPandas()).sort('variant', 'as_gene')
    assert frame.columns == list(vi.OUTPUT_COLUMNS)
    assert set(ANNOTATION_COLUMNS) <= set(frame.columns)
    assert frame.select('variant', 'as_gene', 'bO', 'consequence_class', 'is_lof').rows() == [
        ('1_10_A_G', 'g1', 0.7, 'missense', False),
        ('1_10_A_G', 'g2', DEFAULT_BO, 'other', False),
        ('1_20_C_T', 'g1', 0.7, 'lof', True),
        ('1_99_T_A', 'g1', 0.7, 'unknown', False),
    ]
    assert frame.filter(pl.col('as_gene') == 'g2')['bOse'][0] == DEFAULT_BOSE
    gpn = frame.filter(pl.col('variant') == '1_10_A_G')['as_gpnstar'][0]
    assert gpn == pytest.approx(1 - 1 / (1 + math.exp(-(-4.9 + -0.65 * -6.0))))
    alphamissense = frame.filter(pl.col('variant') == '1_10_A_G').sort('as_gene')['as_alphamissense'].to_list()
    assert alphamissense == pytest.approx([
        0.1,
        0.8,
    ])
    assert frame.filter(pl.col('variant') == '1_99_T_A')['as_gpnstar'][0] is None


def test_map_phenotypes_uses_the_ontoma_mapping(spark, monkeypatch):
    def fake_mapping(evidence, **_):
        return evidence.with_columns(
            pl
            .when(pl.col('diseaseFromSource') == 'p1')
            .then(pl.lit('EFO_1'))
            .otherwise(pl.lit(None))
            .alias(vi.MAPPED_ID_COLUMN)
        )

    monkeypatch.setattr(vi, 'add_efo_mapping', fake_mapping)
    phenotypes = spark.createDataFrame([('p1',), ('p1',), ('p2',), (None,)], ['phenotype'])
    assert vi.map_phenotypes(spark, phenotypes, 'unused').collect() == [('p1', 'EFO_1')]


def test_settings_reject_unknown_keys():
    with pytest.raises(ValueError, match='extra'):
        vi.VidraInputSettings.model_validate({'gpn_star': {'a': 0, 'b': -1}, 'blood_filter': True})
