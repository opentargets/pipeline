"""Tests for the homology PySpark module."""

from pyspark.sql import Row
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.homology import _build_homology, _resolve_homology

# ---------------------------------------------------------------------------
# Shared schemas
# ---------------------------------------------------------------------------

TARGET_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('approvedSymbol', StringType()),
])

ORTHOLOG_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('speciesId', StringType()),
    StructField('speciesName', StringType()),
    StructField('homologyType', StringType()),
    StructField('targetGeneId', StringType()),
    StructField('isHighConfidence', StringType()),
    StructField('targetGeneSymbol', StringType()),
    StructField('queryPercentageIdentity', DoubleType()),
    StructField('targetPercentageIdentity', DoubleType()),
    StructField('priority', IntegerType()),
])


HOMOLOGY_DICT_SCHEMA = StructType([
    StructField('#name', StringType()),
    StructField('species', StringType()),
    StructField('taxonomy_id', StringType()),
])

CODING_PROTEINS_SCHEMA = StructType([
    StructField('gene_stable_id', StringType()),
    StructField('protein_stable_id', StringType()),
    StructField('species', StringType()),
    StructField('identity', DoubleType()),
    StructField('homology_type', StringType()),
    StructField('homology_gene_stable_id', StringType()),
    StructField('homology_protein_stable_id', StringType()),
    StructField('homology_species', StringType()),
    StructField('homology_identity', DoubleType()),
    StructField('dn', DoubleType()),
    StructField('ds', DoubleType()),
    StructField('goc_score', DoubleType()),
    StructField('wga_coverage', DoubleType()),
    StructField('is_high_confidence', StringType()),
    StructField('homology_id', StringType()),
])

GENE_DICT_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('name', StringType()),
])


def _mouse_pair_row(**kwargs):
    """A single homo_sapiens/mouse coding-protein homology row, whitelisted by default."""
    defaults = {
        'gene_stable_id': 'ENSG0001',
        'protein_stable_id': 'P001',
        'species': 'homo_sapiens',
        'identity': 100.0,
        'homology_type': 'ortholog_one2one',
        'homology_gene_stable_id': 'ENSMUSG0001',
        'homology_protein_stable_id': 'P002',
        'homology_species': 'mus_musculus',
        'homology_identity': 88.0,
        'dn': None,
        'ds': None,
        'goc_score': None,
        'wga_coverage': None,
        'is_high_confidence': '1',
        'homology_id': 'h001',
    }
    defaults.update(kwargs)
    return Row(**defaults)


def _ortholog_row(**kwargs):
    defaults = {
        'id': 'ENSG1',
        'speciesId': '10090',
        'speciesName': 'mouse',
        'homologyType': 'ortholog_one2one',
        'targetGeneId': 'ENSMUSG1',
        'isHighConfidence': '1',
        'targetGeneSymbol': 'Gene1',
        'queryPercentageIdentity': 90.0,
        'targetPercentageIdentity': 88.0,
        'priority': 0,
    }
    defaults.update(kwargs)
    return Row(**defaults)


# ---------------------------------------------------------------------------
# _build_homology
# ---------------------------------------------------------------------------


def test_homology_whitelist_filtering(spark):
    """Only species in whitelist are included in homologues."""
    homology_dict_schema = StructType([
        StructField('#name', StringType()),
        StructField('species', StringType()),
        StructField('taxonomy_id', StringType()),
    ])
    homology_dict_data = [
        Row(**{
            '#name': 'mus_musculus',
            'species': 'mus_musculus',
            'taxonomy_id': '10090',
        }),
        Row(**{
            '#name': 'rattus_norvegicus',
            'species': 'rattus_norvegicus',
            'taxonomy_id': '10116',
        }),
        Row(**{
            '#name': 'danio_rerio',
            'species': 'danio_rerio',
            'taxonomy_id': '7955',
        }),
    ]
    homology_dict_df = spark.createDataFrame(homology_dict_data, homology_dict_schema)

    coding_proteins_schema = StructType([
        StructField('gene_stable_id', StringType()),
        StructField('protein_stable_id', StringType()),
        StructField('species', StringType()),
        StructField('identity', DoubleType()),
        StructField('homology_type', StringType()),
        StructField('homology_gene_stable_id', StringType()),
        StructField('homology_protein_stable_id', StringType()),
        StructField('homology_species', StringType()),
        StructField('homology_identity', DoubleType()),
        StructField('dn', DoubleType()),
        StructField('ds', DoubleType()),
        StructField('goc_score', DoubleType()),
        StructField('wga_coverage', DoubleType()),
        StructField('is_high_confidence', StringType()),
        StructField('homology_id', StringType()),
    ])
    coding_proteins_data = [
        Row(
            gene_stable_id='ENSG0001',
            protein_stable_id='P001',
            species='homo_sapiens',
            identity=100.0,
            homology_type='ortholog_one2one',
            homology_gene_stable_id='ENSMUSG0001',
            homology_protein_stable_id='P002',
            homology_species='mus_musculus',
            homology_identity=88.0,
            dn=None,
            ds=None,
            goc_score=None,
            wga_coverage=None,
            is_high_confidence='1',
            homology_id='h001',
        ),
        # rat — NOT in whitelist
        Row(
            gene_stable_id='ENSG0001',
            protein_stable_id='P001',
            species='homo_sapiens',
            identity=100.0,
            homology_type='ortholog_one2one',
            homology_gene_stable_id='ENSRNOG0001',
            homology_protein_stable_id='P003',
            homology_species='rattus_norvegicus',
            homology_identity=85.0,
            dn=None,
            ds=None,
            goc_score=None,
            wga_coverage=None,
            is_high_confidence='1',
            homology_id='h002',
        ),
    ]
    coding_proteins_df = spark.createDataFrame(coding_proteins_data, coding_proteins_schema)

    gene_dict_schema = StructType([
        StructField('id', StringType()),
        StructField('name', StringType()),
    ])
    gene_dict_df = spark.createDataFrame(
        [Row(id='ENSMUSG0001', name='Trp53')],
        gene_dict_schema,
    )

    # Only mouse in whitelist (10090), not rat (10116)
    whitelist = ['10090-mus_musculus']
    result = _build_homology(homology_dict_df, coding_proteins_df, gene_dict_df, whitelist)
    rows = result.collect()
    species_ids = {r.speciesId for r in rows}
    assert '10090' in species_ids
    assert '10116' not in species_ids


def _single_pair_setup(spark, coding_proteins_row, gene_name):
    """Minimal whitelisted mouse pair plus a one-row gene dictionary, for isolated checks."""
    homology_dict_df = spark.createDataFrame(
        [Row(**{'#name': 'mus_musculus', 'species': 'mus_musculus', 'taxonomy_id': '10090'})],
        HOMOLOGY_DICT_SCHEMA,
    )
    coding_proteins_df = spark.createDataFrame([coding_proteins_row], CODING_PROTEINS_SCHEMA)
    gene_dict_df = spark.createDataFrame([Row(id='ENSMUSG0001', name=gene_name)], GENE_DICT_SCHEMA)
    return _build_homology(homology_dict_df, coding_proteins_df, gene_dict_df, ['10090-mus_musculus'])


def test_homology_normalises_stringified_null_confidence(spark):
    """The raw TSV's literal "NULL" text for is_high_confidence becomes a real null.

    Regression test: the upstream Ensembl Compara TSV writes a missing confidence
    flag as the text "NULL", which the CSV reader has no reason to treat as a null
    on its own -- shipping the literal string in ~93% of homology rows.
    """
    result = _single_pair_setup(spark, _mouse_pair_row(is_high_confidence='NULL'), gene_name='Trp53')
    row = result.first()
    assert row is not None
    assert row.isHighConfidence is None


def test_homology_keeps_real_confidence_flag(spark):
    """A genuine "1"/"0" confidence flag passes through unchanged."""
    result = _single_pair_setup(spark, _mouse_pair_row(is_high_confidence='1'), gene_name='Trp53')
    row = result.first()
    assert row is not None
    assert row.isHighConfidence == '1'


def test_homology_keeps_genuine_symbols_that_look_like_null_tokens(spark):
    """"na", "nan" and "Nil" are real FlyBase gene symbols, not placeholder tokens.

    Regression test: an earlier version of this fix treated these three values as
    fake nulls and replaced them with the gene's stable ID. They are not -- "na"
    is *narrow abdomen* (FBgn0002917, the fly NALCN ortholog), "nan" is
    *nanchung* (FBgn0036414, the fly TRPV1/2/4/5/6 ortholog), and "Nil" is
    *Nilkantha* (FBgn0039421), all confirmed against FlyBase. Case-insensitive
    token matching against a small "looks like missing data" vocabulary is not
    safe here because it collides with real gene nomenclature.
    """
    for real_symbol in ('na', 'nan', 'Nil'):
        result = _single_pair_setup(spark, _mouse_pair_row(), gene_name=real_symbol)
        row = result.first()
        assert row is not None
        assert row.targetGeneSymbol == real_symbol, f'genuine symbol {real_symbol!r} was wrongly replaced'


def test_homology_keeps_real_gene_symbol(spark):
    """A genuine gene symbol is not affected by the placeholder-token check."""
    result = _single_pair_setup(spark, _mouse_pair_row(), gene_name='Trp53')
    row = result.first()
    assert row is not None
    assert row.targetGeneSymbol == 'Trp53'


def test_homology_trims_a_real_gene_symbol(spark):
    """A real gene symbol with stray leading/trailing whitespace is trimmed, not just checked.

    Regression test: name_is_missing checked f.trim(name) but the true branch
    returned the untrimmed name, so a padded value would keep its whitespace.
    """
    result = _single_pair_setup(spark, _mouse_pair_row(), gene_name=' Trp53 ')
    row = result.first()
    assert row is not None
    assert row.targetGeneSymbol == 'Trp53'


def test_homology_normalises_whitespace_only_gene_symbol(spark):
    """A whitespace-only name is treated the same as an empty one."""
    result = _single_pair_setup(spark, _mouse_pair_row(), gene_name='   ')
    row = result.first()
    assert row is not None
    assert row.targetGeneSymbol == 'ENSMUSG0001'


# ---------------------------------------------------------------------------
# _resolve_homology
# ---------------------------------------------------------------------------


def test_resolve_homology_output_columns(spark):
    """Output is flat: targetId plus the existing homologue fields, no wrapper array."""
    orthologs = spark.createDataFrame([_ortholog_row()], ORTHOLOG_SCHEMA)
    target_df = spark.createDataFrame([Row(id='ENSG1', approvedSymbol='GENE1')], TARGET_SCHEMA)
    result = _resolve_homology(orthologs, target_df)
    assert set(result.columns) == {
        'targetId',
        'speciesId',
        'speciesName',
        'homologyType',
        'targetGeneId',
        'isHighConfidence',
        'targetGeneSymbol',
        'queryPercentageIdentity',
        'targetPercentageIdentity',
        'priority',
    }


def test_resolve_homology_renames_id_to_target_id(spark):
    """The query gene id becomes targetId."""
    orthologs = spark.createDataFrame([_ortholog_row(id='ENSG1')], ORTHOLOG_SCHEMA)
    target_df = spark.createDataFrame([Row(id='ENSG1', approvedSymbol='GENE1')], TARGET_SCHEMA)
    row = _resolve_homology(orthologs, target_df).first()
    assert row is not None
    assert row.targetId == 'ENSG1'


def test_resolve_homology_drops_unresolvable_rows(spark):
    """A query id absent from output/target is dropped (validation)."""
    orthologs = spark.createDataFrame([_ortholog_row(id='UNKNOWN')], ORTHOLOG_SCHEMA)
    target_df = spark.createDataFrame([Row(id='ENSG1', approvedSymbol='GENE1')], TARGET_SCHEMA)
    result = _resolve_homology(orthologs, target_df)
    assert result.count() == 0


def test_resolve_homology_falls_back_to_ortholog_symbol(spark):
    """Without a within-target paralog match, the ortholog's own symbol is kept."""
    orthologs = spark.createDataFrame(
        [_ortholog_row(id='ENSG1', targetGeneId='ENSMUSG1', targetGeneSymbol='Trp53')], ORTHOLOG_SCHEMA
    )
    target_df = spark.createDataFrame([Row(id='ENSG1', approvedSymbol='GENE1')], TARGET_SCHEMA)
    row = _resolve_homology(orthologs, target_df).first()
    assert row is not None
    assert row.targetGeneSymbol == 'Trp53'


def test_resolve_homology_prefers_paralog_symbol(spark):
    """When targetGeneId matches another target in the universe, its approvedSymbol wins."""
    orthologs = spark.createDataFrame(
        [_ortholog_row(id='ENSG1', targetGeneId='ENSG2', targetGeneSymbol='stale-symbol')], ORTHOLOG_SCHEMA
    )
    target_df = spark.createDataFrame(
        [
            Row(id='ENSG1', approvedSymbol='GENE1'),
            Row(id='ENSG2', approvedSymbol='GENE2'),
        ],
        TARGET_SCHEMA,
    )
    row = _resolve_homology(orthologs, target_df).first()
    assert row is not None
    assert row.targetGeneSymbol == 'GENE2'
