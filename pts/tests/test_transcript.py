"""Tests for the transcript PySpark module."""

from pyspark.sql import Row
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from pts.pyspark.transcript import (
    _build_flags,
    _build_uniprot_lut,
    _join_and_finalise,
    _parse_exons,
    _parse_gff3,
)

# ---------------------------------------------------------------------------
# Shared schemas and helpers
# ---------------------------------------------------------------------------

GFF3_SCHEMA = StructType([
    StructField('_c0', StringType()),
    StructField('_c1', StringType()),
    StructField('_c2', StringType()),
    StructField('_c3', StringType()),
    StructField('_c4', StringType()),
    StructField('_c5', StringType()),
    StructField('_c6', StringType()),
    StructField('_c7', StringType()),
    StructField('_c8', StringType()),
])

ENSEMBL_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField(
        'transcripts',
        ArrayType(StructType([
            StructField('id', StringType()),
            StructField('uniprot_swissprot', ArrayType(StringType())),
            StructField('uniprot_trembl', ArrayType(StringType())),
        ])),
    ),
])


def _gff(chrom, feature, start, end, strand, attrs):
    return Row(
        _c0=chrom, _c1='HAVANA', _c2=feature,
        _c3=str(start), _c4=str(end), _c5='.', _c6=strand, _c7='.', _c8=attrs,
    )


_TX_ATTRS = (
    'gene_id=ENSG00000186092.7;transcript_id=ENST00000641515.2;'
    'transcript_type=protein_coding;protein_id=ENSP00000493376.2;'
    'tag=Ensembl_canonical,GENCODE_Primary,MANE_Select,appris_principal_1'
)
_NC_ATTRS = (
    'gene_id=ENSG00000290825.3;transcript_id=ENST00000832824.1;'
    'transcript_type=lncRNA;tag=basic'
)
_EXON_ATTRS_A = 'transcript_id=ENST00000641515.2;exon_id=ENSE00003899065.1'
_EXON_ATTRS_B = 'transcript_id=ENST00000641515.2;exon_id=ENSE00001234567.2'


# ---------------------------------------------------------------------------
# _parse_gff3
# ---------------------------------------------------------------------------


def test_parse_gff3_excludes_non_transcript_features(spark):
    """Gene and exon rows are dropped; only transcript rows pass."""
    rows = [
        _gff('chr1', 'gene', 65419, 71585, '+', _TX_ATTRS),
        _gff('chr1', 'transcript', 65419, 71585, '+', _TX_ATTRS),
        _gff('chr1', 'exon', 65419, 65433, '+', _TX_ATTRS),
    ]
    result = _parse_gff3(spark.createDataFrame(rows, GFF3_SCHEMA))
    assert result.count() == 1
    row = result.first()
    assert row is not None
    assert row.transcriptId == 'ENST00000641515'


def test_parse_gff3_strips_version_suffix(spark):
    """Version suffixes (.N) are removed from gene, transcript and protein IDs."""
    rows = [_gff('chr1', 'transcript', 65419, 71585, '+', _TX_ATTRS)]
    row = _parse_gff3(spark.createDataFrame(rows, GFF3_SCHEMA)).first()
    assert row is not None
    assert row.targetId == 'ENSG00000186092'
    assert row.transcriptId == 'ENST00000641515'
    assert row.proteinId == 'ENSP00000493376'


def test_parse_gff3_normalises_chrm_to_mt(spark):
    """ChrM is normalised to MT."""
    rows = [_gff('chrM', 'transcript', 1000, 2000, '+', _NC_ATTRS.replace('ENSG00000290825', 'ENSG00000000001'))]
    row = _parse_gff3(spark.createDataFrame(rows, GFF3_SCHEMA)).first()
    assert row is not None
    assert row.chromosome == 'MT'


def test_parse_gff3_filters_noncanonical_chromosomes(spark):
    """Scaffold/patch chromosomes are excluded."""
    rows = [
        _gff('chr1', 'transcript', 65419, 71585, '+', _TX_ATTRS),
        _gff('chr1_KI270706v1_random', 'transcript', 65419, 71585, '+', _TX_ATTRS),
    ]
    result = _parse_gff3(spark.createDataFrame(rows, GFF3_SCHEMA))
    assert result.count() == 1
    row = result.first()
    assert row is not None
    assert row.chromosome == '1'


def test_parse_gff3_tss_positive_strand(spark):
    """TranscriptionStartSite equals start for + strand transcripts."""
    rows = [_gff('chr1', 'transcript', 65419, 71585, '+', _TX_ATTRS)]
    row = _parse_gff3(spark.createDataFrame(rows, GFF3_SCHEMA)).first()
    assert row is not None
    assert row.transcriptionStartSite == 65419


def test_parse_gff3_tss_negative_strand(spark):
    """TranscriptionStartSite equals end for - strand transcripts."""
    rows = [_gff('chr1', 'transcript', 450740, 451678, '-', _TX_ATTRS)]
    row = _parse_gff3(spark.createDataFrame(rows, GFF3_SCHEMA)).first()
    assert row is not None
    assert row.transcriptionStartSite == 451678


def test_parse_gff3_null_protein_id_for_noncoding(spark):
    """ProteinId is null when protein_id is absent from the attributes."""
    rows = [_gff('chr1', 'transcript', 11121, 14413, '+', _NC_ATTRS)]
    row = _parse_gff3(spark.createDataFrame(rows, GFF3_SCHEMA)).first()
    assert row is not None
    assert row.proteinId is None


def test_parse_gff3_is_ensembl_canonical_true_when_tagged(spark):
    """IsEnsemblCanonical is True when Ensembl_canonical is in the tag list."""
    rows = [_gff('chr1', 'transcript', 65419, 71585, '+', _TX_ATTRS)]
    row = _parse_gff3(spark.createDataFrame(rows, GFF3_SCHEMA)).first()
    assert row is not None
    assert row.isEnsemblCanonical is True


def test_parse_gff3_is_ensembl_canonical_false_when_not_tagged(spark):
    """IsEnsemblCanonical is False when Ensembl_canonical is absent."""
    rows = [_gff('chr1', 'transcript', 11121, 14413, '+', _NC_ATTRS)]
    row = _parse_gff3(spark.createDataFrame(rows, GFF3_SCHEMA)).first()
    assert row is not None
    assert row.isEnsemblCanonical is False


# ---------------------------------------------------------------------------
# _build_flags
# ---------------------------------------------------------------------------


def _flags_for(spark, tags: list[str]) -> list:
    tags_col_df = spark.createDataFrame(
        [Row(tags=tags)],
        StructType([StructField('tags', ArrayType(StringType()))]),
    )
    import pyspark.sql.functions as f
    row = tags_col_df.select(_build_flags(f.col('tags')).alias('flags')).first()
    assert row is not None
    return row.flags


def test_build_flags_mane_select(spark):
    flags = _flags_for(spark, ['MANE_Select'])
    assert any(r.label == 'mane' and r.value == 'select' for r in flags)


def test_build_flags_mane_plus_clinical(spark):
    flags = _flags_for(spark, ['MANE_Plus_Clinical'])
    assert any(r.label == 'mane' and r.value == 'plus_clinical' for r in flags)


def test_build_flags_gencode_primary(spark):
    flags = _flags_for(spark, ['GENCODE_Primary'])
    assert any(r.label == 'gencode_primary' and r.value == 'true' for r in flags)


def test_build_flags_ensembl_canonical(spark):
    flags = _flags_for(spark, ['Ensembl_canonical'])
    assert any(r.label == 'ensembl_canonical' and r.value == 'true' for r in flags)


def test_build_flags_appris_principal(spark):
    flags = _flags_for(spark, ['appris_principal_1'])
    assert any(r.label == 'appris' and r.value == 'principal_1' for r in flags)


def test_build_flags_appris_alternative(spark):
    flags = _flags_for(spark, ['appris_alternative_2'])
    assert any(r.label == 'appris' and r.value == 'alternative_2' for r in flags)


def test_build_flags_empty_for_irrelevant_tags(spark):
    flags = _flags_for(spark, ['basic', 'CCDS', 'RNA_Seq_supported_partial'])
    assert flags == []


def test_build_flags_multiple_flags(spark):
    flags = _flags_for(spark, ['MANE_Select', 'GENCODE_Primary', 'Ensembl_canonical', 'appris_principal_1'])
    labels = {r.label for r in flags}
    assert labels == {'mane', 'gencode_primary', 'ensembl_canonical', 'appris'}


# ---------------------------------------------------------------------------
# _parse_exons
# ---------------------------------------------------------------------------


def test_parse_exons_groups_by_transcript(spark):
    """Multiple exon rows for the same transcript are collected into one array."""
    rows = [
        _gff('chr1', 'exon', 65419, 65433, '+', _EXON_ATTRS_A),
        _gff('chr1', 'exon', 65500, 65600, '+', _EXON_ATTRS_B),
        _gff('chr1', 'transcript', 65419, 71585, '+', _TX_ATTRS),  # should be ignored
    ]
    result = _parse_exons(spark.createDataFrame(rows, GFF3_SCHEMA))
    assert result.count() == 1
    row = result.first()
    assert row is not None
    assert row.transcriptId == 'ENST00000641515'
    assert len(row.exons) == 2


def test_parse_exons_strips_version_suffix(spark):
    """Version suffixes are stripped from exon_id and transcript_id."""
    rows = [_gff('chr1', 'exon', 65419, 65433, '+', _EXON_ATTRS_A)]
    row = _parse_exons(spark.createDataFrame(rows, GFF3_SCHEMA)).first()
    assert row is not None
    assert row.transcriptId == 'ENST00000641515'
    assert row.exons[0].exonId == 'ENSE00003899065'


def test_parse_exons_exon_struct_fields(spark):
    """Each exon struct has exonId, chromosome, start, end, strand."""
    rows = [_gff('chr1', 'exon', 65419, 65433, '+', _EXON_ATTRS_A)]
    row = _parse_exons(spark.createDataFrame(rows, GFF3_SCHEMA)).first()
    assert row is not None
    exon = row.exons[0]
    assert exon.chromosome == '1'
    assert exon.start == 65419
    assert exon.end == 65433
    assert exon.strand == '+'


# ---------------------------------------------------------------------------
# _build_uniprot_lut
# ---------------------------------------------------------------------------


def test_build_uniprot_lut_merges_swissprot_and_trembl(spark):
    """UniprotIds contains both Swiss-Prot and TrEMBL IDs."""
    data = [Row(id='ENSG00000001', transcripts=[
        Row(id='ENST00000001', uniprot_swissprot=['P12345'], uniprot_trembl=['A0A001']),
    ])]
    result = _build_uniprot_lut(spark.createDataFrame(data, ENSEMBL_SCHEMA))
    row = result.filter('transcriptId = "ENST00000001"').first()
    assert row is not None
    assert set(row.uniprotIds) == {'P12345', 'A0A001'}


def test_build_uniprot_lut_handles_null_trembl(spark):
    """Null uniprot_trembl does not cause an error."""
    data = [Row(id='ENSG00000002', transcripts=[
        Row(id='ENST00000002', uniprot_swissprot=['P99999'], uniprot_trembl=None),
    ])]
    result = _build_uniprot_lut(spark.createDataFrame(data, ENSEMBL_SCHEMA))
    row = result.filter('transcriptId = "ENST00000002"').first()
    assert row is not None
    assert row.uniprotIds == ['P99999']


def test_build_uniprot_lut_deduplicates(spark):
    """Duplicate IDs appearing in both swissprot and trembl are deduplicated."""
    data = [Row(id='ENSG00000003', transcripts=[
        Row(id='ENST00000003', uniprot_swissprot=['P11111'], uniprot_trembl=['P11111']),
    ])]
    result = _build_uniprot_lut(spark.createDataFrame(data, ENSEMBL_SCHEMA))
    row = result.filter('transcriptId = "ENST00000003"').first()
    assert row is not None
    assert row.uniprotIds.count('P11111') == 1


# ---------------------------------------------------------------------------
# _join_and_finalise
# ---------------------------------------------------------------------------


def test_join_and_finalise_output_schema(spark):
    """Output contains exactly the expected columns."""
    expected_cols = {
        'targetId', 'transcriptId', 'biotype', 'proteinId', 'uniprotIds',
        'isEnsemblCanonical', 'chromosome', 'start', 'end', 'strand',
        'transcriptionStartSite', 'flags', 'exons',
    }
    gff_rows = [_gff('chr1', 'transcript', 65419, 71585, '+', _TX_ATTRS)]
    gff = _parse_gff3(spark.createDataFrame(gff_rows, GFF3_SCHEMA))

    exon_rows = [_gff('chr1', 'exon', 65419, 65433, '+', _EXON_ATTRS_A)]
    exons = _parse_exons(spark.createDataFrame(exon_rows, GFF3_SCHEMA))

    ensembl_data = [Row(id='ENSG00000186092', transcripts=[
        Row(id='ENST00000641515', uniprot_swissprot=['A0A2U3U0J3'], uniprot_trembl=None),
    ])]
    uniprot = _build_uniprot_lut(spark.createDataFrame(ensembl_data, ENSEMBL_SCHEMA))

    result = _join_and_finalise(gff, exons, uniprot)
    assert set(result.columns) == expected_cols


def test_join_and_finalise_propagates_uniprot_ids(spark):
    """UniProt IDs from the Ensembl parquet are attached to the correct transcript."""
    gff_rows = [_gff('chr1', 'transcript', 65419, 71585, '+', _TX_ATTRS)]
    gff = _parse_gff3(spark.createDataFrame(gff_rows, GFF3_SCHEMA))

    exon_rows = [_gff('chr1', 'exon', 65419, 65433, '+', _EXON_ATTRS_A)]
    exons = _parse_exons(spark.createDataFrame(exon_rows, GFF3_SCHEMA))

    ensembl_data = [Row(id='ENSG00000186092', transcripts=[
        Row(id='ENST00000641515', uniprot_swissprot=['A0A2U3U0J3'], uniprot_trembl=None),
    ])]
    uniprot = _build_uniprot_lut(spark.createDataFrame(ensembl_data, ENSEMBL_SCHEMA))

    row = _join_and_finalise(gff, exons, uniprot).first()
    assert row is not None
    assert row.uniprotIds == ['A0A2U3U0J3']
    assert len(row.exons) == 1
