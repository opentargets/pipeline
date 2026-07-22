"""Tests for the target_view PySpark module."""

from pyspark.sql import Row
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.target_view import (
    _build_chemical_probes,
    _build_homologues,
    _build_safety_liabilities,
    _build_tractability,
    _build_transcripts,
)

# ---------------------------------------------------------------------------
# _build_homologues
# ---------------------------------------------------------------------------

HOMOLOGY_SCHEMA = StructType([
    StructField('targetId', StringType()),
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


def test_build_homologues_groups_by_target(spark):
    """Multiple homology rows for one target collapse into one homologues array."""
    data = [
        Row(
            targetId='ENSG1',
            speciesId='10090',
            speciesName='mouse',
            homologyType='ortholog_one2one',
            targetGeneId='ENSMUSG1',
            isHighConfidence='1',
            targetGeneSymbol='Trp53',
            queryPercentageIdentity=90.0,
            targetPercentageIdentity=88.0,
            priority=0,
        ),
        Row(
            targetId='ENSG1',
            speciesId='10116',
            speciesName='rat',
            homologyType='ortholog_one2one',
            targetGeneId='ENSRNOG1',
            isHighConfidence='1',
            targetGeneSymbol='Tp53',
            queryPercentageIdentity=85.0,
            targetPercentageIdentity=80.0,
            priority=1,
        ),
    ]
    df = spark.createDataFrame(data, HOMOLOGY_SCHEMA)
    result = _build_homologues(df)
    row = result.first()
    assert row is not None
    assert row.id == 'ENSG1'
    assert len(row.homologues) == 2
    species = {h.speciesId for h in row.homologues}
    assert species == {'10090', '10116'}


def test_build_homologues_absent_target_gets_no_row(spark):
    """A target with no homology rows simply has no row (null after left join)."""
    df = spark.createDataFrame([], HOMOLOGY_SCHEMA)
    result = _build_homologues(df)
    assert result.count() == 0


# ---------------------------------------------------------------------------
# _build_tractability
# ---------------------------------------------------------------------------

TRACTABILITY_SCHEMA = StructType([
    StructField('targetId', StringType()),
    StructField('modality', StringType()),
    StructField('id', StringType()),
    StructField('value', BooleanType()),
])


def test_build_tractability_groups_by_target(spark):
    data = [
        Row(targetId='ENSG1', modality='SM', id='Bucket_1', value=True),
        Row(targetId='ENSG1', modality='AB', id='Bucket_2', value=False),
    ]
    df = spark.createDataFrame(data, TRACTABILITY_SCHEMA)
    row = _build_tractability(df).first()
    assert row is not None
    assert row.id == 'ENSG1'
    modalities = {a.modality for a in row.tractability}
    assert modalities == {'SM', 'AB'}


# ---------------------------------------------------------------------------
# _build_safety_liabilities
# ---------------------------------------------------------------------------

EFFECT_TYPE = StructType([
    StructField('direction', StringType()),
    StructField('dosing', StringType()),
])
BIOSAMPLE_TYPE = StructType([
    StructField('tissueLabel', StringType()),
    StructField('tissueId', StringType()),
    StructField('cellLabel', StringType()),
    StructField('cellFormat', StringType()),
    StructField('cellId', StringType()),
])
STUDY_TYPE = StructType([
    StructField('description', StringType()),
    StructField('name', StringType()),
    StructField('type', StringType()),
])

SAFETY_EVENT_SCHEMA = StructType([
    StructField('targetId', StringType()),
    StructField('event', StringType()),
    StructField('eventId', StringType()),
    StructField('effects', ArrayType(EFFECT_TYPE)),
    StructField('biosamples', ArrayType(BIOSAMPLE_TYPE)),
    StructField('datasource', StringType()),
    StructField('literature', StringType()),
    StructField('url', StringType()),
    StructField('studies', ArrayType(STUDY_TYPE)),
])


def test_build_safety_liabilities_groups_by_target(spark):
    data = [
        Row(
            targetId='ENSG1',
            event='cardiotoxicity',
            eventId='EVT1',
            effects=[Row(direction='activation', dosing='chronic')],
            biosamples=[],
            datasource='AOPWiki',
            literature=None,
            url=None,
            studies=[],
        ),
    ]
    df = spark.createDataFrame(data, SAFETY_EVENT_SCHEMA)
    row = _build_safety_liabilities(df).first()
    assert row is not None
    assert row.id == 'ENSG1'
    assert len(row.safetyLiabilities) == 1
    assert row.safetyLiabilities[0].event == 'cardiotoxicity'


# ---------------------------------------------------------------------------
# _build_chemical_probes
# ---------------------------------------------------------------------------

URL_TYPE = StructType([
    StructField('niceName', StringType()),
    StructField('url', StringType()),
])

CHEMICAL_PROBES_SCHEMA = StructType([
    StructField('targetFromSourceId', StringType()),
    StructField('id', StringType()),
    StructField('drugFromSourceId', StringType()),
    StructField('drugId', StringType()),
    StructField('mechanismOfAction', ArrayType(StringType())),
    StructField('origin', ArrayType(StringType())),
    StructField('control', StringType()),
    StructField('isHighQuality', BooleanType()),
    StructField('probesDrugsScore', DoubleType()),
    StructField('probeMinerScore', DoubleType()),
    StructField('scoreInCells', DoubleType()),
    StructField('scoreInOrganisms', DoubleType()),
    StructField('urls', ArrayType(URL_TYPE)),
    StructField('targetId', StringType()),
])


def test_build_chemical_probes_groups_by_target(spark):
    data = [
        Row(
            targetFromSourceId='TP53',
            id='probe-1',
            drugFromSourceId=None,
            drugId=None,
            mechanismOfAction=[],
            origin=[],
            control=None,
            isHighQuality=True,
            probesDrugsScore=0.9,
            probeMinerScore=0.8,
            scoreInCells=None,
            scoreInOrganisms=None,
            urls=[],
            targetId='ENSG1',
        ),
    ]
    df = spark.createDataFrame(data, CHEMICAL_PROBES_SCHEMA)
    row = _build_chemical_probes(df).first()
    assert row is not None
    assert row.id == 'ENSG1'
    assert len(row.chemicalProbes) == 1
    assert row.chemicalProbes[0].id == 'probe-1'


# ---------------------------------------------------------------------------
# _build_transcripts
# ---------------------------------------------------------------------------

EXON_TYPE = StructType([
    StructField('exonId', StringType()),
    StructField('chromosome', StringType()),
    StructField('start', LongType()),
    StructField('end', LongType()),
    StructField('strand', StringType()),
])
FLAG_TYPE = StructType([
    StructField('label', StringType()),
    StructField('value', StringType()),
])

TRANSCRIPT_SCHEMA = StructType([
    StructField('targetId', StringType()),
    StructField('transcriptId', StringType()),
    StructField('biotype', StringType()),
    StructField('proteinId', StringType()),
    StructField('uniprotIds', ArrayType(StringType())),
    StructField('isEnsemblCanonical', BooleanType()),
    StructField('chromosome', StringType()),
    StructField('start', LongType()),
    StructField('end', LongType()),
    StructField('strand', StringType()),
    StructField('transcriptionStartSite', IntegerType()),
    StructField('flags', ArrayType(FLAG_TYPE)),
    StructField('exons', ArrayType(EXON_TYPE)),
])


def _transcript_row(**kwargs):
    defaults = {
        'targetId': 'ENSG1',
        'transcriptId': 'ENST1',
        'biotype': 'protein_coding',
        'proteinId': 'ENSP1',
        'uniprotIds': ['P12345'],
        'isEnsemblCanonical': False,
        'chromosome': '17',
        'start': 100,
        'end': 200,
        'strand': '+',
        'transcriptionStartSite': 100,
        'flags': [],
        'exons': [],
    }
    defaults.update(kwargs)
    return Row(**defaults)


def test_build_transcripts_collects_transcript_ids_and_structs(spark):
    """Transcript ids and transcript structs are collected across all transcripts of a target."""
    data = [
        _transcript_row(transcriptId='ENST1'),
        _transcript_row(transcriptId='ENST2', isEnsemblCanonical=True),
    ]
    df = spark.createDataFrame(data, TRANSCRIPT_SCHEMA)
    row = _build_transcripts(df).first()
    assert row is not None
    assert set(row.transcriptIds) == {'ENST1', 'ENST2'}
    assert len(row.transcripts) == 2


def test_build_transcripts_maps_translation_id_from_protein_id(spark):
    """The translationId field is sourced from proteinId (no separate field in the new pipeline)."""
    df = spark.createDataFrame([_transcript_row(proteinId='ENSP99')], TRANSCRIPT_SCHEMA)
    row = _build_transcripts(df).first()
    assert row is not None
    assert row.transcripts[0].translationId == 'ENSP99'


def test_build_transcripts_uniprot_id_takes_first_of_list(spark):
    """The uniprotId field collapses the uniprotIds list to its first element."""
    df = spark.createDataFrame([_transcript_row(uniprotIds=['P1', 'P2'])], TRANSCRIPT_SCHEMA)
    row = _build_transcripts(df).first()
    assert row is not None
    assert row.transcripts[0].uniprotId == 'P1'


def test_build_transcripts_uniprot_id_null_when_list_empty(spark):
    df = spark.createDataFrame([_transcript_row(uniprotIds=[])], TRANSCRIPT_SCHEMA)
    row = _build_transcripts(df).first()
    assert row is not None
    assert row.transcripts[0].uniprotId is None


def test_build_transcripts_legacy_fields_are_stubbed(spark):
    """The isUniprotReviewed field is always false; alphafoldId/uniprotIsoformId are always null."""
    df = spark.createDataFrame([_transcript_row()], TRANSCRIPT_SCHEMA)
    row = _build_transcripts(df).first()
    assert row is not None
    transcript = row.transcripts[0]
    assert transcript.isUniprotReviewed is False
    assert transcript.alphafoldId is None
    assert transcript.uniprotIsoformId is None


def test_build_transcripts_canonical_transcript_struct(spark):
    """The canonicalTranscript struct is derived from the isEnsemblCanonical row."""
    data = [
        _transcript_row(transcriptId='ENST1', isEnsemblCanonical=False),
        _transcript_row(
            transcriptId='ENST2',
            isEnsemblCanonical=True,
            chromosome='7',
            start=500,
            end=900,
            strand='-',
        ),
    ]
    df = spark.createDataFrame(data, TRANSCRIPT_SCHEMA)
    row = _build_transcripts(df).first()
    assert row is not None
    assert row.canonicalTranscript.id == 'ENST2'
    assert row.canonicalTranscript.chromosome == '7'
    assert row.canonicalTranscript.start == 500
    assert row.canonicalTranscript.end == 900
    assert row.canonicalTranscript.strand == '-'


def test_build_transcripts_canonical_exons_flattened_ascending(spark):
    """The canonicalExons field is a flat [start, end, start, end, ...] list, ordered by start."""
    canonical_exons = [
        Row(exonId='EXON2', chromosome='7', start=500, end=600, strand='-'),
        Row(exonId='EXON1', chromosome='7', start=100, end=200, strand='-'),
    ]
    data = [_transcript_row(isEnsemblCanonical=True, exons=canonical_exons)]
    df = spark.createDataFrame(data, TRANSCRIPT_SCHEMA)
    row = _build_transcripts(df).first()
    assert row is not None
    assert row.canonicalExons == ['100', '200', '500', '600']


def test_build_transcripts_no_canonical_row_gives_null_canonical_fields(spark):
    """A target with no isEnsemblCanonical=true row gets null canonicalTranscript/Exons."""
    df = spark.createDataFrame([_transcript_row(isEnsemblCanonical=False)], TRANSCRIPT_SCHEMA)
    row = _build_transcripts(df).first()
    assert row is not None
    assert row.canonicalTranscript is None
    assert row.canonicalExons is None
