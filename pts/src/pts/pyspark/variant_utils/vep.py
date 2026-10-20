"""Parse Ensembl VEP JSON output into the variant index, ported from Gentropy's VEP parser.

Differences from Gentropy: consequence terms use the corrected Sequence Ontology spelling
(see `consequences.py`), SIFT and PolyPhen are not extracted (VEP runs without them), and only
VEP JSON parts are read, never VEP's `<part>.json_warnings.txt` files.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.common.utils import polars_schema_to_spark
from pts.pyspark.variant_utils.consequences import CONSEQUENCE_SCORES, SEQUENCE_ONTOLOGY
from pts.pyspark.variant_utils.index import (
    VARIANT_SCHEMA,
    fill_null_arrays,
    hash_long_variant_ids,
    normalise_variant_effects,
)
from pts.pyspark.variant_utils.spark import (
    map_column_by_dictionary,
    order_structs_by_field,
    order_structs_by_two_fields,
)
from pts.schemas.vep import vep_schema

VEP_SCHEMA: t.StructType = polars_schema_to_spark(vep_schema)
#: VEP data parts, optionally gzipped; excludes VEP's `<part>.json_warnings.txt`.
VEP_PARTS = '*.{json,jsonl,json.gz,jsonl.gz}'
MISSENSE = 'SO_0001583'


def read_vep(spark: SparkSession, path: str) -> DataFrame:
    """Read VEP JSON parts below `path` with the fixed VEP schema."""
    frame = (
        spark.read
        .schema(VEP_SCHEMA)
        .option('recursiveFileLookup', 'true')
        .option('pathGlobFilter', VEP_PARTS)
        .json(path)
    )
    if frame.isEmpty():
        raise ValueError(f'No VEP records found in {path}')
    return frame


def generate_dbxrefs(ids: Column, source: str) -> Column:
    """Turn a list of identifiers into cross-references, dropping null identifiers."""
    xrefs = f.transform(
        f.filter(ids, lambda identifier: identifier.isNotNull()),
        lambda identifier: f.struct(identifier.cast(t.StringType()).alias('id'), f.lit(source).alias('source')),
    )
    return f.when(xrefs.isNull(), f.array().cast(VARIANT_SCHEMA['dbXrefs'].dataType)).otherwise(xrefs)


def colocated_rsids(colocated: Column) -> Column:
    """The identifiers of colocated variants, null when there are none."""
    return f.when(colocated.isNotNull(), f.transform(colocated, lambda variant: variant.getItem('id')))


def omim_xrefs(colocated: Column) -> Column:
    """OMIM cross-references, written `<entry>#<variant>` as on omim.org.

    VEP writes the IDs as JSON numbers; they are read as text so suffixes keep their trailing
    zeros (`600234.0010` -> `600234#0010`). Gentropy read them as doubles and wrote `600234#001`.
    """
    with_omim = f.filter(colocated, lambda variant: variant.getItem('var_synonyms').getItem('OMIM').isNotNull())
    ids = f.transform(
        with_omim,
        lambda variant: f.transform(
            variant.getItem('var_synonyms').getItem('OMIM'),
            lambda omim: f.regexp_replace(omim, r'\.', '#'),
        ),
    )
    return generate_dbxrefs(f.flatten(ids), 'omim')


def clinvar_xrefs(colocated: Column) -> Column:
    """ClinVar variation (VCV) cross-references."""
    with_clinvar = f.filter(colocated, lambda variant: variant.getItem('var_synonyms').getItem('ClinVar').isNotNull())
    ids = f.transform(
        with_clinvar,
        lambda variant: f.filter(variant.getItem('var_synonyms').getItem('ClinVar'), lambda x: x.startswith('VCV')),
    )
    return generate_dbxrefs(f.flatten(ids), 'clinvar')


def variant_effect(
    method: str,
    score: str,
    assessment: str | None = None,
    assessment_flag: str | None = None,
) -> Column:
    """The predictor's effect from the transcript with its highest score, if any."""
    transcript = f.filter(order_structs_by_field('transcript_consequences', score), lambda x: x[score].isNotNull())[0]
    return f.when(
        transcript.isNotNull(),
        f.struct(
            f.lit(method).alias('method'),
            (transcript[assessment] if assessment else f.lit(None)).cast(t.StringType()).alias('assessment'),
            transcript[score].cast(t.FloatType()).alias('score'),
            (transcript[assessment_flag] if assessment_flag else f.lit(None))
            .cast(t.StringType())
            .alias('assessmentFlag'),
            transcript['gene_id'].alias('targetId'),
            f.lit(None).cast(t.DoubleType()).alias('normalisedScore'),
        ),
    )


def alpha_missense_effect(transcripts: Column) -> Column:
    """AlphaMissense from the first transcript that has a prediction."""
    transcript = f.filter(transcripts, lambda x: x.getItem('alphamissense').isNotNull())[0]
    return f.when(
        transcript.isNotNull(),
        f.struct(
            f.lit('AlphaMissense').alias('method'),
            transcript.alphamissense.am_class.alias('assessment'),
            transcript.alphamissense.am_pathogenicity.cast(t.FloatType()).alias('score'),
            f.lit(None).cast(t.StringType()).alias('assessmentFlag'),
            transcript.gene_id.alias('targetId'),
            f.lit(None).cast(t.DoubleType()).alias('normalisedScore'),
        ),
    )


def vep_prediction(consequence: Column) -> Column:
    """VEP's own prediction: the most severe consequence and its score."""
    return f.struct(
        f.lit('VEP').alias('method'),
        consequence.alias('assessment'),
        map_column_by_dictionary(consequence, CONSEQUENCE_SCORES).cast(t.FloatType()).alias('score'),
        f.lit(None).cast(t.StringType()).alias('assessmentFlag'),
        f.lit(None).cast(t.StringType()).alias('targetId'),
        f.lit(None).cast(t.DoubleType()).alias('normalisedScore'),
    )


def amino_acid_change(amino_acids: Column, protein_end: Column) -> Column:
    """One-letter substitution such as `A12B` from VEP's `A/B` and the protein position."""
    parts = f.split(amino_acids, '/')
    return f.when(amino_acids.isNotNull() & protein_end.isNotNull(), f.concat(parts[0], protein_end, parts[1]))


def uniprot_accessions(swissprot: Column, trembl: Column) -> Column:
    """Distinct Swiss-Prot and TrEMBL accessions without version suffixes."""
    accessions = f.transform(
        f.flatten(f.filter(f.array(swissprot, trembl), lambda x: x.isNotNull())), lambda x: f.split(x, r'\.')[0]
    )
    return f.filter(f.array_distinct(accessions), lambda x: x.isNotNull())


def transcript_consequence(transcript: Column) -> Column:
    """One transcript consequence, in the variant schema's field order."""
    return f.struct(
        f.transform(transcript.consequence_terms, lambda term: map_column_by_dictionary(term, SEQUENCE_ONTOLOGY)).alias(
            'variantFunctionalConsequenceIds'
        ),
        amino_acid_change(transcript.amino_acids, transcript.protein_end).alias('aminoAcidChange'),
        uniprot_accessions(transcript.swissprot, transcript.trembl).alias('uniprotAccessions'),
        f.when(transcript.canonical == 1, f.lit(True)).otherwise(f.lit(False)).alias('isEnsemblCanonical'),
        transcript.codons.alias('codons'),
        f.coalesce(transcript.distance, f.lit(0)).cast(t.LongType()).alias('distanceFromFootprint'),
        transcript.tssdistance.cast(t.LongType()).alias('distanceFromTss'),
        transcript.appris.alias('appris'),
        transcript.mane_select.alias('maneSelect'),
        transcript.gene_id.alias('targetId'),
        transcript.impact.alias('impact'),
        transcript.lof.cast(t.StringType()).alias('lofteePrediction'),
        f
        .array_max(
            f.transform(transcript.consequence_terms, lambda term: map_column_by_dictionary(term, CONSEQUENCE_SCORES))
        )
        .cast(t.FloatType())
        .alias('consequenceScore'),
        f.lit(None).cast(t.IntegerType()).alias('transcriptIndex'),
        transcript.gene_symbol.alias('approvedSymbol'),
        transcript.biotype.alias('biotype'),
        transcript.transcript_id.alias('transcriptId'),
    )


def _consequence_text(consequence: Column) -> Column:
    suffix = f.when(consequence.contains('variant'), f.lit('')).otherwise(f.lit(' variant'))
    return f.concat(f.regexp_replace(f.initcap(consequence), '_', ' '), suffix)


def _gene_label(transcript: Column) -> Column:
    return f.coalesce(transcript.getField('approvedSymbol'), transcript.getField('targetId'))


def _overlap_text(transcript: Column) -> Column:
    distance = transcript.getField('distanceFromFootprint')
    return f.when(distance == 0, f.concat(f.lit(' overlapping with '), _gene_label(transcript))).otherwise(
        f.concat(f.lit(' '), f.format_number(distance, 0), f.lit(' basepair away from '), _gene_label(transcript))
    )


def variant_description(consequence: Column, first: Column, first_protein_coding: Column) -> Column:
    """A sentence describing the variant from its most severe consequence and closest genes."""
    amino_acid = f.when(
        first.getField('aminoAcidChange').isNotNull(),
        f.concat(
            f.lit(', causing amino-acid change: '),
            first.getField('aminoAcidChange'),
            f.lit(' with '),
            f.lower(first.getField('impact')),
            f.lit(' impact.'),
        ),
    ).otherwise(f.lit('.'))
    loftee = f.when(
        first.getField('lofteePrediction') == 'HC', f.lit(' A high-confidence loss-of-function variant by loftee.')
    ).otherwise(f.lit(''))
    biotype = first.getField('biotype')
    biotype_text = f.concat(
        f.lit(', a '),
        f.when(biotype.contains('gene'), f.regexp_replace(biotype, '_', ' ')).otherwise(
            f.concat(f.regexp_replace(biotype, '_', ' '), f.lit(' gene.'))
        ),
    )
    closest_protein_coding = f.when(
        first_protein_coding.isNotNull(),
        f.concat(
            f.lit(' The closest protein-coding gene is '),
            _gene_label(first_protein_coding),
            f.lit(' ('),
            f.format_number(first_protein_coding.getField('distanceFromFootprint'), 0),
            f.lit(' basepair away).'),
        ),
    ).otherwise(f.lit(''))
    return (
        f
        .when(first.isNull(), f.lit('Intergenic variant no gene in window'))
        .when(
            biotype == 'protein_coding',
            f.concat(_consequence_text(consequence), _overlap_text(first), amino_acid, loftee),
        )
        .otherwise(f.concat(_consequence_text(consequence), _overlap_text(first), biotype_text, closest_protein_coding))
    )


def parse_vep(frame: DataFrame, hash_threshold: int = 300) -> DataFrame:
    """Turn raw VEP records into variant index rows, keeping canonical transcripts only."""
    fields = f.split(f.col('input'), '\t')
    colocated = f.col('colocated_variants')
    transcripts = f.col('transcript_consequences')
    effects = f.filter(
        f.array(
            variant_effect('CADD', 'cadd_phred'),
            variant_effect('LOFTEE', 'lof', 'lof', 'lof_filter'),
            variant_effect('GERP', 'conservation'),
            alpha_missense_effect(transcripts),
            vep_prediction(f.col('most_severe_consequence')),
        ),
        lambda effect: effect.isNotNull(),
    )
    parsed = (
        frame
        .withColumn('transcript_consequences', f.filter(transcripts, lambda x: x.getItem('canonical') == 1))
        .select(
            f.concat_ws('_', fields[0], fields[1], fields[3], fields[4]).alias('variantId'),
            fields[0].cast(t.StringType()).alias('chromosome'),
            fields[1].cast(t.IntegerType()).alias('position'),
            fields[3].cast(t.StringType()).alias('referenceAllele'),
            fields[4].cast(t.StringType()).alias('alternateAllele'),
            generate_dbxrefs(colocated_rsids(colocated), 'ensembl_variation').alias('_ensembl_xrefs'),
            omim_xrefs(colocated).alias('_omim_xrefs'),
            clinvar_xrefs(colocated).alias('_clinvar_xrefs'),
            f
            .when(transcripts.isNotNull(), effects)
            .otherwise(f.array(vep_prediction(f.col('most_severe_consequence'))))
            .alias('variantEffect'),
            map_column_by_dictionary(f.col('most_severe_consequence'), SEQUENCE_ONTOLOGY).alias(
                'mostSevereConsequenceId'
            ),
            'most_severe_consequence',
            f
            .when(f.size(transcripts) > 0, transcripts[0]['hgvsg'])
            .when(f.size('intergenic_consequences') > 0, f.col('intergenic_consequences')[0]['hgvsg'])
            .alias('hgvsId'),
            f.when(transcripts.isNotNull(), f.transform(transcripts, transcript_consequence)).alias(
                'transcriptConsequences'
            ),
            colocated_rsids(colocated).alias('rsIds'),
            f.array().cast(VARIANT_SCHEMA['alleleFrequencies'].dataType).alias('alleleFrequencies'),
        )
        # Transcripts without a consequence score cannot be ranked, as in Gentropy:
        .withColumn(
            'transcriptConsequences',
            f.filter(
                'transcriptConsequences',
                lambda x: x.consequenceScore.isNotNull() & x.distanceFromFootprint.isNotNull(),
            ),
        )
        .withColumn(
            'transcriptConsequences',
            f.transform(
                order_structs_by_two_fields('transcriptConsequences', 'consequenceScore', 'distanceFromFootprint'),
                lambda x, i: x.withField('transcriptIndex', (i + 1).cast(t.IntegerType())),
            ),
        )
        # The protvar cross-reference uses the identifier before long ones are hashed:
        .withColumn(
            '_protvar_xrefs',
            f.when(
                f.size(
                    f.filter(
                        'transcriptConsequences',
                        lambda x: f.array_contains(x.variantFunctionalConsequenceIds, MISSENSE),
                    )
                )
                > 0,
                generate_dbxrefs(f.array('variantId'), 'protvar'),
            ),
        )
        .withColumn(
            'dbXrefs',
            f.flatten(
                f.filter(
                    f.array('_ensembl_xrefs', '_omim_xrefs', '_clinvar_xrefs', '_protvar_xrefs'),
                    lambda x: x.isNotNull(),
                )
            ),
        )
        .withColumn(
            'variantId',
            hash_long_variant_ids(f.col('variantId'), f.col('chromosome'), f.col('position'), hash_threshold),
        )
        .withColumn('_protein_coding', f.filter('transcriptConsequences', lambda x: x.biotype == 'protein_coding'))
        .withColumn(
            'variantDescription',
            variant_description(
                f.col('most_severe_consequence'),
                f.filter('transcriptConsequences', lambda x: x.transcriptIndex == 1)[0],
                order_structs_by_field('_protein_coding', 'transcriptIndex')[f.size('_protein_coding') - 1],
            ),
        )
        .withColumn('variantEffect', normalise_variant_effects(f.col('variantEffect')))
        .filter(f.col('position').isNotNull())
    )
    return fill_null_arrays(parsed.select(*VARIANT_SCHEMA.fieldNames()))
