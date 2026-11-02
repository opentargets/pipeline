"""Build the variant index rows from a chunk's variants and consequences frames, ported from Gentropy's VEP parser.

Differences from Gentropy: consequence terms use the corrected Sequence Ontology spelling
(see `consequences.py`), SIFT comes from the canonical transcript with the lowest score rather
than the highest, PolyPhen is not extracted (VEP runs without it), and OMIM variant suffixes keep
their four digits.
"""

from __future__ import annotations

import polars as pl

from pts.schemas.variant import variant_schema
from pts.transformers.utils.variant import hash_long_variant_ids
from pts.transformers.variant.annotators.sift import SEVERITY
from pts.transformers.variant.index import fill_null_arrays, normalise_score
from pts.transformers.variant.vep.consequences import CONSEQUENCE_SCORES, SEQUENCE_ONTOLOGY, consequence_score
from pts.transformers.variant.vep.frames import CONSEQUENCES_SCHEMA, EFFECTS_SCHEMA, ID_LISTS, VARIANTS_SCHEMA, require

MISSENSE = 'SO_0001583'

_xref_dtype = variant_schema['dbXrefs']


def generate_dbxrefs(ids: pl.Expr, source: str) -> pl.Expr:
    """Turn a list of identifiers into cross-references, dropping null identifiers."""
    return ids.fill_null([]).list.eval(
        pl.struct(pl.element().alias('id'), pl.lit(source).alias('source')).filter(pl.element().is_not_null())
    )


def _picked(consequences: pl.DataFrame, effect: pl.Expr) -> pl.DataFrame:
    """`row` and the `effect` of the first consequence of each row."""
    return consequences.unique('row', keep='first', maintain_order=True).select('row', effect)


def _highest(consequences: pl.DataFrame, field: str) -> pl.DataFrame:
    """The consequences with the highest non-null `field` of their row, in the index's order."""
    value = pl.col(field)
    return consequences.filter(value == value.max().over('row'))


def _effect(
    method: str,
    assessment: pl.Expr,
    score: pl.Expr,
    flag: pl.Expr,
    target: pl.Expr,
) -> pl.Expr:
    score = score.cast(pl.Float32, strict=False)
    return pl.struct(
        pl.lit(method).alias('method'),
        assessment.alias('assessment'),
        score.alias('score'),
        flag.alias('assessmentFlag'),
        target.alias('targetId'),
        normalise_score(method, score, assessment).alias('normalisedScore'),
    )


def _variant_effect(
    consequences: pl.DataFrame, method: str, score: str, assessment: str | None = None, flag: str | None = None
) -> pl.DataFrame:
    """`row` and the predictor's effect, as `method`, from the first transcript with its row's highest score.

    Takes consequences in (row, rank) order.
    """

    def field(name: str | None) -> pl.Expr:
        return pl.lit(None, dtype=pl.String) if name is None else pl.col(name).cast(pl.String)

    return _picked(
        _highest(consequences, score),
        _effect(method, field(assessment), pl.col(score), field(flag), pl.col('gene_id')).alias(method),
    )


def _sift_effect(consequences: pl.DataFrame) -> pl.DataFrame:
    """`row` and SIFT from the transcript with the lowest score, then the more severe prediction, then the lowest id.

    Takes consequences in (row, rank) order.
    """
    severity = {prediction: rank for rank, prediction in enumerate(SEVERITY)}
    transcripts = consequences.filter(pl.col('sift_score').is_not_null()).sort(
        'row',
        'sift_score',
        pl.col('sift_prediction').replace_strict(severity, default=len(severity), return_dtype=pl.Int32),
        'transcript_id',
        maintain_order=True,
    )
    return _picked(
        transcripts,
        _effect(
            'SIFT', pl.col('sift_prediction'), pl.col('sift_score'), pl.lit(None, dtype=pl.String), pl.col('gene_id')
        ).alias('SIFT'),
    )


def _alpha_missense_effect(consequences: pl.DataFrame) -> pl.DataFrame:
    """`row` and AlphaMissense from the first transcript that has a prediction.

    Takes consequences in (row, rank) order.
    """
    return _picked(
        consequences.filter(pl.col('am_class').is_not_null()),
        _effect(
            'AlphaMissense',
            pl.col('am_class'),
            pl.col('am_pathogenicity'),
            pl.lit(None, dtype=pl.String),
            pl.col('gene_id'),
        ).alias('AlphaMissense'),
    )


def vep_prediction(consequence: pl.Expr) -> pl.Expr:
    """VEP's own prediction: the most severe consequence and its score."""
    return _effect(
        'VEP',
        consequence,
        consequence.replace_strict(CONSEQUENCE_SCORES, default=None, return_dtype=pl.Float64),
        pl.lit(None, dtype=pl.String),
        pl.lit(None, dtype=pl.String),
    )


def amino_acid_change(amino_acids: pl.Expr, protein_end: pl.Expr) -> pl.Expr:
    """One-letter substitution such as `A12B` from VEP's `A/B` and the protein position."""
    parts = amino_acids.str.split('/')
    return pl.concat_str(
        parts.list.get(0, null_on_oob=True), protein_end.cast(pl.String), parts.list.get(1, null_on_oob=True)
    )


def uniprot_accessions(swissprot: pl.Expr, trembl: pl.Expr) -> pl.Expr:
    """Distinct Swiss-Prot and TrEMBL accessions without version suffixes."""
    return (
        pl
        .concat_list(swissprot.fill_null([]), trembl.fill_null([]))
        .list.eval(pl.element().str.split('.').list.first())
        .list.unique(maintain_order=True)
        .list.drop_nulls()
    )


def _transcript_consequences(consequences: pl.DataFrame) -> pl.DataFrame:
    """`row` and its transcript consequences in the index's order, numbered; unscored transcripts are dropped.

    Takes consequences in (row, rank) order.
    """
    terms = pl.col('consequence_terms')
    score = consequence_score(terms).cast(pl.Float32)
    return (
        consequences
        .filter(score.is_not_null())
        .select(
            'row',
            pl.struct(
                terms.list.eval(
                    pl.element().replace_strict(SEQUENCE_ONTOLOGY, default=None, return_dtype=pl.String)
                ).alias('variantFunctionalConsequenceIds'),
                amino_acid_change(pl.col('amino_acids'), pl.col('protein_end')).alias('aminoAcidChange'),
                uniprot_accessions(pl.col('swissprot'), pl.col('trembl')).alias('uniprotAccessions'),
                pl.lit(True).alias('isEnsemblCanonical'),
                pl.col('codons').alias('codons'),
                pl.col('distance').fill_null(0).alias('distanceFromFootprint'),
                pl.col('tssdistance').alias('distanceFromTss'),
                pl.col('appris').alias('appris'),
                pl.col('mane_select').alias('maneSelect'),
                pl.col('gene_id').alias('targetId'),
                pl.col('impact').alias('impact'),
                pl.col('lof').alias('lofteePrediction'),
                score.alias('consequenceScore'),
                pl.int_range(1, pl.len() + 1, dtype=pl.Int32).over('row').alias('transcriptIndex'),
                pl.col('gene_symbol').alias('approvedSymbol'),
                pl.col('biotype').alias('biotype'),
                pl.col('transcript_id').alias('transcriptId'),
            ).alias('transcriptConsequences'),
        )
        .group_by('row', maintain_order=True)
        .agg('transcriptConsequences')
    )


def _format_number(value: pl.Expr) -> pl.Expr:
    """Spark's `format_number(value, 0)` for whole numbers: thousands separated by commas."""
    digits = value.abs().cast(pl.String).str.reverse().str.replace_all(r'(\d{3})', '${1},').str.strip_chars_end(',')
    return pl.concat_str(pl.when(value < 0).then(pl.lit('-')).otherwise(pl.lit('')), digits.str.reverse())


def _consequence_text(consequence: pl.Expr) -> pl.Expr:
    # Spark's initcap splits words on spaces only, so a term keeps one capital letter.
    capitalised = pl.concat_str(
        consequence.str.slice(0, 1).str.to_uppercase(), consequence.str.slice(1).str.to_lowercase()
    )
    suffix = pl.when(consequence.str.contains('variant', literal=True)).then(pl.lit('')).otherwise(pl.lit(' variant'))
    return pl.concat_str(capitalised.str.replace_all('_', ' ', literal=True), suffix)


def _gene_label(transcript: pl.Expr) -> pl.Expr:
    return pl.coalesce(transcript.struct.field('approvedSymbol'), transcript.struct.field('targetId'))


def _overlap_text(transcript: pl.Expr) -> pl.Expr:
    distance = transcript.struct.field('distanceFromFootprint')
    return (
        pl
        .when(distance == 0)
        .then(pl.concat_str(pl.lit(' overlapping with '), _gene_label(transcript)))
        .otherwise(
            pl.concat_str(
                pl.lit(' '), _format_number(distance), pl.lit(' basepair away from '), _gene_label(transcript)
            )
        )
    )


def variant_description(consequence: pl.Expr, transcripts: pl.Expr) -> pl.Expr:
    """A sentence describing the variant from its most severe consequence and closest genes."""
    first = transcripts.list.first()
    protein_coding = transcripts.list.eval(
        pl.element().filter(pl.element().struct.field('biotype') == 'protein_coding')
    ).list.first()
    change = first.struct.field('aminoAcidChange')
    amino_acid = (
        pl
        .when(change.is_not_null())
        .then(
            pl.concat_str(
                pl.lit(', causing amino-acid change: '),
                change,
                pl.lit(' with '),
                first.struct.field('impact').str.to_lowercase(),
                pl.lit(' impact.'),
            )
        )
        .otherwise(pl.lit('.'))
    )
    loftee = (
        pl
        .when(first.struct.field('lofteePrediction') == 'HC')
        .then(pl.lit(' A high-confidence loss-of-function variant by loftee.'))
        .otherwise(pl.lit(''))
    )
    biotype = first.struct.field('biotype')
    biotype_text = pl.concat_str(
        pl.lit(', a '),
        biotype.str.replace_all('_', ' ', literal=True),
        pl.when(biotype.str.contains('gene', literal=True)).then(pl.lit('')).otherwise(pl.lit(' gene.')),
    )
    closest_protein_coding = (
        pl
        .when(protein_coding.is_not_null())
        .then(
            pl.concat_str(
                pl.lit(' The closest protein-coding gene is '),
                _gene_label(protein_coding),
                pl.lit(' ('),
                _format_number(protein_coding.struct.field('distanceFromFootprint')),
                pl.lit(' basepair away).'),
            )
        )
        .otherwise(pl.lit(''))
    )
    return (
        pl
        .when(first.is_null())
        .then(pl.lit('Intergenic variant no gene in window'))
        .when(biotype == 'protein_coding')
        .then(pl.concat_str(_consequence_text(consequence), _overlap_text(first), amino_acid, loftee))
        .otherwise(
            pl.concat_str(_consequence_text(consequence), _overlap_text(first), biotype_text, closest_protein_coding)
        )
    )


def index_rows(variants: pl.DataFrame, consequences: pl.DataFrame, hash_threshold: int = 300) -> pl.DataFrame:
    """The variant index rows of a chunk, one per variant with a position, in row order.

    `variants` carries the `VARIANTS_SCHEMA` columns and the `ID_LISTS` of `add_colocated`, and `consequences` the
    `CONSEQUENCES_SCHEMA` and `EFFECTS_SCHEMA` columns. Consequences are sorted into (row, rank) order here, once,
    and the effect builders take them in that order.
    Effects whose score ties between transcripts come from the transcript ranked first.
    """
    require(variants, VARIANTS_SCHEMA | ID_LISTS, 'variants')
    require(consequences, CONSEQUENCES_SCHEMA | EFFECTS_SCHEMA, 'consequences')
    consequence = pl.col('most_severe_consequence')
    consequences = consequences.sort('row', 'rank', maintain_order=True)
    effects = [
        _sift_effect(consequences),
        _variant_effect(consequences, 'LOFTEE', 'lof', 'lof', 'lof_filter'),
        _variant_effect(consequences, 'GERP', 'conservation'),
        _alpha_missense_effect(consequences),
        _transcript_consequences(consequences),
    ]
    frame = variants
    for found in effects:
        frame = frame.join(found, on='row', how='left', maintain_order='left')
    frame = frame.rechunk()
    canonical = pl.col('row').is_in(consequences['row'].unique().implode())
    missense = (
        pl
        .col('transcriptConsequences')
        .list.eval(pl.element().struct.field('variantFunctionalConsequenceIds').list.contains(MISSENSE))
        .list.any()
    )
    result = (
        frame
        .select(
            pl.concat_str(
                pl.col('chromosome'),
                pl.col('position').cast(pl.String),
                pl.col('referenceAllele'),
                pl.col('alternateAllele'),
                separator='_',
                ignore_nulls=True,
            ).alias('variantId'),
            'chromosome',
            pl.col('position').cast(pl.Int32),
            'referenceAllele',
            'alternateAllele',
            pl
            .concat_list('SIFT', 'LOFTEE', 'GERP', 'AlphaMissense', vep_prediction(consequence))
            .list.drop_nulls()
            .alias('variantEffect'),
            consequence.replace_strict(SEQUENCE_ONTOLOGY, default=None, return_dtype=pl.String).alias(
                'mostSevereConsequenceId'
            ),
            consequence,
            pl.when(canonical | ~pl.col('reported_transcripts')).then(pl.col('hgvsg')).alias('hgvsId'),
            'transcriptConsequences',
            'rsIds',
            pl.lit([], dtype=variant_schema['alleleFrequencies']).alias('alleleFrequencies'),
            generate_dbxrefs(pl.col('rsIds'), 'ensembl_variation').alias('_ensembl'),
            generate_dbxrefs(pl.col('omimIds'), 'omim').alias('_omim'),
            generate_dbxrefs(pl.col('clinvarIds'), 'clinvar').alias('_clinvar'),
        )
        # The protvar cross-reference uses the identifier before long ones are hashed:
        .with_columns(
            pl.concat_list(
                '_ensembl',
                '_omim',
                '_clinvar',
                pl
                .when(missense)
                .then(generate_dbxrefs(pl.concat_list('variantId'), 'protvar'))
                .otherwise(pl.lit([], dtype=_xref_dtype)),
            ).alias('dbXrefs'),
            variant_description(consequence, pl.col('transcriptConsequences')).alias('variantDescription'),
        )
        .with_columns(
            hash_long_variant_ids(pl.col('variantId'), pl.col('chromosome'), pl.col('position'), hash_threshold).alias(
                'variantId'
            )
        )
        .filter(pl.col('position').is_not_null())
        .select(variant_schema.names())
    )
    return fill_null_arrays(result)
