"""Parse Ensembl VEP JSON output into the variant index, ported from Gentropy's VEP parser.

Differences from Gentropy: consequence terms use the corrected Sequence Ontology spelling
(see `consequences.py`), SIFT and PolyPhen are not extracted (VEP runs without them), OMIM
variant suffixes keep their four digits, and only VEP JSON parts are read, never VEP's
`<part>.json_warnings.txt` files.
"""

from __future__ import annotations

import re

import polars as pl

from pts.schemas.variant import variant_schema
from pts.schemas.vep import vep_schema
from pts.transformers.utils.dataset import list_parts
from pts.transformers.utils.variant import hash_variant_index_ids
from pts.transformers.variant_utils.consequences import CONSEQUENCE_SCORES, SEQUENCE_ONTOLOGY
from pts.transformers.variant_utils.index import fill_null_arrays, normalise_score

#: VEP data parts, optionally gzipped; excludes VEP's `<part>.json_warnings.txt`.
VEP_PART = re.compile(r'\.jsonl?(\.gz)?$')
MISSENSE = 'SO_0001583'

_xref_dtype = variant_schema['dbXrefs']


def list_vep_parts(path: str) -> list[str]:
    """The VEP JSON parts below `path`, or `path` itself when it is a file."""
    parts = [part for part in list_parts(path, '**/*.json*') if VEP_PART.search(part)]
    if not parts:
        raise ValueError(f'No VEP JSON files found in {path}')
    return parts


def read_vep(path: str) -> pl.LazyFrame:
    """Scan one VEP JSON part with the fields the index reads."""
    return pl.scan_ndjson(path, schema=vep_schema)


def generate_dbxrefs(ids: pl.Expr, source: str) -> pl.Expr:
    """Turn a list of identifiers into cross-references, dropping null identifiers."""
    return ids.fill_null([]).list.eval(
        pl.struct(pl.element().alias('id'), pl.lit(source).alias('source')).filter(pl.element().is_not_null())
    )


def omim_ids(colocated: pl.Expr) -> pl.Expr:
    """OMIM cross-references, written `<entry>#<variant>` as on omim.org.

    The variant suffix is padded to its four digits, which VEP's numbers do not always keep
    (`600234.001` -> `600234#0010`).
    """
    synonyms = colocated.list.eval(
        pl.element().struct.field('var_synonyms').struct.field('OMIM').drop_nulls().explode()
    )
    parts = pl.element().str.split_exact('.', 1)
    entry, suffix = parts.struct.field('field_0'), parts.struct.field('field_1')
    return synonyms.list.eval(
        pl.when(suffix.is_null()).then(entry).otherwise(pl.concat_str(entry, pl.lit('#'), suffix.str.pad_end(4, '0')))
    )


def clinvar_ids(colocated: pl.Expr) -> pl.Expr:
    """ClinVar variation (VCV) identifiers."""
    return colocated.list.eval(
        pl.element().struct.field('var_synonyms').struct.field('ClinVar').drop_nulls().explode()
    ).list.eval(pl.element().filter(pl.element().str.starts_with('VCV')))


def _highest(transcripts: pl.Expr, field: str) -> pl.Expr:
    """The first transcript with the highest non-null `field`."""
    # `list.arg_max` returns the last maximum for strings, not the first.
    values = transcripts.list.eval(pl.element().struct.field(field))
    first = values.list.eval((pl.element() == pl.element().max()).arg_true().first())
    return transcripts.list.get(first.list.first(), null_on_oob=True)


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


def variant_effect(
    transcripts: pl.Expr, method: str, score: str, assessment: str | None = None, flag: str | None = None
) -> pl.Expr:
    """The predictor's effect from the transcript with its highest score, if any."""
    transcript = _highest(transcripts, score)

    def field(name: str | None) -> pl.Expr:
        return pl.lit(None, dtype=pl.String) if name is None else transcript.struct.field(name).cast(pl.String)

    return pl.when(transcript.is_not_null()).then(
        _effect(
            method, field(assessment), transcript.struct.field(score), field(flag), transcript.struct.field('gene_id')
        )
    )


def alpha_missense_effect(transcripts: pl.Expr) -> pl.Expr:
    """AlphaMissense from the first transcript that has a prediction."""
    transcript = transcripts.list.eval(
        pl.element().filter(pl.element().struct.field('alphamissense').is_not_null())
    ).list.first()
    prediction = transcript.struct.field('alphamissense')
    return pl.when(transcript.is_not_null()).then(
        _effect(
            'AlphaMissense',
            prediction.struct.field('am_class'),
            prediction.struct.field('am_pathogenicity'),
            pl.lit(None, dtype=pl.String),
            transcript.struct.field('gene_id'),
        )
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


def transcript_consequences(transcripts: pl.Expr) -> pl.Expr:
    """Transcript consequences ranked by severity, then distance; unscored transcripts are dropped."""

    def field(name: str) -> pl.Expr:
        return pl.element().struct.field(name)

    terms = field('consequence_terms')
    consequences = transcripts.list.eval(
        pl.struct(
            terms.list.eval(pl.element().replace_strict(SEQUENCE_ONTOLOGY, default=None, return_dtype=pl.String)).alias(
                'variantFunctionalConsequenceIds'
            ),
            amino_acid_change(field('amino_acids'), field('protein_end')).alias('aminoAcidChange'),
            uniprot_accessions(field('swissprot'), field('trembl')).alias('uniprotAccessions'),
            (field('canonical') == 1).fill_null(False).alias('isEnsemblCanonical'),
            field('codons').alias('codons'),
            field('distance').fill_null(0).alias('distanceFromFootprint'),
            field('tssdistance').alias('distanceFromTss'),
            field('appris').alias('appris'),
            field('mane_select').alias('maneSelect'),
            field('gene_id').alias('targetId'),
            field('impact').alias('impact'),
            field('lof').alias('lofteePrediction'),
            terms.list
            .eval(pl.element().replace_strict(CONSEQUENCE_SCORES, default=None, return_dtype=pl.Float64))
            .list.max()
            .cast(pl.Float32)
            .alias('consequenceScore'),
            pl.lit(None, dtype=pl.Int32).alias('transcriptIndex'),
            field('gene_symbol').alias('approvedSymbol'),
            field('biotype').alias('biotype'),
            field('transcript_id').alias('transcriptId'),
        )
    )
    return (
        consequences.list
        .eval(pl.element().filter(pl.element().struct.field('consequenceScore').is_not_null()))
        # Two stable sorts: a multi-key `sort_by` inside `list.eval` fails on empty lists.
        .list.eval(pl.element().sort_by(pl.element().struct.field('distanceFromFootprint'), maintain_order=True))
        .list.eval(
            pl.element().sort_by(pl.element().struct.field('consequenceScore'), descending=True, maintain_order=True)
        )
        .list.eval(
            pl.element().struct.with_fields(pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias('transcriptIndex'))
        )
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


def parse_vep(frame: pl.LazyFrame, hash_threshold: int = 300) -> pl.LazyFrame:
    """Turn raw VEP records into variant index rows, keeping canonical transcripts only."""
    fields = pl.col('input').str.split('\t')
    colocated = pl.col('colocated_variants')
    transcripts = pl.col('transcript_consequences')
    consequence = pl.col('most_severe_consequence')
    rsids = colocated.list.eval(pl.element().struct.field('id'))
    effects = pl.concat_list(
        variant_effect(transcripts, 'CADD', 'cadd_phred'),
        variant_effect(transcripts, 'LOFTEE', 'lof', 'lof', 'lof_filter'),
        variant_effect(transcripts, 'GERP', 'conservation'),
        alpha_missense_effect(transcripts),
        vep_prediction(consequence),
    ).list.drop_nulls()
    missense = (
        pl
        .col('transcriptConsequences')
        .list.eval(pl.element().struct.field('variantFunctionalConsequenceIds').list.contains(MISSENSE))
        .list.any()
    )
    return (
        frame
        .with_columns(transcripts.list.filter(pl.element().struct.field('canonical') == 1))
        .select(
            pl.concat_str(
                [fields.list.get(i, null_on_oob=True) for i in (0, 1, 3, 4)], separator='_', ignore_nulls=True
            ).alias('variantId'),
            fields.list.get(0, null_on_oob=True).alias('chromosome'),
            fields.list.get(1, null_on_oob=True).cast(pl.Int32, strict=False).alias('position'),
            fields.list.get(3, null_on_oob=True).alias('referenceAllele'),
            fields.list.get(4, null_on_oob=True).alias('alternateAllele'),
            effects.alias('variantEffect'),
            consequence.replace_strict(SEQUENCE_ONTOLOGY, default=None, return_dtype=pl.String).alias(
                'mostSevereConsequenceId'
            ),
            consequence,
            pl
            .when(transcripts.list.len() > 0)
            .then(transcripts.list.first().struct.field('hgvsg'))
            .when(pl.col('intergenic_consequences').list.len() > 0)
            .then(pl.col('intergenic_consequences').list.first().struct.field('hgvsg'))
            .alias('hgvsId'),
            transcript_consequences(transcripts).alias('transcriptConsequences'),
            rsids.alias('rsIds'),
            pl.lit([], dtype=variant_schema['alleleFrequencies']).alias('alleleFrequencies'),
            generate_dbxrefs(rsids, 'ensembl_variation').alias('_ensembl'),
            generate_dbxrefs(omim_ids(colocated), 'omim').alias('_omim'),
            generate_dbxrefs(clinvar_ids(colocated), 'clinvar').alias('_clinvar'),
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
            hash_variant_index_ids(pl.col('variantId'), pl.col('chromosome'), pl.col('position'), hash_threshold).alias(
                'variantId'
            )
        )
        .filter(pl.col('position').is_not_null())
        .select(variant_schema.names())
        .pipe(fill_null_arrays)
    )
