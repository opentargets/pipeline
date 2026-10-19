"""Lazy parsing of Variant Effect Predictor JSON annotations."""

from __future__ import annotations

import polars as pl
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.variant import variant_schema
from pts.schemas.vep import vep_schema
from pts.transformers.utils.variant import hash_variant_index_ids
from pts.transformers.variant_utils.annotations import normalize_variant_effects

# Gentropy's table misspells SO_0001626 as `incomplete_labelinal_codon_variant`, so VEP's
# `incomplete_terminal_codon_variant` gets no SO ID or score there. Both tables use the
# correct term, so this output intentionally differs from Gentropy for that consequence.
SEQUENCE_ONTOLOGY = {
    'transcript_ablation': 'SO_0001893',
    'splice_acceptor_variant': 'SO_0001574',
    'splice_donor_variant': 'SO_0001575',
    'stop_gained': 'SO_0001587',
    'frameshift_variant': 'SO_0001589',
    'stop_lost': 'SO_0001578',
    'start_lost': 'SO_0002012',
    'transcript_amplification': 'SO_0001889',
    'feature_elongation': 'SO_0001907',
    'feature_truncation': 'SO_0001906',
    'inframe_insertion': 'SO_0001821',
    'inframe_deletion': 'SO_0001822',
    'missense_variant': 'SO_0001583',
    'protein_altering_variant': 'SO_0001818',
    'splice_donor_5th_base_variant': 'SO_0001787',
    'splice_region_variant': 'SO_0001630',
    'splice_donor_region_variant': 'SO_0002170',
    'splice_polypyrimidine_tract_variant': 'SO_0002169',
    'incomplete_terminal_codon_variant': 'SO_0001626',
    'start_retained_variant': 'SO_0002019',
    'stop_retained_variant': 'SO_0001567',
    'synonymous_variant': 'SO_0001819',
    'coding_sequence_variant': 'SO_0001580',
    'mature_miRNA_variant': 'SO_0001620',
    '5_prime_UTR_variant': 'SO_0001623',
    '3_prime_UTR_variant': 'SO_0001624',
    'non_coding_transcript_exon_variant': 'SO_0001792',
    'intron_variant': 'SO_0001627',
    'NMD_transcript_variant': 'SO_0001621',
    'non_coding_transcript_variant': 'SO_0001619',
    'coding_transcript_variant': 'SO_0001968',
    'upstream_gene_variant': 'SO_0001631',
    'downstream_gene_variant': 'SO_0001632',
    'TFBS_ablation': 'SO_0001895',
    'TFBS_amplification': 'SO_0001892',
    'TF_binding_site_variant': 'SO_0001782',
    'regulatory_region_ablation': 'SO_0001894',
    'regulatory_region_amplification': 'SO_0001891',
    'regulatory_region_variant': 'SO_0001566',
    'intergenic_variant': 'SO_0001628',
    'sequence_variant': 'SO_0001060',
}
CONSEQUENCE_SCORES = {
    'transcript_ablation': 0.98,
    'splice_acceptor_variant': 0.95,
    'splice_donor_variant': 0.93,
    'stop_gained': 0.9,
    'frameshift_variant': 0.88,
    'stop_lost': 0.85,
    'start_lost': 0.83,
    'transcript_amplification': 0.8,
    'feature_elongation': 0.78,
    'feature_truncation': 0.76,
    'inframe_insertion': 0.73,
    'inframe_deletion': 0.71,
    'missense_variant': 0.68,
    'protein_altering_variant': 0.66,
    'splice_donor_5th_base_variant': 0.63,
    'splice_region_variant': 0.61,
    'splice_donor_region_variant': 0.59,
    'splice_polypyrimidine_tract_variant': 0.56,
    'incomplete_terminal_codon_variant': 0.54,
    'start_retained_variant': 0.51,
    'stop_retained_variant': 0.49,
    'synonymous_variant': 0.46,
    'coding_sequence_variant': 0.44,
    'mature_miRNA_variant': 0.41,
    '5_prime_UTR_variant': 0.39,
    '3_prime_UTR_variant': 0.37,
    'non_coding_transcript_exon_variant': 0.34,
    'intron_variant': 0.32,
    'NMD_transcript_variant': 0.29,
    'non_coding_transcript_variant': 0.27,
    'coding_transcript_variant': 0.24,
    'upstream_gene_variant': 0.22,
    'downstream_gene_variant': 0.2,
    'TFBS_ablation': 0.17,
    'TFBS_amplification': 0.15,
    'TF_binding_site_variant': 0.12,
    'regulatory_region_ablation': 0.1,
    'regulatory_region_amplification': 0.07,
    'regulatory_region_variant': 0.05,
    'intergenic_variant': 0.02,
    'sequence_variant': 0.0,
}


def read_vep(path: str) -> pl.LazyFrame:
    """Scan recursive VEP JSON parts with fixed nested types, including gzip parts."""
    handle = StorageHandle(path)
    if handle.stat().is_dir:
        parts = sorted(handle.glob('**/*.json*'))
        if not parts:
            raise ValueError(f'No VEP JSON files found in {path}')
    else:
        parts = [path]
    frame = pl.scan_ndjson(parts, schema=vep_schema)
    if frame.limit(1).collect().is_empty():
        raise ValueError(f'The VEP dataset is empty: {path}')
    return frame


def generate_dbxrefs(ids: pl.Expr, source: str) -> pl.Expr:
    """Build typed cross references, dropping null identifiers."""
    return ids.fill_null([]).list.eval(
        pl.struct(pl.element().cast(pl.String).alias('id'), pl.lit(source).alias('source')).filter(
            pl.element().is_not_null()
        )
    )


def amino_acid_change(amino_acids: pl.Expr, protein_end: pl.Expr) -> pl.Expr:
    """Format a one-letter amino acid substitution when both inputs are available."""
    parts = amino_acids.str.split('/')
    return pl.concat_str(
        parts.list.get(0, null_on_oob=True),
        protein_end.cast(pl.String),
        parts.list.get(1, null_on_oob=True),
    )


def uniprot_accessions(swissprot: pl.Expr, trembl: pl.Expr) -> pl.Expr:
    """Merge accession arrays, remove versions and preserve first occurrence order."""
    return (
        pl
        .concat_list(swissprot.fill_null([]), trembl.fill_null([]))
        .list.eval(pl.element().str.split('.').list.first().filter(pl.element().is_not_null()))
        .list.unique(maintain_order=True)
    )


def most_severe_transcript(transcripts: pl.Expr, score_field: str) -> pl.Expr:
    """Pick the highest non-null prediction, preserving source order on ties."""
    return (
        transcripts.list
        .eval(
            pl.element().sort_by(
                pl.element().struct.field(score_field), descending=True, nulls_last=True, maintain_order=True
            )
        )
        .list.eval(pl.element().filter(pl.element().struct.field(score_field).is_not_null()))
        .list.first()
    )


def variant_effect(
    transcript: pl.Expr,
    method: str,
    score: str,
    assessment: str | None = None,
    flag: str | None = None,
) -> pl.Expr:
    """Extract a prediction into the complete variant-effect struct."""

    def field(name: str | None, dtype: pl.DataType | type[pl.DataType]) -> pl.Expr:
        return pl.lit(None, dtype=dtype) if name is None else transcript.struct.field(name).cast(dtype, strict=False)

    return pl.when(transcript.is_not_null()).then(
        pl.struct(
            pl.lit(method).alias('method'),
            field(assessment, pl.String).alias('assessment'),
            field(score, pl.Float32).alias('score'),
            field(flag, pl.String).alias('assessmentFlag'),
            transcript.struct.field('gene_id').alias('targetId'),
            pl.lit(None, dtype=pl.Float64).alias('normalisedScore'),
        )
    )


def vep_prediction(consequence: pl.Expr) -> pl.Expr:
    """Build the consequence prediction including unmapped consequence terms."""
    return pl.struct(
        pl.lit('VEP').alias('method'),
        consequence.alias('assessment'),
        consequence.replace_strict(CONSEQUENCE_SCORES, default=None, return_dtype=pl.Float32).alias('score'),
        pl.lit(None, dtype=pl.String).alias('assessmentFlag'),
        pl.lit(None, dtype=pl.String).alias('targetId'),
        pl.lit(None, dtype=pl.Float64).alias('normalisedScore'),
    )


def transcript_consequences(transcripts: pl.Expr) -> pl.Expr:
    """Map canonical transcript consequences and rank by severity and proximity."""
    t = pl.element()

    def field(name: str) -> pl.Expr:
        return t.struct.field(name)

    result = transcripts.list.eval(
        pl.struct(
            field('consequence_terms')
            .list.eval(pl.element().replace_strict(SEQUENCE_ONTOLOGY, default=None, return_dtype=pl.String))
            .alias('variantFunctionalConsequenceIds'),
            amino_acid_change(field('amino_acids'), field('protein_end')).alias('aminoAcidChange'),
            uniprot_accessions(field('swissprot'), field('trembl')).alias('uniprotAccessions'),
            (field('canonical') == 1).fill_null(False).alias('isEnsemblCanonical'),
            field('codons').alias('codons'),
            field('distance').fill_null(0).cast(pl.Int64).alias('distanceFromFootprint'),
            field('tssdistance').cast(pl.Int64).alias('distanceFromTss'),
            field('appris').alias('appris'),
            field('mane_select').alias('maneSelect'),
            field('gene_id').alias('targetId'),
            field('impact').alias('impact'),
            field('lof').cast(pl.String).alias('lofteePrediction'),
            # Gentropy fills both from `lof` (only 'HC'/'LC'), so they are always null there.
            # This intentionally differs: they carry the transcript's SIFT and PolyPhen scores.
            # PolyPhen stays null until VEP runs with --polyphen (annotate_variants.sh omits it).
            field('sift_score').cast(pl.Float32).alias('siftPrediction'),
            field('polyphen_score').cast(pl.Float32).alias('polyphenPrediction'),
            field('consequence_terms')
            .list.eval(pl.element().replace_strict(CONSEQUENCE_SCORES, default=None, return_dtype=pl.Float32))
            .list.max()
            .alias('consequenceScore'),
            field('canonical').cast(pl.Int32).alias('transcriptIndex'),
            field('gene_symbol').alias('approvedSymbol'),
            field('biotype').alias('biotype'),
            field('transcript_id').alias('transcriptId'),
        )
    )
    result = (
        result.list
        .eval(
            pl.element().filter(
                pl.element().struct.field('consequenceScore').is_not_null()
                & pl.element().struct.field('distanceFromFootprint').is_not_null()
            )
        )
        .list.eval(
            pl.element().sort_by(
                pl.element().struct.field('distanceFromFootprint'),
                maintain_order=True,
            )
        )
        .list.eval(
            pl.element().sort_by(pl.element().struct.field('consequenceScore'), descending=True, maintain_order=True)
        )
    )
    return result.list.eval(
        pl.element().struct.with_fields(pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias('transcriptIndex'))
    )


def _format_distance(distance: pl.Expr) -> pl.Expr:
    """Format integer distances with thousands separators."""
    return (
        distance
        .cast(pl.String)
        .str.reverse()
        .str.replace_all(r'(\d{3})', '${1},')
        .str.strip_chars_end(',')
        .str.reverse()
    )


def variant_description(consequence: pl.Expr, transcripts: pl.Expr) -> pl.Expr:
    """Describe the first consequence and the first protein coding transcript."""
    first = transcripts.list.first()
    protein = transcripts.list.eval(
        pl.element().filter(pl.element().struct.field('biotype') == 'protein_coding')
    ).list.first()

    def field(t: pl.Expr, name: str) -> pl.Expr:
        return t.struct.field(name)

    def label(t: pl.Expr) -> pl.Expr:
        return pl.coalesce(field(t, 'approvedSymbol'), field(t, 'targetId'))

    # Spark initcap treats underscores as word characters, so only the first letter is capitalized.
    term = pl.concat_str(consequence.str.slice(0, 1).str.to_uppercase(), consequence.str.slice(1).str.to_lowercase())
    term = pl.concat_str(
        term.str.replace_all('_', ' '),
        pl.when(consequence.str.contains('variant', literal=True)).then(pl.lit('')).otherwise(pl.lit(' variant')),
    )
    overlap = (
        pl
        .when(field(first, 'distanceFromFootprint') == 0)
        .then(pl.concat_str(pl.lit(' overlapping with '), label(first)))
        .otherwise(
            pl.concat_str(
                pl.lit(' '),
                _format_distance(field(first, 'distanceFromFootprint')),
                pl.lit(' basepair away from '),
                label(first),
            )
        )
    )
    aa = (
        pl
        .when(field(first, 'aminoAcidChange').is_not_null())
        .then(
            pl.concat_str(
                pl.lit(', causing amino-acid change: '),
                field(first, 'aminoAcidChange'),
                pl.lit(' with '),
                field(first, 'impact').str.to_lowercase(),
                pl.lit(' impact.'),
            )
        )
        .otherwise(pl.lit('.'))
    )
    lof = (
        pl
        .when(field(first, 'lofteePrediction') == 'HC')
        .then(pl.lit(' A high-confidence loss-of-function variant by loftee.'))
        .otherwise(pl.lit(''))
    )
    biotype = field(first, 'biotype')
    biotype_text = pl.concat_str(
        pl.lit(', a '),
        biotype.str.replace_all('_', ' '),
        pl.when(biotype.str.contains('gene', literal=True)).then(pl.lit('')).otherwise(pl.lit(' gene.')),
    )
    closest = (
        pl
        .when(protein.is_not_null())
        .then(
            pl.concat_str(
                pl.lit(' The closest protein-coding gene is '),
                label(protein),
                pl.lit(' ('),
                _format_distance(field(protein, 'distanceFromFootprint')),
                pl.lit(' basepair away).'),
            )
        )
        .otherwise(pl.lit(''))
    )
    return (
        pl
        .when(first.is_null())
        .then(pl.lit('Intergenic variant no gene in window'))
        .when(field(first, 'biotype') == 'protein_coding')
        .then(pl.concat_str(term, overlap, aa, lof))
        .otherwise(pl.concat_str(term, overlap, biotype_text, closest))
    )


def parse_vep(frame: pl.LazyFrame, hash_threshold: int = 300) -> pl.LazyFrame:
    """Parse VEP annotations into the complete variant dataset without materializing rows."""
    transcripts = pl.col('transcript_consequences')
    canonical = transcripts.list.eval(pl.element().filter(pl.element().struct.field('canonical') == 1))
    frame = frame.with_columns(canonical.alias('transcript_consequences'))
    fields = pl.col('input').str.split('\t')
    cv = pl.col('colocated_variants')
    rsids = cv.list.eval(pl.element().struct.field('id'))
    omim = (
        cv.list
        .eval(
            pl
            .element()
            .struct.field('var_synonyms')
            .struct.field('OMIM')
            .filter(pl.element().struct.field('var_synonyms').struct.field('OMIM').is_not_null())
        )
        .list.eval(pl.element().explode())
        .list.eval(pl.element().cast(pl.String).str.replace_all('.', '#', literal=True))
    )
    clinvar = (
        cv.list
        .eval(
            pl
            .element()
            .struct.field('var_synonyms')
            .struct.field('ClinVar')
            .filter(pl.element().struct.field('var_synonyms').struct.field('ClinVar').is_not_null())
        )
        .list.eval(pl.element().explode())
        .list.eval(pl.element().filter(pl.element().str.starts_with('VCV')))
    )
    alpha = transcripts.list.eval(
        pl.element().filter(pl.element().struct.field('alphamissense').is_not_null())
    ).list.first()
    alpha_effect = pl.when(alpha.is_not_null()).then(
        pl.struct(
            pl.lit('AlphaMissense').alias('method'),
            alpha.struct.field('alphamissense').struct.field('am_class').alias('assessment'),
            alpha.struct.field('alphamissense').struct.field('am_pathogenicity').cast(pl.Float32).alias('score'),
            pl.lit(None, dtype=pl.String).alias('assessmentFlag'),
            alpha.struct.field('gene_id').alias('targetId'),
            pl.lit(None, dtype=pl.Float64).alias('normalisedScore'),
        )
    )
    effects = pl.concat_list(
        variant_effect(most_severe_transcript(transcripts, 'cadd_phred'), 'CADD', 'cadd_phred'),
        variant_effect(
            most_severe_transcript(transcripts, 'polyphen_score'), 'PolyPhen', 'polyphen_score', 'polyphen_prediction'
        ),
        variant_effect(most_severe_transcript(transcripts, 'sift_score'), 'SIFT', 'sift_score', 'sift_prediction'),
        variant_effect(most_severe_transcript(transcripts, 'lof'), 'LOFTEE', 'lof', 'lof', 'lof_filter'),
        variant_effect(most_severe_transcript(transcripts, 'conservation'), 'GERP', 'conservation'),
        alpha_effect,
        vep_prediction(pl.col('most_severe_consequence')),
    ).list.eval(pl.element().filter(pl.element().is_not_null()))
    frame = frame.select(
        pl.concat_str(
            [fields.list.get(i, null_on_oob=True) for i in [0, 1, 3, 4]], separator='_', ignore_nulls=True
        ).alias('variantId'),
        fields.list.get(0, null_on_oob=True).alias('chromosome'),
        fields.list.get(1, null_on_oob=True).cast(pl.Int32, strict=False).alias('position'),
        fields.list.get(3, null_on_oob=True).alias('referenceAllele'),
        fields.list.get(4, null_on_oob=True).alias('alternateAllele'),
        effects.alias('variantEffect'),
        pl
        .col('most_severe_consequence')
        .replace_strict(SEQUENCE_ONTOLOGY, default=None, return_dtype=pl.String)
        .alias('mostSevereConsequenceId'),
        transcript_consequences(transcripts).alias('transcriptConsequences'),
        rsids.alias('rsIds'),
        pl
        .when(transcripts.list.len() > 0)
        .then(transcripts.list.first().struct.field('hgvsg'))
        .when(pl.col('intergenic_consequences').list.len() > 0)
        .then(pl.col('intergenic_consequences').list.first().struct.field('hgvsg'))
        .otherwise(None)
        .alias('hgvsId'),
        pl.lit([], dtype=variant_schema['alleleFrequencies']).alias('alleleFrequencies'),
        pl.concat_list(
            generate_dbxrefs(rsids, 'ensembl_variation'),
            generate_dbxrefs(omim, 'omim'),
            generate_dbxrefs(clinvar, 'clinvar'),
        ).alias('dbXrefs'),
        pl.col('most_severe_consequence'),
    ).filter(pl.col('position').is_not_null())
    missense = (
        pl
        .col('transcriptConsequences')
        .list.eval(pl.element().struct.field('variantFunctionalConsequenceIds').list.contains('SO_0001583'))
        .list.any()
    )
    frame = frame.with_columns(
        pl.concat_list(
            pl.col('dbXrefs'),
            pl
            .when(missense)
            .then(generate_dbxrefs(pl.concat_list('variantId'), 'protvar'))
            .otherwise(pl.lit([], dtype=variant_schema['dbXrefs'])),
        ).alias('dbXrefs'),
        variant_description(pl.col('most_severe_consequence'), pl.col('transcriptConsequences')).alias(
            'variantDescription'
        ),
    )
    return (
        frame
        .with_columns(
            hash_variant_index_ids(pl.col('variantId'), pl.col('chromosome'), pl.col('position'), hash_threshold).alias(
                'variantId'
            )
        )
        .pipe(normalize_variant_effects)
        .select([
            pl.col(name).fill_null([]).cast(dtype) if isinstance(dtype, pl.List) else pl.col(name).cast(dtype)
            for name, dtype in variant_schema.items()
        ])
    )
