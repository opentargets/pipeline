"""Schemas for the Ensembl core database dump: the columns read per table and the query outputs."""

import polars as pl

#: Columns read from each core table, by name, with the dtype to read them as.
GENE = {
    'gene_id': pl.Int64(),
    'biotype': pl.String(),
    'seq_region_id': pl.Int64(),
    'seq_region_start': pl.Int64(),
    'seq_region_end': pl.Int64(),
    'seq_region_strand': pl.Int32(),
    'display_xref_id': pl.Int64(),
    'description': pl.String(),
    'canonical_transcript_id': pl.Int64(),
    'stable_id': pl.String(),
}
TRANSCRIPT = {
    'transcript_id': pl.Int64(),
    'gene_id': pl.Int64(),
    'seq_region_id': pl.Int64(),
    'seq_region_start': pl.Int64(),
    'seq_region_end': pl.Int64(),
    'seq_region_strand': pl.Int32(),
    'biotype': pl.String(),
    'canonical_translation_id': pl.Int64(),
    'stable_id': pl.String(),
}
TRANSLATION = {'translation_id': pl.Int64(), 'transcript_id': pl.Int64(), 'stable_id': pl.String()}
EXON = {
    'exon_id': pl.Int64(),
    'seq_region_id': pl.Int64(),
    'seq_region_start': pl.Int64(),
    'seq_region_end': pl.Int64(),
    'seq_region_strand': pl.Int32(),
    'stable_id': pl.String(),
}
EXON_TRANSCRIPT = {'exon_id': pl.Int64(), 'transcript_id': pl.Int64(), 'rank': pl.Int64()}
TRANSCRIPT_ATTRIB = {'transcript_id': pl.Int64(), 'attrib_type_id': pl.Int64(), 'value': pl.String()}
ATTRIB_TYPE = {'attrib_type_id': pl.Int64(), 'code': pl.String()}
SEQ_REGION = {'seq_region_id': pl.Int64(), 'name': pl.String(), 'coord_system_id': pl.Int64()}
COORD_SYSTEM = {'coord_system_id': pl.Int64(), 'name': pl.String()}
XREF = {'xref_id': pl.Int64(), 'external_db_id': pl.Int64(), 'dbprimary_acc': pl.String(), 'display_label': pl.String()}
OBJECT_XREF = {'ensembl_id': pl.Int64(), 'ensembl_object_type': pl.String(), 'xref_id': pl.Int64()}
EXTERNAL_DB = {'external_db_id': pl.Int64(), 'db_name': pl.String()}
PROTEIN_FEATURE = {'translation_id': pl.Int64(), 'hit_name': pl.String(), 'analysis_id': pl.Int64()}
ANALYSIS = {'analysis_id': pl.Int64(), 'logic_name': pl.String()}

attribute = pl.Struct({'code': pl.String(), 'value': pl.String()})
"""One `transcript_attrib` row: the `attrib_type.code` and its value."""

exon = pl.Struct({
    'exonId': pl.String(),
    'chromosome': pl.String(),
    'start': pl.Int64(),
    'end': pl.Int64(),
    'strand': pl.Int32(),
})
"""One exon of a transcript."""

gene_names = pl.Schema({'id': pl.String(), 'name': pl.String()})
"""Output of `pts.transformers.utils.ensembl.queries.gene_names`."""

genes = pl.Schema({
    'id': pl.String(),
    'biotype': pl.String(),
    'description': pl.String(),
    'chromosome': pl.String(),
    'start': pl.Int64(),
    'end': pl.Int64(),
    'strand': pl.Int32(),
    'approvedSymbol': pl.String(),
    'canonicalTranscriptId': pl.String(),
})
"""Output of `pts.transformers.utils.ensembl.queries.genes`."""

transcripts = pl.Schema({
    'geneId': pl.String(),
    'transcriptId': pl.String(),
    'biotype': pl.String(),
    'chromosome': pl.String(),
    'start': pl.Int64(),
    'end': pl.Int64(),
    'strand': pl.Int32(),
    'translationId': pl.String(),
    'attributes': pl.List(attribute),
    'exons': pl.List(exon),
})
"""Output of `pts.transformers.utils.ensembl.queries.transcripts`."""

translation_xrefs = pl.Schema({
    'transcriptId': pl.String(),
    'translationId': pl.String(),
    'uniprot_swissprot': pl.List(pl.String()),
    'uniprot_trembl': pl.List(pl.String()),
    'uniprot_isoform': pl.List(pl.String()),
    'alphafold': pl.List(pl.String()),
    'signalp': pl.List(pl.String()),
})
"""Output of `pts.transformers.utils.ensembl.queries.translation_xrefs`."""
