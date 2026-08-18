"""Build the gene frame the target step reads, from the Ensembl core database dump.

Only the fields downstream actually consumes are emitted. Anything the JSON dump carried that
no consumer reads -- gene-level alphafold and uniprot isoform, most transcript-level fields,
exon ids and strands -- is left out rather than carried for symmetry.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.ensembl_core import CoreDump
from pts.transformers.utils.dataset import write_dataset

#: Only canonical chromosomes reach the release. Everything on a scaffold, a patch or an LRG
#: region is dropped, which is what the JSON route's filter did by name too.
INCLUDED_CHROMOSOMES = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']

#: The coordinate system genes must sit on to be included.
PRIMARY_COORD_SYSTEM = 'chromosome'


def _genes(dump: CoreDump) -> pl.LazyFrame:
    """Gene-level scalar fields, filtered to the canonical chromosomes."""
    gene = dump.scan('gene', {
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
    })
    seq_region = dump.scan('seq_region', {
        'seq_region_id': pl.Int64(),
        'name': pl.String(),
        'coord_system_id': pl.Int64(),
    })
    coord_system = dump.scan('coord_system', {'coord_system_id': pl.Int64(), 'name': pl.String()})
    xref = dump.scan('xref', {'xref_id': pl.Int64(), 'display_label': pl.String()})

    return (
        gene
        .join(seq_region, on='seq_region_id', how='left')
        .join(
            coord_system.filter(pl.col('name') == PRIMARY_COORD_SYSTEM).select('coord_system_id'),
            on='coord_system_id',
            how='inner',
        )
        .filter(pl.col('name').is_in(INCLUDED_CHROMOSOMES))
        .join(xref, left_on='display_xref_id', right_on='xref_id', how='left')
        .select(
            pl.col('stable_id').alias('id'),
            'biotype',
            'description',
            pl.col('name').alias('chromosome'),
            pl.col('seq_region_start').alias('start'),
            pl.col('seq_region_end').alias('end'),
            pl.col('seq_region_strand').alias('strand'),
            pl.col('display_label').alias('approvedSymbol'),
            'gene_id',
            'canonical_transcript_id',
        )
    )


def _gene_dictionary(dump: CoreDump) -> pl.LazyFrame:
    """Every gene's stable id and symbol, for the homology lookup.

    Deliberately unfiltered: `_build_homologues` resolves human paralog symbols through this,
    and paralogs land on scaffolds that the release's own gene set excludes.
    """
    gene = dump.scan('gene', {'stable_id': pl.String(), 'display_xref_id': pl.Int64()})
    xref = dump.scan('xref', {'xref_id': pl.Int64(), 'display_label': pl.String()})
    return (
        gene
        .join(xref, left_on='display_xref_id', right_on='xref_id', how='left')
        .select(pl.col('stable_id').alias('id'), pl.col('display_label').alias('name'))
        .sort('id')
    )


def ensembl(
    source: str,
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Transform the Ensembl core dump into the gene frame and the homology gene dictionary.

    Args:
        source: Directory holding the core dump's `.txt.gz` tables and its `.sql.gz` schema.
        destination: Dictionary with paths to:
            - genes: the gene frame the target step reads.
            - gene_dictionary: gene id to symbol, for the homology lookup.
        settings: Unused.
        config: Config object, unused.
    """
    dump = CoreDump(source)
    logger.info(f'transforming ensembl release {dump.release} from {source}')

    genes = _genes(dump).drop('gene_id', 'canonical_transcript_id').collect()
    logger.info(f'built {genes.height} genes')
    write_dataset(genes, destination['genes'])

    dictionary = _gene_dictionary(dump).collect()
    logger.info(f'built {dictionary.height} gene dictionary entries')
    write_dataset(dictionary, destination['gene_dictionary'])
    logger.info('transformation complete')
