"""Gene Ontology annotations, protein (GOA) and RNA (RNAcentral), by Ensembl gene id."""

from __future__ import annotations

import polars as pl

from pts.transformers.utils.dataset import scan_dataset

GO_FIELDS = ['id', 'source', 'evidence', 'aspect', 'geneProduct', 'ecoId']


def read_tsv(path: str, comment_prefix: str | None = None) -> pl.DataFrame:
    """A headerless TSV as text columns named as spark names them, `_c0`, `_c1`, ..."""
    frame = scan_dataset(
        path,
        format='tsv',
        has_header=False,
        comment_prefix=comment_prefix,
        quote_char=None,
        infer_schema=False,
    ).collect()
    return frame.rename({name: f'_c{i}' for i, name in enumerate(frame.columns)})


def _annotations(gaf: pl.DataFrame) -> pl.DataFrame:
    return gaf.select(
        pl.col('_c1').alias('dbObjectId'),
        pl.col('_c4').alias('goId'),
        pl.col('_c5').alias('source'),
        pl.col('_c6').alias('evidence'),
        pl.col('_c8').alias('aspect'),
        pl.col('_c1').alias('geneProduct'),
    )


def build_go(
    human: pl.DataFrame,
    rna: pl.DataFrame,
    rna_lookup: pl.DataFrame,
    eco_lookup: pl.DataFrame,
    ensembl: pl.DataFrame,
) -> pl.DataFrame:
    """GO annotations per Ensembl gene id.

    Args:
        human: `goa_human.gaf`, from `read_tsv(..., comment_prefix='!')`.
        rna: `goa_human_rna.gaf`, likewise.
        rna_lookup: RNAcentral `ensembl.tsv`, from `read_tsv`.
        eco_lookup: `goa_human_eco.gpa`, from `read_tsv(..., comment_prefix='!')`.
        ensembl: the ensembl builder's [id, approvedSymbol, proteinIds]; maps UniProt accessions
            and symbols to genes.

    Returns:
        [ensemblId, go], `go` a sorted list of distinct {id, source, evidence, aspect, geneProduct, ecoId}.
    """
    eco = eco_lookup.select(
        pl.col('_c3').alias('goId'),
        pl.col('_c1').alias('geneProduct'),
        pl.col('_c11').str.split('=').list.get(1, null_on_oob=True).alias('evidence'),
        pl.col('_c5').alias('ecoId'),
    ).unique(maintain_order=True)

    rna_genes = (
        rna_lookup
        .select(pl.col('_c0').alias('rnaCentralId'), pl.col('_c5').alias('ensemblId'))
        .filter(pl.col('ensemblId').str.starts_with('ENSG0'))
        .with_columns(pl.col('ensemblId').str.extract(r'(ENSG[0-9]+)', 1).fill_null(''))
    )
    protein_genes = (
        ensembl
        .select(
            pl.col('id').alias('ensemblId'),
            pl
            .concat_list(
                pl.col('proteinIds').list.eval(pl.element().struct.field('id')),
                pl.col('approvedSymbol'),
            )
            .list.unique(maintain_order=True)
            .alias('uniprotId'),
        )
        .explode('uniprotId', empty_as_null=False, keep_nulls=False)
        .drop_nulls('uniprotId')
    )

    rna_go = (
        _annotations(rna)
        .with_columns(pl.col('dbObjectId').str.split('_').list.first())
        .join(rna_genes, left_on='dbObjectId', right_on='rnaCentralId', how='inner')
    )
    human_go = _annotations(human).join(protein_genes, left_on='dbObjectId', right_on='uniprotId', how='inner')

    return (
        pl
        .concat([human_go.select(rna_go.columns), rna_go])
        .join(eco, on=['goId', 'geneProduct', 'evidence'], how='left')
        .select('ensemblId', pl.col('goId').alias('id'), *GO_FIELDS[1:])
        .unique()
        .sort(pl.all(), nulls_last=False)
        .group_by('ensemblId', maintain_order=True)
        .agg(pl.struct(*GO_FIELDS).alias('go'))
    )
