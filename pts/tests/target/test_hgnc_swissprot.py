"""Promoting HGNC's reviewed UniProt accession to Swiss-Prot where Ensembl gives none (opentargets/issues#4604)."""

from __future__ import annotations

import polars as pl

from pts.schemas.target import ID_SOURCE, UNIPROT_ENTRY
from pts.transformers.target.assemble import filter_and_sort_protein_ids, promote_hgnc_swissprot


def _ids(*pairs: tuple[str, str]) -> list[dict]:
    return [{'id': i, 'source': s} for i, s in pairs]


def _ensembl(*genes: tuple[str, str, list[dict] | None]) -> pl.DataFrame:
    return pl.DataFrame(
        genes, schema={'id': pl.String, 'biotype': pl.String, 'proteinIds': pl.List(ID_SOURCE)}, orient='row'
    )


def _hgnc(*genes: tuple[str, list[str]]) -> pl.DataFrame:
    return pl.DataFrame(genes, schema={'ensemblId': pl.String, 'uniprotIds': pl.List(pl.String)}, orient='row')


def _entries(*entries: tuple[bool, list[str]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {'id': f'E{n}', 'reviewed': reviewed, 'accessions': accessions}
            for n, (reviewed, accessions) in enumerate(entries)
        ],
        schema=UNIPROT_ENTRY,
    )


NLRP3 = 'ENSG00000162711'
ENSEMBL = _ensembl(
    (NLRP3, 'protein_coding', _ids(('A0A2R8YEG7', 'uniprot_trembl'), ('ENSP00000337383', 'ensembl_PRO'))),
    ('G_HAS', 'protein_coding', _ids(('P00001', 'uniprot_swissprot'))),
    ('G_NONCODING', 'lncRNA', []),
    ('G_HELD', 'protein_coding', []),
    ('G_SECONDARY', 'protein_coding', []),
    ('G_UNREVIEWED', 'protein_coding', []),
    ('G_SHARED_A', 'protein_coding', []),
    ('G_SHARED_B', 'protein_coding', []),
    ('G_NULL', 'protein_coding', None),
    ('G_NOHGNC', 'protein_coding', []),
)
HGNC = _hgnc(
    (NLRP3, ['Q96P20']),
    ('G_HAS', ['P00002']),
    ('G_NONCODING', ['P00003']),
    ('G_HELD', ['P00001']),
    ('G_SECONDARY', ['S00004']),
    ('G_UNREVIEWED', ['U00005']),
    ('G_SHARED_A', ['P00006']),
    ('G_SHARED_B', ['P00006']),
    ('G_NULL', ['P00007']),
)
ENTRIES = _entries(
    (True, ['Q96P20', 'A0A024R5Q0']),
    (True, ['P00001']),
    (True, ['P00002']),
    (True, ['P00003']),
    (True, ['P00004', 'S00004']),
    (False, ['U00005']),
    (True, ['P00006']),
    (True, ['P00007']),
)


def _promoted() -> dict[str, list[dict] | None]:
    result = promote_hgnc_swissprot(ENSEMBL, HGNC, ENTRIES)
    return dict(zip(result['id'].to_list(), result['proteinIds'].to_list(), strict=True))


def test_a_protein_coding_gene_without_ensembl_swissprot_gets_hgncs_reviewed_accession() -> None:
    assert _promoted()[NLRP3] == _ids(
        ('A0A2R8YEG7', 'uniprot_trembl'), ('ENSP00000337383', 'ensembl_PRO'), ('Q96P20', 'uniprot_swissprot')
    )


def test_a_gene_with_null_protein_ids_is_promoted_too() -> None:
    assert _promoted()['G_NULL'] == _ids(('P00007', 'uniprot_swissprot'))


def test_every_other_gene_is_returned_unchanged() -> None:
    result = promote_hgnc_swissprot(ENSEMBL, HGNC, ENTRIES)
    promoted = {NLRP3, 'G_NULL'}

    assert result.schema == ENSEMBL.schema
    assert result['id'].to_list() == ENSEMBL['id'].to_list()
    assert result.filter(~pl.col('id').is_in(promoted)).equals(ENSEMBL.filter(~pl.col('id').is_in(promoted)))


def test_ensembls_swissprot_is_kept_and_not_supplemented() -> None:
    assert _promoted()['G_HAS'] == _ids(('P00001', 'uniprot_swissprot'))


def test_non_coding_genes_are_not_promoted() -> None:
    assert _promoted()['G_NONCODING'] == []


def test_an_accession_ensembl_gives_another_gene_is_not_promoted() -> None:
    assert _promoted()['G_HELD'] == []


def test_only_the_primary_accession_of_an_entry_is_promoted() -> None:
    assert _promoted()['G_SECONDARY'] == []


def test_unreviewed_entries_are_not_promoted() -> None:
    assert _promoted()['G_UNREVIEWED'] == []


def test_an_accession_hgnc_maps_to_two_genes_is_promoted_to_neither() -> None:
    promoted = _promoted()
    assert promoted['G_SHARED_A'] == promoted['G_SHARED_B'] == []


def test_the_promoted_label_wins_over_the_obsolete_one_the_hgnc_path_adds() -> None:
    """`assemble` unions the Ensembl ids with `uniprot_by_gene`'s, which label the same accession obsolete."""
    gene = promote_hgnc_swissprot(ENSEMBL, HGNC, ENTRIES).filter(pl.col('id') == NLRP3)
    merged = gene.with_columns(
        pl.concat_list('proteinIds', pl.lit(_ids(('Q96P20', 'uniprot_obsolete')), dtype=pl.List(ID_SOURCE)))
    )

    assert filter_and_sort_protein_ids(merged)['proteinIds'].item().to_list() == _ids(
        ('Q96P20', 'uniprot_swissprot'), ('A0A2R8YEG7', 'uniprot_trembl'), ('ENSP00000337383', 'ensembl_PRO')
    )
