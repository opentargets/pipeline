"""Write small Ensembl core dumps for tests."""

from __future__ import annotations

import gzip
from collections.abc import Iterable
from pathlib import Path

NULL = '\\N'
"""How the dump spells NULL."""


def write_gz(path: Path, text: str) -> None:
    """Write `text` to `path` gzipped, as the FTP site ships it."""
    with gzip.open(path, 'wt', encoding='utf8') as handle:
        handle.write(text)


def ddl(tables: dict[str, list[str]]) -> str:
    """`CREATE TABLE` statements for `tables` (name to columns in order), with an index line each."""
    statements = []
    for name, columns in tables.items():
        lines = [f'  `{column}` text,' for column in columns]
        body = '\n'.join([*lines, f'  PRIMARY KEY (`{columns[0]}`)'])
        statements.append(f'CREATE TABLE `{name}` (\n{body}\n) ENGINE=MyISAM DEFAULT CHARSET=latin1;\n')
    return ''.join(statements)


def cells(*values: object) -> list[str]:
    """One dump row: `None` becomes the null token, everything else `str()`."""
    return [NULL if value is None else str(value) for value in values]


def write_dump(
    tmp_path: Path,
    tables: dict[str, list[list[str]]],
    ddl: str,
    *,
    db: str = 'homo_sapiens_core_116_38',
) -> str:
    """Write an Ensembl core dump into `tmp_path`.

    Args:
        tmp_path: Directory to write into.
        tables: Table name to its rows, each row the already-escaped field strings in DDL order.
        ddl: Contents of the schema file.
        db: Database name, which names the schema file `<db>.sql.gz`.

    Returns:
        `tmp_path` as a string, ready for `CoreDump`.
    """
    write_gz(tmp_path / f'{db}.sql.gz', ddl)
    for name, rows in tables.items():
        write_gz(tmp_path / f'{name}.txt.gz', ''.join('\t'.join(row) + '\n' for row in rows))
    return str(tmp_path)


#: Release-116 column order of the tables the target and homology queries read.
RELEASE_116_COLUMNS: dict[str, list[str]] = {
    'analysis': [
        'analysis_id', 'created', 'logic_name', 'db', 'db_version', 'db_file', 'program', 'program_version',
        'program_file', 'parameters', 'module', 'module_version', 'gff_source', 'gff_feature',
    ],
    'attrib_type': ['attrib_type_id', 'code', 'name', 'description'],
    'coord_system': ['coord_system_id', 'species_id', 'name', 'version', 'rank', 'attrib'],
    'exon': [
        'exon_id', 'seq_region_id', 'seq_region_start', 'seq_region_end', 'seq_region_strand', 'phase',
        'end_phase', 'is_current', 'is_constitutive', 'stable_id', 'version', 'created_date', 'modified_date',
    ],
    'exon_transcript': ['exon_id', 'transcript_id', 'rank'],
    'external_db': [
        'external_db_id', 'db_name', 'db_release', 'status', 'priority', 'db_display_name', 'type',
        'secondary_db_name', 'secondary_db_table', 'description',
    ],
    'gene': [
        'gene_id', 'biotype', 'analysis_id', 'seq_region_id', 'seq_region_start', 'seq_region_end',
        'seq_region_strand', 'display_xref_id', 'source', 'description', 'is_current', 'canonical_transcript_id',
        'stable_id', 'version', 'created_date', 'modified_date',
    ],
    'object_xref': [
        'object_xref_id', 'ensembl_id', 'ensembl_object_type', 'xref_id', 'linkage_annotation', 'analysis_id',
    ],
    'protein_feature': [
        'protein_feature_id', 'translation_id', 'seq_start', 'seq_end', 'hit_start', 'hit_end', 'hit_name',
        'analysis_id', 'score', 'evalue', 'perc_ident', 'external_data', 'hit_description', 'cigar_line',
        'align_type',
    ],
    'seq_region': ['seq_region_id', 'name', 'coord_system_id', 'length'],
    'transcript': [
        'transcript_id', 'gene_id', 'analysis_id', 'seq_region_id', 'seq_region_start', 'seq_region_end',
        'seq_region_strand', 'display_xref_id', 'source', 'biotype', 'description', 'is_current',
        'canonical_translation_id', 'stable_id', 'version', 'created_date', 'modified_date',
    ],
    'transcript_attrib': ['transcript_id', 'attrib_type_id', 'value'],
    'translation': [
        'translation_id', 'transcript_id', 'seq_start', 'start_exon_id', 'seq_end', 'end_exon_id', 'stable_id',
        'version', 'created_date', 'modified_date',
    ],
    'xref': [
        'xref_id', 'external_db_id', 'dbprimary_acc', 'display_label', 'version', 'description', 'info_type',
        'info_text',
    ],
}  # fmt: skip


def release_116_ddl(tables: Iterable[str]) -> str:
    """`CREATE TABLE` statements for `tables`, with release 116's columns."""
    return ddl({table: RELEASE_116_COLUMNS[table] for table in tables})
