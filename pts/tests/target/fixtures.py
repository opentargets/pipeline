"""Small Ensembl frames shared by the target builder tests."""

from __future__ import annotations

import gzip
import io
import json
import zipfile
from pathlib import Path

import polars as pl

from tests.ensembl_fixtures import cells, release_116_ddl, write_dump

GENES_SCHEMA = {
    'id': pl.String,
    'biotype': pl.String,
    'description': pl.String,
    'chromosome': pl.String,
    'start': pl.Int64,
    'end': pl.Int64,
    'strand': pl.Int32,
    'approvedSymbol': pl.String,
    'canonicalTranscriptId': pl.String,
}
TRANSCRIPTS_SCHEMA = {
    'geneId': pl.String,
    'transcriptId': pl.String,
    'biotype': pl.String,
    'chromosome': pl.String,
    'start': pl.Int64,
    'end': pl.Int64,
    'strand': pl.Int32,
    'translationId': pl.String,
}
XREFS_SCHEMA = {
    'transcriptId': pl.String,
    'translationId': pl.String,
    'uniprot_swissprot': pl.List(pl.String),
    'uniprot_trembl': pl.List(pl.String),
    'uniprot_isoform': pl.List(pl.String),
    'alphafold': pl.List(pl.String),
    'signalp': pl.List(pl.String),
}

GENES = pl.DataFrame(
    [
        (
            'ENSG00000141510',
            'protein_coding',
            'tumor protein p53 [Source:HGNC Symbol;Acc:HGNC:11998]',
            '17',
            7661779,
            7687546,
            -1,
            'TP53',
            'ENST00000269305',
        ),
        ('ENSG00000000002', 'TEC', 'TEC protein detection [Source:x]', 'X', 1000, 2000, 1, None, 'ENST00000000021'),
        ('ENSG00000000003', 'Mt_tRNA', None, 'MT', 577, 647, 1, 'MT-TF', 'ENST00000000031'),
        ('LRG_1', 'protein_coding', 'lrg', '1', 10, 20, 1, 'LRG', 'LRG_1t1'),
    ],
    schema=GENES_SCHEMA,
    orient='row',
)
TRANSCRIPTS = pl.DataFrame(
    [
        ('ENSG00000000002', 'ENST00000000021', 'TEC', 'X', 1000, 2000, 1, None),
        ('ENSG00000000003', 'ENST00000000031', 'Mt_tRNA', 'MT', 577, 647, 1, None),
        ('ENSG00000141510', 'ENST00000269305', 'protein_coding', '17', 7661779, 7687538, -1, 'ENSP00000269305'),
        ('ENSG00000141510', 'ENST00000413465', 'protein_coding', '17', 7661779, 7687500, -1, 'ENSP00000410739'),
        ('LRG_1', 'LRG_1t1', 'protein_coding', '1', 10, 20, 1, 'LRG_1p1'),
    ],
    schema=TRANSCRIPTS_SCHEMA,
    orient='row',
)
XREFS = pl.DataFrame(
    [
        ('ENST00000269305', 'ENSP00000269305', ['P04637'], ['K7PPA8'], [], [], []),
        ('ENST00000413465', 'ENSP00000410739', [], ['H2EHT1', 'K7PPA8'], [], [], ['SignalP-noTM']),
        ('LRG_1t1', 'LRG_1p1', ['P00001'], [], [], [], []),
    ],
    schema=XREFS_SCHEMA,
    orient='row',
)


N = None

#: TP53 (reverse strand, one coding transcript) and a TEC gene on X; the scaffold gene is dropped.
CORE_ROWS = {
    'analysis': [
        (1, N, 'ensembl', N, N, N, N, N, N, N, N, N, N, N),
        (2, N, 'signalp', N, N, N, N, N, N, N, N, N, N, N),
        (3, N, 'alphafold', N, N, N, N, N, N, N, N, N, N, N),
    ],
    'attrib_type': [(1, 'is_canonical', 'Canonical', N)],
    'coord_system': [(1, 1, 'chromosome', 'GRCh38', 1, 'default_version'), (2, 1, 'scaffold', 'GRCh38', 2, N)],
    'exon': [
        (1001, 17, 7661779, 7687538, -1, 0, 0, 1, 0, 'ENSE00001', 1, N, N),
        (2001, 23, 1000, 2000, 1, -1, -1, 1, 0, 'ENSE00002', 1, N, N),
        (3001, 99, 1, 50, 1, -1, -1, 1, 0, 'ENSE00003', 1, N, N),
    ],
    'exon_transcript': [(1001, 11, 1), (2001, 21, 1), (3001, 31, 1)],
    'external_db': [
        (1, 'HGNC', N, 'KNOWNXREF', 1, 'HGNC Symbol', 'MISC', N, N, N),
        (2, 'Uniprot/SWISSPROT', N, 'KNOWN', 1, 'UniProtKB/Swiss-Prot', 'MISC', N, N, N),
        (3, 'Uniprot/SPTREMBL', N, 'KNOWN', 1, 'UniProtKB/TrEMBL', 'MISC', N, N, N),
    ],
    'gene': [
        (
            1,
            'protein_coding',
            1,
            17,
            7661779,
            7687546,
            -1,
            10,
            'ensembl_havana',
            'tumor protein p53 [Source:HGNC Symbol;Acc:HGNC:11998]',
            1,
            11,
            'ENSG00000141510',
            19,
            N,
            N,
        ),
        (2, 'TEC', 1, 23, 1000, 2000, 1, N, 'havana', N, 1, 21, 'ENSG00000000002', 1, N, N),
        (3, 'lncRNA', 1, 99, 1, 50, 1, N, 'havana', N, 1, 31, 'ENSG00000000099', 1, N, N),
    ],
    'object_xref': [(1, 1, 'Gene', 10, N, N), (2, 111, 'Translation', 20, N, N), (3, 111, 'Translation', 30, N, N)],
    'protein_feature': [
        (1, 111, 1, 22, 1, 22, 'SignalP-noTM', 2, N, N, N, N, N, N, N),
        (2, 111, 1, 393, 1, 393, 'AF-P04637-F1', 3, N, N, N, N, N, N, N),
    ],
    'seq_region': [(17, '17', 1, 83257441), (23, 'X', 1, 156040895), (99, 'KI270728.1', 2, 1872759)],
    'transcript': [
        (
            11,
            1,
            1,
            17,
            7661779,
            7687538,
            -1,
            N,
            'ensembl_havana',
            'protein_coding',
            N,
            1,
            111,
            'ENST00000269305',
            9,
            N,
            N,
        ),
        (21, 2, 1, 23, 1000, 2000, 1, N, 'havana', 'TEC', N, 1, N, 'ENST00000000021', 1, N, N),
        (31, 3, 1, 99, 1, 50, 1, N, 'havana', 'lncRNA', N, 1, N, 'ENST00000000031', 1, N, N),
    ],
    'transcript_attrib': [(11, 1, '1'), (21, 1, '1'), (31, 1, '1')],
    'translation': [(111, 11, 1, 1001, 100, 1001, 'ENSP00000269305', 4, N, N)],
    'xref': [
        (10, 1, 'HGNC:11998', 'TP53', '0', N, 'DIRECT', ''),
        (20, 2, 'P04637', 'P53_HUMAN', '1', N, 'DIRECT', ''),
        (30, 3, 'K7PPA8', 'K7PPA8_HUMAN', '1', N, 'SEQUENCE_MATCH', ''),
    ],
}


def write_core_dump(path: Path) -> str:
    """Write CORE_ROWS as a human core dump."""
    path.mkdir(parents=True, exist_ok=True)
    tables = {table: [cells(*row) for row in rows] for table, rows in CORE_ROWS.items()}
    return write_dump(path, tables, release_116_ddl(CORE_ROWS))


def _gz(path: Path, text: str) -> str:
    path.write_bytes(gzip.compress(text.encode()))
    return str(path)


def write_inputs(path: Path) -> dict[str, str]:
    """Every non-dump input of the target step, as small files."""
    path.mkdir(parents=True, exist_ok=True)
    hgnc = path / 'hgnc_complete_set.json'
    hgnc.write_text(
        json.dumps({
            'response': {
                'docs': [
                    {
                        'ensembl_gene_id': 'ENSG00000141510',
                        'hgnc_id': 'HGNC:11998',
                        'symbol': 'TP53',
                        'name': 'tumor protein p53',
                        'uniprot_ids': ['P04637'],
                        'alias_symbol': ['p53'],
                        'prev_name': ['Li-Fraumeni syndrome'],
                    },
                ]
            }
        })
    )
    gaf = '\t'.join([
        'UniProtKB',
        'P04637',
        'TP53',
        'enables',
        'GO:0003677',
        'PMID:1',
        'IDA',
        '',
        'F',
        'p53',
        '',
        'protein',
        'taxon:9606',
        '20230101',
        'UniProt',
        '',
        '',
    ])
    rna_gaf = '\t'.join([
        'RNAcentral',
        'URS1_9606',
        'URS1_9606',
        'enables',
        'GO:0030533',
        'GO_REF:0000115',
        'IEA',
        '',
        'F',
        'tRNA',
        '',
        'tRNA',
        'taxon:9606',
        '20230101',
        'RNAcentral',
        '',
        '',
    ])
    gpa = '\t'.join([
        'UniProtKB',
        'P04637',
        'enables',
        'GO:0003677',
        'PMID:1',
        'ECO:0000314',
        '',
        '',
        '20230101',
        'UniProt',
        '',
        'go_evidence=IDA',
    ])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr(
            'subcellular_location.tsv',
            'Gene\tGene name\tMain location\tAdditional location\tExtracellular location\n'
            'ENSG00000141510\tTP53\tNucleoplasm\t\t\n',
        )
    (path / 'subcellular_location.tsv.zip').write_bytes(buffer.getvalue())
    (path / 'subcellular_locations_ssl.tsv').write_text(
        'HPA_location\ttermSL\tlabelSL\nNucleoplasm\tSL-0190\tNucleus\n'
    )
    gnomad_columns = (
        ['gene_id', 'canonical', 'transcript_type', 'lof.oe_ci.upper_rank']
        + [f'{k}.{m}' for k in ('syn', 'mis') for m in ('z_score', 'exp', 'obs', 'oe', 'oe_ci.lower', 'oe_ci.upper')]
        + ['lof.pLI', 'lof.exp', 'lof.obs', 'lof.oe', 'lof.oe_ci.lower', 'lof.oe_ci.upper']
    )
    gnomad_row = ['ENSG00000141510', 'true', 'protein_coding', '1'] + ['1'] * (len(gnomad_columns) - 4)
    pl.DataFrame({
        'id': ['R-HSA-1', 'R-HSA-69278'],
        'label': ['Cell Cycle', 'Cell Cycle, Mitotic'],
        'path': [[['R-HSA-1']], [['R-HSA-1', 'R-HSA-69278']]],
    }).write_parquet(path / 'reactome.parquet')
    uniprot_entry = (
        'ID   P53_HUMAN               Reviewed;         393 AA.\nAC   P04637; Q15086;\n'
        'DE   RecName: Full=Cellular tumor antigen p53;\nGN   Name=TP53; Synonyms=P53;\n'
        'DR   PDB; 1A1U; X-ray; -.\nCC   -!- FUNCTION: Acts as a tumor suppressor.\n'
        'CC   -!- SUBCELLULAR LOCATION: Nucleus.\n//\n'
    )
    return {
        'hgnc': str(hgnc),
        'gene_ontology_human': _gz(path / 'goa_human.gaf.gz', f'!gaf-version: 2.2\n{gaf}\n'),
        'gene_ontology_rna': _gz(path / 'goa_human_rna.gaf.gz', f'!gaf-version: 2.2\n{rna_gaf}\n'),
        'gene_ontology_rna_lookup': str(
            _write(path / 'ensembl.tsv', 'URS1\tENSEMBL\tENST1\t9606\tlncRNA\tENSG00000000002.1\n')
        ),
        'gene_ontology_eco_lookup': _gz(path / 'goa_human_eco.gpa.gz', f'!gpa-version: 1.1\n{gpa}\n'),
        'hallmarks': _gz(
            path / 'cosmic-hallmarks.tsv.gz',
            'GENE_SYMBOL\tCELL_TYPE\tPUBMED_PMID\tHALLMARK\tIMPACT\tDESCRIPTION\tCELL_LINE\r\n'
            'TP53\t\t1\tangiogenesis\tpromotes\tx\t\r\n',
        ),
        'ncbi': _gz(
            path / 'Homo_sapiens.gene_info.gz',
            '#tax_id\tGeneID\tSymbol\tSynonyms\tdbXrefs\tOther_designations\n'
            '9606\t7157\tTP53\tBCC7|P53\tEnsembl:ENSG00000141510\tcellular tumor antigen p53\n',
        ),
        'hpa': str(path / 'subcellular_location.tsv.zip'),
        'hpa_sl': str(path / 'subcellular_locations_ssl.tsv'),
        'gnomad': _gz(
            path / 'gnomad_constraint_metrics.tsv.bgz', '\t'.join(gnomad_columns) + '\n' + '\t'.join(gnomad_row) + '\n'
        ),
        'reactome_pathways': str(
            _write(
                path / 'Ensembl2Reactome.txt',
                'ENSG00000141510\tR-HSA-69278\turl\tCell Cycle, Mitotic\tTAS\tHomo sapiens\n',
            )
        ),
        'reactome_etl': str(path / 'reactome.parquet'),
        'uniprot': _gz(path / 'uniprot.txt.gz', uniprot_entry),
        'uniprot_ssl': _gz(
            path / 'uniprot-ssl.tsv.gz',
            'Subcellular location ID\tName\tCategory\nSL-0191\tNucleus\tCellular component\n',
        ),
    }


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path

