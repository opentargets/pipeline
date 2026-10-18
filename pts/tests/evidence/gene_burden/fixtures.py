"""Input writers for the gene burden parser and step tests."""

from __future__ import annotations

import gzip
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from pts.schemas.gene_burden import CURATED_INPUT_COLUMNS
from tests.evidence.gene_burden.workbook import write_rows, write_workbook

BINARY_COLUMNS = [
    'Gene', 'Phenotype', 'CollapsingModel', 'Type', 'pValue', 'Category', 'nSamples', 'BinNcases', 'BinQVcases',
    'BinNcontrols', 'BinQVcontrols', 'BinCaseFreq', 'BinCtrlFreq', 'BinOddsRatio', 'BinOddsRatioLCI',
    'BinOddsRatioUCI',
]  # fmt: skip


QUANTITATIVE_COLUMNS = [
    'Gene', 'Phenotype', 'CollapsingModel', 'Type', 'pValue', 'Category', 'nSamples', 'YesQV', 'NoQV',
    'ConCaseMedian', 'ConCtrMedian', 'beta', 'ConBetaSe', 'LCI', 'UCI',
]  # fmt: skip


GENE_LINK = 'https://azphewas.com/geneView/{}/GENE/glr/binary'


def _write_parquet(path: Path, rows: list[dict[str, str | None]], columns: list[str]) -> str:
    path.mkdir(parents=True)
    pl.DataFrame(rows, schema=dict.fromkeys(columns, pl.String)).write_parquet(path / 'part-00000.parquet')
    return str(path)


def _write_lines(path: Path, lines: list[str]) -> str:
    path.write_text(''.join(f'{line}\n' for line in lines))
    return str(path)


def write_az(
    tmp: Path,
    binary: list[dict[str, str | None]],
    quantitative: list[dict[str, str | None]],
    genes: list[str] | None = None,
    phenotypes: list[str] | None = None,
) -> tuple[str, str, str, str]:
    genes = [f'ENSG1,{GENE_LINK.format("release-1")}'] if genes is None else genes
    phenotypes = (
        ['Disease A,https://azphewas.com/phenotypeView/a', 'Trait B,https://azphewas.com/phenotypeView/b']
        if (phenotypes is None)
        else phenotypes
    )
    return (
        _write_parquet(tmp / 'binary', binary, BINARY_COLUMNS),
        _write_parquet(tmp / 'quantitative', quantitative, QUANTITATIVE_COLUMNS),
        _write_lines(tmp / 'genes.csv', genes),
        _write_lines(tmp / 'phenotypes.csv', phenotypes),
    )


HEADER = '\t'.join(CURATED_INPUT_COLUMNS) + '\n'


def curated_line(**fields: str) -> str:
    row = {
        'projectId': 'REGENERON',
        'targetFromSource': 'GENE1',
        'targetFromSourceId': 'ENSG1',
        'diseaseFromSource': 'Disease A',
        'diseaseFromSourceMappedId': 'EFO_0000001',
        'resourceScore': '1.5e-9',
        'pValueMantissa': '1.5',
        'pValueExponent': '-9',
        'cohortId': 'UK Biobank',
        'studySampleSize': '1000',
        'statisticalMethod': 'M1',
        'literature': '12345',
    }
    row.update(fields)
    unknown = set(row) - set(CURATED_INPUT_COLUMNS)
    assert not unknown, unknown
    return '\t'.join(row.get(c) or '' for c in CURATED_INPUT_COLUMNS) + '\n'


def write_curated(path: Path, *lines: str, header: str = HEADER) -> str:
    path.write_text(header + ''.join(lines))
    return str(path)


GENEBASS_SCHEMA = {
    'gene_id': pl.String,
    'gene_symbol': pl.String,
    'annotation': pl.String,
    'n_cases': pl.Int32,
    'n_controls': pl.Int32,
    'trait_type': pl.String,
    'phenocode': pl.String,
    'description': pl.String,
    'category': pl.String,
    'Pvalue': pl.Float64,
    'Pvalue_Burden': pl.Float64,
    'BETA_Burden': pl.Float64,
    'SE_Burden': pl.Float64,
}


def genebass_row(**fields: Any) -> dict[str, Any]:
    row = {
        'gene_id': 'ENSG1',
        'gene_symbol': 'G1',
        'annotation': 'pLoF',
        'n_cases': 1000,
        'n_controls': 9000,
        'trait_type': 'continuous',
        'phenocode': '30000',
        'description': 'White blood cell count',
        'category': 'Blood',
        'Pvalue': 0.5,
        'Pvalue_Burden': 2e-9,
        'BETA_Burden': 0.25,
        'SE_Burden': 0.125,
    }
    row.update(fields)
    return row


def write_genebass(tmp: Path, *files: list[dict[str, Any]]) -> str:
    path = tmp / 'genebass'
    path.mkdir()
    for i, rows in enumerate(files):
        pl.DataFrame(rows, schema=GENEBASS_SCHEMA).write_parquet(path / f'part-{i:05d}.parquet')
    return str(path)


FINNGEN_COLUMNS = ['PHENO', 'CHROM', 'GENPOS', 'ID', 'ALLELE0', 'ALLELE1', 'A1FREQ', 'N', 'TEST', 'BETA', 'SE', 'CHISQ']
FINNGEN_COLUMNS += ['LOG10P', 'EXTRA']


MANIFEST = [
    {'phenocode': 'T2D', 'phenostring': 'Type 2 diabetes', 'num_cases': 1000, 'hasRisteys': None},
    {'phenocode': 'ASTHMA', 'phenostring': 'Asthma', 'num_cases': 2000, 'hasRisteys': True},
]


def write_finngen(
    tmp: Path, lines: list[dict[str, str]], manifest: list[dict[str, Any]] | None = None
) -> tuple[str, str]:
    text = (
        '\t'.join(FINNGEN_COLUMNS) + '\n' + ''.join('\t'.join(row[c] for c in FINNGEN_COLUMNS) + '\n' for row in lines)
    )
    results = tmp / 'finngen.txt.gz'
    results.write_bytes(gzip.compress(text.encode()))
    phenos = tmp / 'phenos'
    phenos.write_text(json.dumps(MANIFEST if manifest is None else manifest))
    return str(results), str(phenos)


CVDI_INDEX = ['phenotype', 'Gene ID Ensembl', 'Gene']


CVDI_FIELDS = ['cMAC', 'OR [95%CI]', 'Meta P-value']


CVDI_MASKS = ['LOF (MAF<0.1%)', 'LOF + missense0.8 (MAF<0.1%)', 'missense0.5 (MAF<0.001%)']


# ST3's FDR1% block names one label, then FDR5% follows.
CVDI_CUTOFFS = [
    ('FDR1%', 'LOF (MAF<0.1%)', 1e-6),
    (None, 'LOF + missense0.8 (MAF<0.1%)', 2e-7),
    (None, 'Missense0.5 (MAF<0.001%)', 5e-8),
    (None, 'Cauchy', 3e-7),
    ('FDR5%', 'LOF (MAF<0.1%)', 1e-3),
    (None, 'Cauchy', 1e-3),
]


def cvdi_row(**cells: Any) -> dict[Any, Any]:
    """An ST6 row: `phenotype`, `Gene ID Ensembl` and `Gene`, plus `(mask, field)` cells."""
    row: dict[Any, Any] = {'phenotype': 'Type_2_diabetes', 'Gene ID Ensembl': 'ENSG1', 'Gene': 'G1'}
    return row | cells


def _st6(rows: Sequence[dict[Any, Any]], masks: Sequence[str], footer: list[Any]) -> list[list[Any]]:
    """ST6 as the sheet lays it out: a title, three header rows (ancestry, mask, field), the rows, a footnote.

    The `EUR ancestry` group repeats the first mask with values that would pass every cutoff.
    """
    groups = [*(('ALL ancestry', mask, field) for mask in masks for field in CVDI_FIELDS)]
    groups += [('ALL ancestry', 'Cauchy', 'Cauchy P-value')]
    groups += [('EUR ancestry', masks[0], field) for field in CVDI_FIELDS]
    levels: list[list[Any]] = [list(CVDI_INDEX), [None] * 3, [None] * 3]
    for i, group in enumerate(groups):
        for level, name in enumerate(group):
            levels[level].append(name if i == 0 or groups[i - 1][: level + 1] != group[: level + 1] else None)
    eur = {('EUR ancestry', masks[0], 'cMAC'): 99, ('EUR ancestry', masks[0], 'Meta P-value'): 1e-300}
    eur[('EUR ancestry', masks[0], 'OR [95%CI]')] = '9.99 [9.99; 9.99]'
    data = [
        [row.get(c) for c in CVDI_INDEX] + [(row | eur).get(g if g[0] == 'EUR ancestry' else g[1:]) for g in groups]
        for row in rows
    ]
    return [['Supplementary Table 6. Significant gene-trait associations'], *levels, *data, footer]


def write_cvdi(
    tmp: Path,
    st6: Sequence[dict[Any, Any]] = (),
    cutoffs: Sequence[tuple[str | None, str, Any]] = CVDI_CUTOFFS,
    *,
    masks: Sequence[str] = CVDI_MASKS,
    footer: list[Any] | None = None,
) -> str:
    """The CVDI workbook: ST6 associations and ST3 cutoffs, whose other cohorts' cutoffs are all 0.5."""
    cohorts = ['UKB', 'AoU', 'MGB', 'META (no correction)', 'META (overlap corrected)']
    st3 = [
        ['Supplementary Table 3. P-value cutoffs'],
        [None, None, *cohorts],
        ['Significance cutoff', 'Mask', *(['P cutoff'] * len(cohorts))],
        *([label, mask, 0.5, 0.5, 0.5, 0.5, cutoff] for label, mask, cutoff in cutoffs),
        ['FDR: false discovery rate.'],
    ]
    footer = ['Associations are significant at FDR 1%.'] if footer is None else footer
    return write_rows(tmp / 'cvdi.xlsx', {'ST3': st3, 'ST6': _st6(st6, masks, footer)})


ST9_COLUMNS = [
    'Gene', 'Gene ID', 'Phenotype', 'EFO ID', 'QT', 'TEST', 'Mask', 'Variant Mask', 'Freq', 'N', 'BETA', 'SE',
    'LOG10P',
]  # fmt: skip


ST13_COLUMNS = [
    'Gene', 'Gene ID', 'Phenotype', 'EFO ID', 'QT', 'Mask', 'Variant Mask', 'Freq', 'N G&H', 'N UKB', 'BETA', 'SE',
    'LOG10P',
]  # fmt: skip


ST15_COLUMNS = [
    'Gene', 'Gene ID', 'Phenotype', 'EFO ID', 'QT', 'Variant consequence', 'Number for ExWAS', 'Number of Cases',
    'Biallelic Carriers', 'Recessive log(OR)', 'Recessive p-value', 'Significant REC P-value',
]  # fmt: skip


Rows = list[dict[str, Any]]


def write_genes_and_health(tmp: Path, st9: Rows, st13: Rows, st15: Rows) -> str:
    return write_workbook(
        tmp / 'genes_health.xlsx',
        {'ST9': (ST9_COLUMNS, st9), 'ST13': (ST13_COLUMNS, st13), 'ST15': (ST15_COLUMNS, st15)},
    )


S4_COLUMNS = ['Phenotype ID', 'Ancestry', 'Biobank', 'N cases', 'N controls']


S5_COLUMNS = ['Phenotype ID', 'Ancestry', 'Biobank', 'N']


GRANULAR_COLUMNS = [
    'Gene', 'Gene ID', 'Description', 'Phenotype ID', 'meta analyzed', 'class', 'Mask', 'max MAF', 'Pvalue',
    'BETA Burden', 'SE Burden',
]  # fmt: skip


GROUP = 'Cauchy combination with minimum P-value'


PVALUE = 'Min P-value of significant Cauchy associations'


CAUCHY_COLUMNS = ['Gene', 'Gene ID', 'Description', 'Phenotype ID', GROUP, PVALUE]


Sheet = Sequence[dict[str, Any]]


# Binary contributions: two biobanks for EUR, one each for AFR and MID.
S4_ROWS = [
    {'Phenotype ID': 'T2Diab', 'Ancestry': 'EUR', 'Biobank': 'uk-biobank', 'N cases': 100, 'N controls': 900},
    {'Phenotype ID': 'T2Diab', 'Ancestry': 'EUR', 'Biobank': 'all-of-us', 'N cases': 200, 'N controls': 800},
    {'Phenotype ID': 'T2Diab', 'Ancestry': 'AFR', 'Biobank': 'all-of-us', 'N cases': 30, 'N controls': 70},
    {'Phenotype ID': 'T2Diab', 'Ancestry': 'MID', 'Biobank': 'ccpm', 'N cases': 5, 'N controls': 15},
]


# Quantitative contributions: 'all-of-us' is listed twice, identically, as Table S5 does.
S5_ROWS = [
    {'Phenotype ID': 'Height', 'Ancestry': 'EUR', 'Biobank': 'uk-biobank', 'N': 5000},
    {'Phenotype ID': 'Height', 'Ancestry': 'EUR', 'Biobank': 'all-of-us', 'N': 1000},
    {'Phenotype ID': 'Height', 'Ancestry': 'EUR', 'Biobank': 'all-of-us', 'N': 1000},
    {'Phenotype ID': 'Height', 'Ancestry': 'AFR', 'Biobank': 'all-of-us', 'N': 400},
]


def write_brava(
    tmp: Path,
    *,
    s14: Sheet = (),
    s15: Sheet = (),
    s12: Sheet = (),
    s13: Sheet = (),
    s4: Sheet = S4_ROWS,
    s5: Sheet = S5_ROWS,
) -> str:
    return write_workbook(
        tmp / 'brava.xlsx',
        {
            'Table S4': (S4_COLUMNS, s4),
            'Table S5': (S5_COLUMNS, s5),
            'Table S12': (CAUCHY_COLUMNS, list(s12)),
            'Table S13': (CAUCHY_COLUMNS, list(s13)),
            'Table S14': (GRANULAR_COLUMNS, list(s14)),
            'Table S15': (GRANULAR_COLUMNS, list(s15)),
        },
    )
