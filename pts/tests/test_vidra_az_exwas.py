"""Tests of the VIDRA AZ ExWAS transformer."""

from __future__ import annotations

import bz2
from pathlib import Path

import polars as pl
import pytest
from otter.config.model import Config

from pts.transformers.vidra.az_exwas import COLUMNS, AzExwasSettings, az_exwas, select_associations

HEADER = (
    'Variant,Variant type,Phenotype,Category,Model,Consequence type,Gene,Transcript,cDNA change,Amino acid change,'
    'Exon rank,No. cases,No. AA cases,No. AB cases,No. BB cases,Case AAF,% AB or BB cases,% BB cases,No. controls,'
    'No. AA controls,No. AB controls,No. BB controls,Control AAF,% AB or BB controls,% BB controls,p-value,'
    'Odds ratio,Odds ratio LCI,Odds ratio UCI,Case MAF,Control MAF'
)


def _row(variant: str, model: str, pvalue: str, gene: str = "'SLC35A3'") -> str:
    return (
        f'{variant},snv,131465#Source of report of J30 vasomotor and allergic rhinitis,Chapter X,{model},'
        f"missense_variant,{gene},ENST00000638876,c.193G>A,p.Val65Ile,'3/5',41539,41530,9,0,0.0001083,0.0002,0,"
        f'377295,377273,22,0,2.915e-05,5.8e-05,0,{pvalue},3.716,1.711,8.0706,0.0001083,2.915e-05'
    )


def _export(path: Path, rows: list[str]) -> str:
    with bz2.open(path, 'wt') as stream:
        stream.write('\n'.join([HEADER, *rows]) + '\n')
    return str(path)


def test_select_associations_filters_models_and_pvalue_and_types_columns():
    export = pl.read_csv(
        '\n'.join([
            HEADER,
            _row('1-100007039-G-A', 'allelic', '1e-9'),
            _row('1-100007039-G-A', 'dominant', '1e-9'),
            _row('2-5-C-T', 'allelic', '0.0025'),
            _row('3-5-C-T', 'allelic', 'n/a'),
        ]).encode(),
        schema_overrides=dict.fromkeys(COLUMNS, pl.String),
        infer_schema=False,
    ).lazy()
    kept = select_associations(export, AzExwasSettings()).collect()
    assert kept.columns == ['variantId', *COLUMNS.values()]
    assert kept['variantId'].to_list() == ['1_100007039_G_A']
    assert kept['geneSymbol'].to_list() == ['SLC35A3']
    assert kept['model'].to_list() == ['allelic']
    assert kept.schema['pValue'] == pl.Float64
    assert kept.schema['nCases'] == pl.Int64
    assert kept['oddsRatioUpper'][0] == pytest.approx(8.0706)

    loose = select_associations(export, AzExwasSettings(models=['allelic', 'dominant'], p_value_threshold=0.01))
    assert loose.collect().height == 3


def test_az_exwas_decompresses_filters_and_writes(tmp_path: Path):
    export = _export(
        tmp_path / 'export.csv.bz2', [_row('1-10-A-G', 'allelic', '1e-9'), _row('1-20-A-G', 'allelic', '0.1')]
    )
    config = Config(step='vidra_az_exwas', steps=['vidra_az_exwas'], work_path=tmp_path / 'work')
    config.work_path.mkdir()
    az_exwas(export, str(tmp_path / 'out'), {}, config)
    written = pl.read_parquet(tmp_path / 'out')
    assert written['variantId'].to_list() == ['1_10_A_G']


def test_az_exwas_refuses_an_empty_result(tmp_path: Path):
    export = _export(tmp_path / 'export.csv.bz2', [_row('1-20-A-G', 'allelic', '0.1')])
    config = Config(step='vidra_az_exwas', steps=['vidra_az_exwas'], work_path=tmp_path / 'work')
    config.work_path.mkdir()
    with pytest.raises(ValueError, match='no ExWAS associations kept'):
        az_exwas(export, str(tmp_path / 'out'), {}, config)
