"""Tests of the VIDRA Bayesian step (`pts.pyspark.vidra_bayes`).

The preprocessing tests run everywhere. The fits need a CmdStan installation (cmdstanpy's
default lookup) and compile the models once into a temporary cache; they are skipped without it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pts.pyspark import vidra_bayes as vb
from pts.pyspark.vidra_utils import stan
from pts.pyspark.vidra_utils.constants import SOURCE_AZ_RARE, SOURCE_CLINVAR, SOURCE_CODING_GWAS, SOURCE_COMMON_QTL

ROW_COLUMNS = [
    'variant',
    'as_gene',
    'as_disease',
    'GsourceLab',
    'GqtlLab',
    'yc',
    'ycse',
    'xc',
    'xcse',
    'bO',
    'bOse',
    'as_clinicalSignificance',
    'as_gpnstar',
    'as_alphamissense',
    'consequence_class',
]


def _rows(*rows: tuple) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=ROW_COLUMNS)


def _qtl(variant: str, disease: str = 'd', yc: float = 0.3, xc: float = 0.6, qtl: int = 0, bo: float = 0.0) -> tuple:
    return (
        variant,
        'g',
        disease,
        SOURCE_COMMON_QTL,
        qtl,
        yc,
        0.05,
        xc,
        0.1,
        bo,
        2.0 if bo == 0 else 0.3,
        0.0,
        0.7,
        0.8,
        'other',
    )


def _rare(variant: str, disease: str = 'd', yc: float = 1.2, bo: float = 0.0, gpn: float = 0.2) -> tuple:
    return (
        variant,
        'g',
        disease,
        SOURCE_AZ_RARE,
        2,
        yc,
        0.3,
        0.0,
        0.1,
        bo,
        2.0 if bo == 0 else 0.3,
        0.0,
        gpn,
        0.1,
        'missense',
    )


def _clinvar(variant: str, disease: str = 'd', significance: float = 1.0) -> tuple:
    return (variant, 'g', disease, SOURCE_CLINVAR, 2, 0.0, 0.14, 0.0, 0.1, 0.0, 2.0, significance, 0.3, 0.2, 'missense')


def _coding(variant: str, disease: str = 'd', yc: float = 0.2) -> tuple:
    return (variant, 'g', disease, SOURCE_CODING_GWAS, 2, yc, 0.04, 0.0, 0.1, 0.0, 2.0, 0.0, 0.5, 0.6, 'missense')


# -------------------------------------------------------------------------------------------
# Preprocessing
# -------------------------------------------------------------------------------------------


def test_impute_scores_by_consequence_class():
    gene = pd.DataFrame({
        'consequence_class': ['missense', 'missense', 'lof', 'other', 'inframe_or_splice', 'unknown', None],
        'as_gpnstar': [0.2, None, 0.9, None, None, 0.5, None],
        'as_alphamissense': [0.4, 0.6, None, 0.3, None, None, None],
    })
    out = vb.impute_scores(gene)
    assert out['as_gpnstar'].tolist() == pytest.approx([0.2, 0.2, 0.0, 1.0, 0.2, 0.5, 0.2])
    assert out['as_alphamissense'].tolist() == pytest.approx([0.4, 0.6, 0.0, 0.3, 0.5, 0.5, 0.5])


def test_impute_scores_falls_back_to_any_observed_then_benign():
    gene = pd.DataFrame({
        'consequence_class': ['other', 'inframe_or_splice', 'unknown'],
        'as_gpnstar': [0.4, None, None],
        'as_alphamissense': [None, None, None],
    })
    out = vb.impute_scores(gene)
    assert out['as_gpnstar'].tolist() == pytest.approx([0.4, 0.4, 0.4])
    assert out['as_alphamissense'].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_preprocess_gene_dedups_fills_and_drops_lone_coding_variants():
    gene = _rows(
        _clinvar('v1', significance=0.5),
        _clinvar('v1', significance=1.0),
        _qtl('v1', yc=0.3),
        _coding('v2', disease='d'),  # alone in its group: dropped
        _coding('v3', disease='e'),
        _coding('v4', disease='e'),
    )
    gene.loc[gene['variant'] == 'v3', 'xc'] = np.nan
    out = vb.preprocess_gene(gene)
    assert sorted(out['variant']) == ['v1', 'v1', 'v3', 'v4']
    clinvar = out[(out['variant'] == 'v1') & (out['GsourceLab'] == SOURCE_CLINVAR)]
    assert clinvar['as_clinicalSignificance'].tolist() == [1.0]
    assert out[out['variant'] == 'v3']['xc'].tolist() == [0.0]
    assert out['GsourceLab'].dtype.kind == 'i'
    assert list(out.columns) == ROW_COLUMNS


def test_preprocess_gene_keeps_empty_frames():
    assert vb.preprocess_gene(_rows()).empty


def test_process_pair_without_fit_returns_full_schema():
    columns = list(vb.RESULT_COLUMNS)
    assert vb.process_pair(_rows(), vb.VidraBayesSettings()).columns.tolist() == columns
    assert len(vb.result_schema().fields) == len(columns)
    assert len(vb.result_schema(True).fields) == len(vb.VARIANT_RESULT_COLUMNS)


def test_unidentified_pairs_do_not_call_stan():
    settings = vb.VidraBayesSettings()
    clinvar = vb.fit_pair(_rows(_clinvar('v1')), settings)
    assert clinvar[0]['model'] == vb.MODEL_UNIDENTIFIED
    assert clinvar[0]['error'] is None
    assert clinvar[0]['sources'] == [SOURCE_CLINVAR]
    rare = vb.fit_pair(_rows(_rare('v1', bo=0.0)), settings)
    assert rare[0]['model'] == vb.MODEL_UNIDENTIFIED
    assert vb.fit_pair(_rows(_clinvar('v1')), settings, variant_outputs=True) == []


def test_failed_fit_is_an_error_row(monkeypatch):
    def boom(*_, **__):
        raise RuntimeError('no stan here')

    monkeypatch.setattr(vb, 'load_models', boom)
    rows = vb.fit_pair(_rows(_qtl('v1')), vb.VidraBayesSettings())
    assert rows[0]['model'] == vb.MODEL_ERROR
    assert 'no stan here' in rows[0]['error']
    assert rows[0]['hasBurden'] is False
    frame = vb.process_pair(_rows(_qtl('v1')), vb.VidraBayesSettings())
    assert frame['mean'].isna().all()


# -------------------------------------------------------------------------------------------
# Fits (CmdStan needed)
# -------------------------------------------------------------------------------------------


def _cmdstan_available() -> bool:
    try:
        from cmdstanpy import cmdstan_path

        return Path(cmdstan_path()).exists()
    except Exception:
        return False


needs_stan = pytest.mark.skipif(not _cmdstan_available(), reason='no CmdStan installation')


@pytest.fixture(scope='module')
def settings(tmp_path_factory) -> vb.VidraBayesSettings:
    os.environ['SPARK_LOCAL_DIRS'] = str(tmp_path_factory.mktemp('stan'))
    stan.load_models.cache_clear()
    return vb.VidraBayesSettings(advi={'max_iter': 2000, 'draws': 200})


@needs_stan
@pytest.mark.slow
def test_single_qtl_pair_recovers_the_wald_ratio(settings):
    rows = vb.fit_pair(_rows(_qtl('v1', yc=0.3, xc=0.6)), settings)
    assert len(rows) == 1
    row = rows[0]
    assert row['model'] == vb.MODEL_SINGLE
    assert row['parameter'] == 'slope'
    assert row['mean'] == pytest.approx(0.5, abs=0.15)
    assert row['probabilityPositive'] > 0.95
    assert row['variantCount'] == 1 and row['qtlTypes'] == [0]


@needs_stan
@pytest.mark.slow
def test_multi_variant_pair_reports_every_parameter_and_pools_present_sources(settings):
    pair = _rows(_qtl('v1', yc=0.3, xc=0.6, bo=0.5), _qtl('v2', yc=-0.2, xc=-0.4, bo=0.5), _rare('v3', bo=0.5, gpn=0.2))
    rows = vb.fit_pair(pair, settings)
    frame = pd.DataFrame(rows)
    assert set(frame['parameter']) == {'slope', 'intercept', *vb.SLOPE_RANDOM_NAMES, 'meta_slope'}
    assert frame['model'].unique().tolist() == ['multiple_variant_fullrank']
    assert frame['hasBurden'].all()
    assert frame['sources'].iloc[0] == [SOURCE_COMMON_QTL, SOURCE_AZ_RARE]
    eqtl = frame.set_index('parameter').loc['slope_eqtl']
    assert eqtl['mean'] == pytest.approx(0.5, abs=0.3)


@needs_stan
@pytest.mark.slow
def test_variant_outputs_have_one_row_per_variant_and_parameter(settings):
    pair = _rows(_qtl('v1'), _qtl('v2', yc=-0.2, xc=-0.4))
    rows = pd.DataFrame(vb.fit_pair(pair, settings, variant_outputs=True))
    assert sorted(rows['variantId'].unique()) == ['v1', 'v2']
    assert set(rows['parameter']) == set(vb.VARIANT_PARAMETERS)
    assert len(rows) == 2 * len(vb.VARIANT_PARAMETERS)


@needs_stan
@pytest.mark.slow
def test_vidra_bayes_end_to_end(spark, tmp_path, settings):
    source = tmp_path / 'analysis_ready'
    rows = _rows(
        _qtl('v1'), _qtl('v2', yc=-0.2, xc=-0.4), _clinvar('v3', disease='e'), _rare('v4', disease='f', bo=0.4)
    )
    spark.createDataFrame(rows).write.partitionBy('as_gene').parquet(str(source))
    genes = tmp_path / 'genes.txt'
    genes.write_text('g\n')
    destination = {'results': str(tmp_path / 'results'), 'stan_inputs': str(tmp_path / 'stan_inputs')}
    vb.vidra_bayes(
        str(source),
        destination,
        {**settings.model_dump(), 'genes': str(genes), 'save_stan_inputs': True, 'output_partitions': 1},
        {},
    )
    results = spark.read.parquet(destination['results']).toPandas()
    assert set(results['diseaseId']) == {'d', 'e', 'f'}
    by_disease = results.groupby('diseaseId')['model'].first()
    assert by_disease['d'] == 'multiple_variant_fullrank'
    assert by_disease['e'] == vb.MODEL_UNIDENTIFIED
    assert by_disease['f'] == vb.MODEL_SINGLE
    assert spark.read.parquet(destination['stan_inputs']).count() == 4
