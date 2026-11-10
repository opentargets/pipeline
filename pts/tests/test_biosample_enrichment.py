"""Tests for the biosample_enrichment transformer."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pts.transformers.biosample_enrichment import (
    OUTPUT_SCHEMA,
    Panel,
    biosample_enrichment,
    biosample_sources,
    gene_covariates,
    gwas_studies,
    specificity_matrix,
)
from tests.credible_set_fixtures import published_credible_set

N_GENES = 400
TISSUES = ['UBERON_A', 'UBERON_B', 'UBERON_HOT', 'UBERON_C', 'UBERON_D']
CELLTYPES = ['CL_1', 'CL_2', 'CL_3']
TABULA_PANELS = [
    {'datasourceId': 'tabula_sapiens', 'biosampleType': 'celltype'},
    {'datasourceId': 'tabula_sapiens', 'biosampleType': 'tissue_celltype'},
]


def _target(n: int = N_GENES) -> pl.DataFrame:
    """Genes 50 kb apart over two chromosomes, the last ten not protein coding."""
    half = n // 2
    starts = np.r_[np.arange(half), np.arange(n - half)] * 50_000 + 1_000
    rng = np.random.default_rng(3)
    lengths = rng.integers(1_000, 200_000, size=n)
    return pl.DataFrame({
        'id': [f'ENSG{i:011d}' for i in range(n)],
        'biotype': ['protein_coding'] * (n - 10) + ['lncRNA'] * 10,
        'tss': starts,
        'genomicLocation': [
            {'chromosome': '1' if i < half else '2', 'start': int(s), 'end': int(s + ln), 'strand': 1}
            for i, (s, ln) in enumerate(zip(starts, lengths, strict=True))
        ],
    })


def _baseline_expression(genes: list[str], rng: np.random.Generator) -> tuple[pl.DataFrame, np.ndarray]:
    specificity = rng.beta(0.3, 3.0, size=(len(genes), len(TISSUES)))
    rows = [
        {
            'targetId': gene,
            'datasourceId': 'gtex',
            'tissueBiosampleId': tissue,
            'celltypeBiosampleId': None,
            'tissueBiosampleFromSource': f'{tissue} source',
            'celltypeBiosampleFromSource': None,
            'specificity_score': float(specificity[g, t]),
        }
        for g, gene in enumerate(genes)
        for t, tissue in enumerate(TISSUES)
    ]
    # rows a tissue panel must ignore: another datasource, and a tissue x cell type pair
    rows += [
        {
            'targetId': genes[0],
            'datasourceId': 'other',
            'tissueBiosampleId': 'UBERON_X',
            'celltypeBiosampleId': None,
            'tissueBiosampleFromSource': 'X',
            'celltypeBiosampleFromSource': None,
            'specificity_score': 1.0,
        },
        {
            'targetId': genes[0],
            'datasourceId': 'gtex',
            'tissueBiosampleId': 'UBERON_A',
            'celltypeBiosampleId': 'CL_1',
            'tissueBiosampleFromSource': 'A pair',
            'celltypeBiosampleFromSource': 'CL 1',
            'specificity_score': 1.0,
        },
    ]
    # a repeated gene and tissue, as several source ids mapping to one target produce
    rows.append(rows[0])
    # tabula_sapiens: cell types on their own, and CL_1 profiled in every tissue, each pair with its
    # tissue's specificity, so UBERON_HOT|CL_1 carries the planted signal
    cell_specificity = rng.beta(0.3, 3.0, size=(len(genes), len(CELLTYPES)))
    rows += [
        {
            'targetId': gene,
            'datasourceId': 'tabula_sapiens',
            'tissueBiosampleId': None,
            'celltypeBiosampleId': celltype,
            'tissueBiosampleFromSource': None,
            'celltypeBiosampleFromSource': f'{celltype} cell',
            'specificity_score': float(cell_specificity[g, c]),
        }
        for g, gene in enumerate(genes)
        for c, celltype in enumerate(CELLTYPES)
    ]
    rows += [
        {
            'targetId': gene,
            'datasourceId': 'tabula_sapiens',
            'tissueBiosampleId': tissue,
            'celltypeBiosampleId': 'CL_1',
            'tissueBiosampleFromSource': f'{tissue} organ',
            'celltypeBiosampleFromSource': 'CL_1 cell',
            'specificity_score': float(specificity[g, t]),
        }
        for g, gene in enumerate(genes)
        for t, tissue in enumerate(TISSUES)
    ]
    return pl.DataFrame(rows, infer_schema_length=None), specificity


def _write(frame: pl.DataFrame, path: Path) -> str:
    path.mkdir(parents=True)
    frame.write_parquet(path / 'part-0.parquet')
    return str(path)


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, str]:
    """Input datasets for four GWAS studies and one eQTL study.

    GCST_HOT prioritises the genes most specific to UBERON_HOT. GCST_TWIN has the same single
    disease and one gene of its own. GCST_LNC prioritises only non-coding genes. GCST_EMPTY has the
    same disease as GCST_HOT but no credible sets. QTL is not a GWAS study.
    """
    rng = np.random.default_rng(11)
    target = _target()
    genes = target['id'].to_list()
    coding = genes[:-10]
    expression, specificity = _baseline_expression(coding, rng)
    hot = np.argsort(-specificity[:, TISSUES.index('UBERON_HOT')])[:40]

    l2g = pl.DataFrame({
        'credibleSetId': [f'hot{i}' for i in range(len(hot))] + ['twin0', 'lnc0', 'lnc1', 'qtl0'],
        'geneId': [coding[g] for g in hot] + [coding[5], genes[-1], genes[-2], coding[0]],
        'score': [*list(rng.uniform(0.3, 0.95, size=len(hot))), 0.4, 0.9, 0.8, 0.99],
    })
    credible_set = published_credible_set(*[
        {'id': set_id, 'studyId': study_id}
        for set_id, study_id in zip(
            l2g['credibleSetId'],
            ['GCST_HOT'] * len(hot) + ['GCST_TWIN', 'GCST_LNC', 'GCST_LNC', 'QTL'],
            strict=True,
        )
    ])
    study = pl.DataFrame({
        'studyId': ['GCST_HOT', 'GCST_TWIN', 'GCST_LNC', 'QTL', 'GCST_EMPTY'],
        'studyType': ['gwas', 'gwas', 'gwas', 'eqtl', 'gwas'],
        'diseaseIds': [['EFO_1'], ['EFO_1', 'EFO_1'], ['EFO_3'], [], ['EFO_1']],
    })
    return {
        'l2g_prediction': _write(l2g, tmp_path / 'l2g'),
        'credible_set': _write(credible_set, tmp_path / 'cs'),
        'study': _write(study, tmp_path / 'study'),
        'target': _write(target, tmp_path / 'target'),
        'baseline_expression': _write(expression, tmp_path / 'be'),
        'out': str(tmp_path / 'out'),
    }


def _run(workspace: dict[str, str], **settings) -> pl.DataFrame:
    source = {k: v for k, v in workspace.items() if k != 'out'}
    settings = {'panels': [{'datasourceId': 'gtex', 'biosampleType': 'tissue'}], **settings}
    biosample_enrichment(source, workspace['out'], settings, None)
    return pl.read_parquet(f'{workspace["out"]}/*.parquet')


def test_output_has_the_pinned_schema(workspace) -> None:
    out = _run(workspace)
    assert out.schema == OUTPUT_SCHEMA


def test_the_planted_biosample_gets_the_top_p(workspace) -> None:
    hot = _run(workspace).filter(pl.col('studyId') == 'GCST_HOT').sort('pValue')
    assert hot['biosampleId'][0] == 'UBERON_HOT'
    assert (hot['beta'][0] / hot['betaError'][0]) > 3
    assert hot['pValue'][0] < 0.05 / len(TISSUES)


def test_rows_carry_the_panel_and_approach(workspace) -> None:
    out = _run(workspace)
    assert out['biosampleDataset'].unique().to_list() == ['gtex']
    assert out['enrichmentApproach'].unique().to_list() == ['l2g_regression']
    assert out['biosampleFromSource'].to_list() == [f'{b} source' for b in out['biosampleId']]


def test_only_gwas_studies_with_a_scored_gene_in_the_universe_get_rows(workspace) -> None:
    out = _run(workspace)
    assert sorted(out['studyId'].unique()) == ['GCST_HOT', 'GCST_TWIN']
    assert out.group_by('studyId').len()['len'].to_list() == [len(TISSUES)] * 2
    assert out['diseaseId'].null_count() == out.height


def test_rows_are_unique_on_the_key(workspace) -> None:
    panels = [{'datasourceId': 'gtex', 'biosampleType': 'tissue'}, *TABULA_PANELS]
    for settings in ({}, {'pool_by_disease': True}):
        workspace['out'] += '_x'
        out = _run(workspace, panels=panels, **settings)
        key = out.select('diseaseId', 'studyId', 'biosampleId', 'tissueBiosampleId', 'biosampleDataset')
        assert not key.is_duplicated().any()


def test_tissue_celltype_pairs_report_the_cell_type_and_its_tissue(workspace) -> None:
    out = _run(workspace, panels=TABULA_PANELS)
    pairs = out.filter(pl.col('tissueBiosampleId').is_not_null())
    assert pairs['biosampleDataset'].unique().to_list() == ['tabula_sapiens']
    assert sorted(pairs.filter(pl.col('studyId') == 'GCST_HOT')['tissueBiosampleId']) == sorted(TISSUES)
    assert pairs['biosampleId'].unique().to_list() == ['CL_1']
    assert pairs['biosampleFromSource'].to_list() == [f'{t} organ | CL_1 cell' for t in pairs['tissueBiosampleId']]
    # the cell type panel's rows have no tissue, so CL_1 on its own is a separate row
    cells = out.filter(pl.col('tissueBiosampleId').is_null(), pl.col('studyId') == 'GCST_HOT')
    assert sorted(cells['biosampleId']) == CELLTYPES


def test_the_planted_tissue_celltype_pair_gets_the_top_p(workspace) -> None:
    hot = _run(workspace, panels=TABULA_PANELS[1:]).filter(pl.col('studyId') == 'GCST_HOT').sort('pValue')
    assert hot.row(0, named=True)['tissueBiosampleId'] == 'UBERON_HOT'
    # a pair with its tissue's specificity gives exactly that tissue's fit
    workspace['out'] += '_gtex'
    gtex = _run(workspace).filter(pl.col('studyId') == 'GCST_HOT').sort('biosampleId')
    np.testing.assert_allclose(hot.sort('tissueBiosampleId')['beta'].to_numpy(), gtex['beta'].to_numpy())
    np.testing.assert_allclose(hot.sort('tissueBiosampleId')['pValue'].to_numpy(), gtex['pValue'].to_numpy())


def test_pooling_reports_one_disease_once(workspace) -> None:
    single = _run(workspace).filter(pl.col('studyId') == 'GCST_HOT').sort('biosampleId')
    workspace['out'] += '_pooled'
    pooled = _run(workspace, pool_by_disease=True)
    # exactly one of studyId and diseaseId is set
    assert (pooled['studyId'].is_null() ^ pooled['diseaseId'].is_null()).all()
    disease = pooled.filter(pl.col('diseaseId') == 'EFO_1').sort('biosampleId')
    assert disease.height == len(TISSUES)
    assert pooled['diseaseId'].drop_nulls().unique().to_list() == ['EFO_1']
    # GCST_LNC's only genes are non-coding, so EFO_3 has no rows; no study is reported on its own
    assert pooled['studyId'].null_count() == pooled.height
    # GCST_TWIN's own gene joins the 40, so the pooled fit differs from GCST_HOT's alone
    assert not np.allclose(disease['beta'].to_numpy(), single['beta'].to_numpy())


def test_standard_error_kinds_change_only_the_uncertainty(workspace) -> None:
    region = _run(workspace).sort('studyId', 'biosampleId')
    workspace['out'] += '_gene'
    gene = _run(workspace, standard_errors='gene').sort('studyId', 'biosampleId')
    np.testing.assert_allclose(region['beta'].to_numpy(), gene['beta'].to_numpy())
    assert not np.allclose(region['betaError'].to_numpy(), gene['betaError'].to_numpy())


def test_an_unknown_standard_error_kind_is_rejected(workspace) -> None:
    with pytest.raises(ValueError, match='standard_errors'):
        _run(workspace, standard_errors='hc3')


def test_specificity_matrix_reads_one_panel_and_collapses_repeats(workspace) -> None:
    matrix = specificity_matrix(
        pl.scan_parquet(f'{workspace["baseline_expression"]}/*.parquet'), Panel('gtex', 'tissue')
    )
    assert matrix.columns == ['geneId', *sorted(TISSUES)]
    assert matrix.height == N_GENES - 10
    assert matrix['geneId'].is_unique().all()


def test_biosample_sources_read_one_panel(workspace) -> None:
    sources = biosample_sources(
        pl.scan_parquet(f'{workspace["baseline_expression"]}/*.parquet'), Panel('gtex', 'tissue')
    ).sort('biosampleId')
    assert sources['biosampleId'].to_list() == sorted(TISSUES)
    # UBERON_A's tissue x cell type row is not part of the tissue panel
    assert sources['biosampleFromSource'].to_list() == [f'{t} source' for t in sorted(TISSUES)]


def test_several_source_biosamples_are_joined() -> None:
    expression = pl.DataFrame(
        {
            'datasourceId': ['gtex'] * 3,
            'tissueBiosampleId': ['U1', 'U1', 'U2'],
            'celltypeBiosampleId': [None, None, None],
            'tissueBiosampleFromSource': ['Skin b', 'Skin a', 'Liver'],
            'celltypeBiosampleFromSource': [None, None, None],
        },
        schema_overrides={'celltypeBiosampleId': pl.String, 'celltypeBiosampleFromSource': pl.String},
    )
    sources = biosample_sources(expression.lazy(), Panel('gtex', 'tissue'))
    assert dict(sources.select('biosampleId', 'biosampleFromSource').iter_rows()) == {
        'U1': 'Skin a; Skin b',
        'U2': 'Liver',
    }


def test_specificity_matrix_names_a_pair_by_its_tissue_and_cell_type(workspace) -> None:
    expression = pl.scan_parquet(f'{workspace["baseline_expression"]}/*.parquet')
    matrix = specificity_matrix(expression, Panel('tabula_sapiens', 'tissue_celltype'))
    assert matrix.columns == ['geneId', *sorted(f'{t}|CL_1' for t in TISSUES)]
    # the gtex panel's lone UBERON_A x CL_1 row belongs to another datasource
    assert matrix.height == N_GENES - 10


def test_the_tissue_panel_has_no_tissue_context(workspace) -> None:
    out = _run(workspace)
    assert out['tissueBiosampleId'].null_count() == out.height


def test_an_empty_panel_is_an_error(workspace) -> None:
    expression = pl.scan_parquet(f'{workspace["baseline_expression"]}/*.parquet')
    with pytest.raises(ValueError, match='no specificity scores'):
        specificity_matrix(expression, Panel('gtex', 'celltype'))


def test_a_panel_needs_a_known_biosample_type() -> None:
    with pytest.raises(ValueError, match='biosampleType'):
        Panel.from_settings({'datasourceId': 'gtex', 'biosampleType': 'organ'})


def test_gene_covariates_count_tsss_within_the_window() -> None:
    target = pl.DataFrame({
        'id': ['a', 'b', 'c', 'd', 'e'],
        'biotype': ['protein_coding'] * 5,
        'tss': [100, 600_000, 1_100_101, 50, 7],
        'genomicLocation': [
            {'chromosome': '1', 'start': 100, 'end': 1_100, 'strand': 1},
            {'chromosome': '1', 'start': 600_000, 'end': 600_010, 'strand': 1},
            {'chromosome': '1', 'start': 1_100_101, 'end': 1_100_201, 'strand': 1},
            {'chromosome': '2', 'start': 50, 'end': 150, 'strand': 1},
            {'chromosome': '2', 'start': 7, 'end': 7, 'strand': 1},  # zero length: dropped
        ],
    })
    cov = gene_covariates(target.lazy(), density_window=1_000_000).sort('geneId')
    assert cov['geneId'].to_list() == ['a', 'b', 'c', 'd']
    # a and c are 1,000,001 apart, just outside each other's window; b sees both
    assert cov['density'].to_list() == [2, 3, 2, 1]
    np.testing.assert_allclose(cov['logLength'].to_numpy(), np.log10([1_000, 10, 100, 100]))


def test_gwas_studies_pool_only_single_disease_studies() -> None:
    study = pl.DataFrame({
        'studyId': ['s1', 's2', 's3', 's4', 's5', 'q'],
        'studyType': ['gwas', 'gwas', 'gwas', 'gwas', 'gwas', 'eqtl'],
        'diseaseIds': [['A'], ['A', 'A'], ['A', 'B'], [], None, ['A']],
    })
    pooled = {r[0]: r[1:] for r in gwas_studies(study.lazy(), pool_by_disease=True).collect().iter_rows()}
    assert pooled == {
        's1': ('disease:A', 'A'),
        's2': ('disease:A', 'A'),
        's3': ('study:s3', None),
        's4': ('study:s4', None),
        's5': ('study:s5', None),
    }
    unpooled = {r[0]: r[1:] for r in gwas_studies(study.lazy(), pool_by_disease=False).collect().iter_rows()}
    assert unpooled == {s: (s, None) for s in ['s1', 's2', 's3', 's4', 's5']}
