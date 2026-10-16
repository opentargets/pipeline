"""Tests for the search_facet transformer."""

from pathlib import Path

import polars as pl

from pts.transformers.search_facet import (
    DEFAULT_CATEGORIES,
    DISEASE_COLUMNS,
    TARGET_COLUMNS,
    disease_facets,
    disease_name_facets,
    gene_set_facets,
    search_facet,
    simple_facet,
    subcellular_location_facets,
    target_class_facets,
    therapeutic_area_facets,
    tractability_facets,
)

CATEGORIES = DEFAULT_CATEGORIES

LOCATION = pl.Struct({'location': pl.String, 'termSL': pl.String})

TARGET_SCHEMA = {
    'id': pl.String,
    'approvedSymbol': pl.String,
    'approvedName': pl.String,
    'subcellularLocations': pl.List(LOCATION),
}
DISEASE_SCHEMA = {'id': pl.String, 'name': pl.String, 'therapeuticAreas': pl.List(pl.String)}
GENE_SET_SCHEMA = {
    'id': pl.String,
    'label': pl.String,
    'source': pl.String,
    'category': pl.String,
    'entityIds': pl.List(pl.String),
}
TRACTABILITY_SCHEMA = {'targetId': pl.String, 'modality': pl.String, 'category': pl.String, 'value': pl.Boolean}


def target(rows: list[dict]) -> pl.LazyFrame:
    return pl.LazyFrame([{k: row.get(k) for k in TARGET_SCHEMA} for row in rows], schema=TARGET_SCHEMA)


def disease(rows: list[tuple]) -> pl.LazyFrame:
    return pl.LazyFrame(rows, schema=DISEASE_SCHEMA, orient='row')


def gene_sets(rows: list[tuple]) -> pl.LazyFrame:
    return pl.LazyFrame(rows, schema=GENE_SET_SCHEMA, orient='row')


def tractability(rows: list[tuple]) -> pl.LazyFrame:
    return pl.LazyFrame(rows, schema=TRACTABILITY_SCHEMA, orient='row')


def by_key(lf: pl.LazyFrame, key: str = 'datasourceId') -> dict:
    return {row[key]: row for row in lf.collect().to_dicts()}


def test_disease_name_facet() -> None:
    rows = disease_name_facets(disease([('EFO_1', 'breast cancer', None), ('EFO_2', 'lung cancer', [])]), CATEGORIES)
    assert by_key(rows) == {
        'EFO_1': {'label': 'breast cancer', 'category': 'Disease', 'datasourceId': 'EFO_1', 'entityIds': ['EFO_1']},
        'EFO_2': {'label': 'lung cancer', 'category': 'Disease', 'datasourceId': 'EFO_2', 'entityIds': ['EFO_2']},
    }


def test_therapeutic_area_facet_groups_diseases_under_the_area_name() -> None:
    rows = therapeutic_area_facets(
        disease([
            ('EFO_10', 'neoplasm', []),
            ('EFO_1', 'breast cancer', ['EFO_10']),
            ('EFO_2', 'lung cancer', ['EFO_10', None]),
            ('EFO_3', 'orphan', ['EFO_99']),
            ('EFO_4', 'none', None),
        ]),
        CATEGORIES,
    )
    assert by_key(rows) == {
        'EFO_10': {
            'label': 'neoplasm',
            'category': 'Therapeutic Area',
            'datasourceId': 'EFO_10',
            'entityIds': ['EFO_1', 'EFO_2'],
        },
    }


def test_tractability_facet_keeps_true_values_only() -> None:
    rows = tractability_facets(
        tractability([
            ('ENSG2', 'SM', 'Approved Drug', True),
            ('ENSG1', 'SM', 'Approved Drug', True),
            ('ENSG1', 'SM', 'Druggable Family', False),
            ('ENSG1', 'SM', 'Unknown', None),
            ('ENSG1', 'AB', 'Advanced Clinical', True),
            ('ENSG1', 'XX', 'Other', True),
        ]),
        CATEGORIES,
    ).collect()
    assert sorted(rows.select('category', 'label', 'entityIds', 'datasourceId').rows()) == [
        ('Tractability Antibody', 'Advanced Clinical', ['ENSG1'], None),
        ('Tractability Small Molecule', 'Approved Drug', ['ENSG1', 'ENSG2'], None),
        ('XX', 'Other', ['ENSG1'], None),
    ]


def test_gene_set_facets_keep_one_source_keyed_by_set_id() -> None:
    rows = gene_set_facets(
        gene_sets([
            ('GO:1', 'binding', 'go', 'GO:MF', ['ENSG1', 'ENSG2']),
            ('GO:3', 'nucleus', 'go', 'GO:CC', ['ENSG1']),
            ('R-HSA-1', 'Cell Cycle', 'reactome', 'Reactome', ['ENSG1']),
        ]),
        'go',
    )
    assert by_key(rows) == {
        'GO:1': {'label': 'binding', 'category': 'GO:MF', 'entityIds': ['ENSG1', 'ENSG2'], 'datasourceId': 'GO:1'},
        'GO:3': {'label': 'nucleus', 'category': 'GO:CC', 'entityIds': ['ENSG1'], 'datasourceId': 'GO:3'},
    }


def test_subcellular_location_facet() -> None:
    rows = subcellular_location_facets(
        target([
            {'id': 'ENSG2', 'subcellularLocations': [{'location': 'nucleus', 'termSL': 'SL-0191'}]},
            {'id': 'ENSG1', 'subcellularLocations': [{'location': 'nucleus', 'termSL': 'SL-0191'}]},
            {'id': 'ENSG3', 'subcellularLocations': [{'location': 'cytosol', 'termSL': None}]},
            {'id': 'ENSG4', 'subcellularLocations': []},
        ]),
        CATEGORIES,
    )
    assert by_key(rows, 'label') == {
        'nucleus': {
            'label': 'nucleus',
            'category': 'Subcellular Location',
            'datasourceId': 'SL-0191',
            'entityIds': ['ENSG1', 'ENSG2'],
        },
        'cytosol': {
            'label': 'cytosol',
            'category': 'Subcellular Location',
            'datasourceId': None,
            'entityIds': ['ENSG3'],
        },
    }


def test_target_class_facet_merges_classes_sharing_a_label_with_null_datasource() -> None:
    rows = target_class_facets(
        gene_sets([
            ('1', 'Kinase', 'chembl_target_class', 'ChEMBL Target Class', ['ENSG2', 'ENSG1']),
            ('7', 'Kinase', 'chembl_target_class', 'ChEMBL Target Class', ['ENSG1', 'ENSG3']),
            ('GO:1', 'Kinase', 'go', 'GO:MF', ['ENSG4']),
        ]),
    )
    assert by_key(rows, 'label') == {
        'Kinase': {
            'label': 'Kinase',
            'category': 'ChEMBL Target Class',
            'entityIds': ['ENSG1', 'ENSG2', 'ENSG3'],
            'datasourceId': None,
        },
    }


def test_simple_facet_groups_ids_by_label_and_keeps_null_labels() -> None:
    rows = simple_facet(
        target([
            {'id': 'ENSG2', 'approvedSymbol': 'A'},
            {'id': 'ENSG1', 'approvedSymbol': 'A'},
            {'id': 'ENSG3', 'approvedSymbol': None},
        ]),
        'approvedSymbol',
        'Approved Symbol',
    )
    assert by_key(rows, 'label') == {
        'A': {'label': 'A', 'category': 'Approved Symbol', 'entityIds': ['ENSG1', 'ENSG2'], 'datasourceId': None},
        None: {'label': None, 'category': 'Approved Symbol', 'entityIds': ['ENSG3'], 'datasourceId': None},
    }


def test_disease_facets_column_order_and_sort() -> None:
    rows = disease_facets(
        disease([('EFO_2', 'b', ['EFO_1']), ('EFO_1', 'a', None)]),
        CATEGORIES,
    ).collect()
    assert rows.columns == DISEASE_COLUMNS
    assert rows.select('category', 'label').rows() == [('Disease', 'a'), ('Disease', 'b'), ('Therapeutic Area', 'a')]


def test_search_facet_end_to_end(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    src.mkdir()
    targets = target([
        {
            'id': 'ENSG1',
            'approvedSymbol': 'A1',
            'approvedName': 'alpha',
            'subcellularLocations': [{'location': 'nucleus', 'termSL': 'SL-1'}],
        },
    ]).collect()
    pl.concat([targets, targets.with_columns(pl.lit('ENSG2').alias('id'))], rechunk=False).write_parquet(
        src / 'target.parquet'
    )
    disease([('EFO_2', 'b', ['EFO_1']), ('EFO_1', 'a', None)]).collect().write_parquet(src / 'disease.parquet')
    gene_sets([
        ('GO:1', 'binding', 'go', 'GO:MF', ['ENSG1', 'ENSG2']),
        ('GO:2', 'DNA binding', 'go', 'GO:MF', ['ENSG1', 'ENSG2']),
        ('R-HSA-2', 'p', 'reactome', 'Reactome', ['ENSG1', 'ENSG2']),
        ('1', 'Kinase', 'chembl_target_class', 'ChEMBL Target Class', ['ENSG1', 'ENSG2']),
    ]).collect().write_parquet(src / 'gene_sets.parquet')
    tractability([('ENSG1', 'SM', 'Approved Drug', True)]).collect().write_parquet(src / 'tractability.parquet')

    def run(dst: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
        search_facet(
            {
                'diseases': str(src / 'disease.parquet'),
                'targets': str(src / 'target.parquet'),
                'gene_sets': str(src / 'gene_sets.parquet'),
                'tractability': str(src / 'tractability.parquet'),
            },
            {'targets': str(dst / 'targets'), 'diseases': str(dst / 'diseases')},
            {},
            None,
        )
        return pl.read_parquet(dst / 'targets'), pl.read_parquet(dst / 'diseases')

    targets_out, diseases_out = run(tmp_path / 'a')
    assert targets_out.columns == TARGET_COLUMNS
    assert diseases_out.columns == DISEASE_COLUMNS
    assert dict(targets_out.schema) == {
        'label': pl.String,
        'category': pl.String,
        'entityIds': pl.List(pl.String),
        'datasourceId': pl.String,
    }
    assert sorted(targets_out['category'].unique().to_list()) == [
        'Approved Name',
        'Approved Symbol',
        'ChEMBL Target Class',
        'GO:MF',
        'Reactome',
        'Subcellular Location',
        'Target ID',
        'Tractability Small Molecule',
    ]
    assert targets_out.filter(pl.col('category') == 'GO:MF')['entityIds'].to_list() == [['ENSG1', 'ENSG2']] * 2
    assert diseases_out.height == 3

    again = run(tmp_path / 'b')
    assert targets_out.equals(again[0])
    assert diseases_out.equals(again[1])
