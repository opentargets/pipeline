"""The per-source evidence views and the clinical splits: names, products, columns, nullability and serving."""

import dataclasses

import pytest

from ot_datamodels import DatasetModel
from ot_datamodels.discovery import VIEWS_PACKAGE, discover_datasets
from ot_datamodels.spec import dataset_spec, walk
from ot_datamodels.views import evidence_views
from ot_datamodels.views.clinical_indication_by_disease_view import ClinicalIndicationByDiseaseView
from ot_datamodels.views.clinical_indication_by_drug_view import ClinicalIndicationByDrugView
from ot_datamodels.views.clinical_target_by_target_view import ClinicalTargetByTargetView
from ot_datamodels.views.evidence_views import EVIDENCE_VIEWS

OUTPUTS = discover_datasets()
EVIDENCE = {name: model for name, model in OUTPUTS.items() if name.startswith('evidence_')}
BY_NAME = {m.dataset_name: m for m in EVIDENCE_VIEWS}

#: the clinical splits: view -> (output, columns, the column its rows are non-null in, sort key)
SPLITS = {
    'clinical_indication_by_drug_view': (
        'clinical_indication',
        ['id', 'drugId', 'diseaseId', 'maxClinicalStage', 'clinicalReportIds'],
        'drugId',
        ('drugId', 'id'),
    ),
    'clinical_indication_by_disease_view': (
        'clinical_indication',
        ['id', 'drugId', 'diseaseId', 'maxClinicalStage', 'clinicalReportIds'],
        'diseaseId',
        ('diseaseId', 'id'),
    ),
    'clinical_target_by_target_view': (
        'clinical_target',
        ['id', 'drugId', 'targetId', 'diseases', 'maxClinicalStage', 'clinicalReportIds'],
        'targetId',
        ('targetId', 'id'),
    ),
}
SPLIT_VIEWS: dict[str, type[DatasetModel]] = {
    'clinical_indication_by_drug_view': ClinicalIndicationByDrugView,
    'clinical_indication_by_disease_view': ClinicalIndicationByDiseaseView,
    'clinical_target_by_target_view': ClinicalTargetByTargetView,
}

SOURCES = (
    'cancer_biomarkers', 'cancer_gene_census', 'clingen', 'clinical_precedence', 'crispr', 'crispr_screen', 'encore',
    'europepmc', 'eva', 'eva_somatic', 'expression_atlas', 'gene2phenotype', 'gene_burden', 'genomics_england',
    'gwas_credible_sets', 'impc', 'intogen', 'orphanet', 'ot_crispr', 'ot_crispr_validation', 'reactome',
    'uniprot_literature', 'uniprot_variants',
)  # fmt: skip
VARIANT_SOURCES = ('eva', 'eva_somatic', 'uniprot_variants')
PPP_ONLY = ('encore', 'ot_crispr', 'ot_crispr_validation')

INDEX_COLUMNS = {
    'by_disease_and_target': ['datasourceId', 'diseaseId', 'targetId', 'score', 'id'],
    'by_variant': ['datasourceId', 'variantId', 'score', 'id'],
}
INDEX_ORDER = {'by_disease_and_target': ('diseaseId', 'targetId', 'score'), 'by_variant': ('variantId', 'score')}
INDEX_KEYS = {'by_disease_and_target': {'diseaseId', 'targetId'}, 'by_variant': {'variantId'}}


def _columns(model: type[DatasetModel]) -> list[str]:
    return [f.name for f in dataset_spec(model).fields]


def _nodes(model: type[DatasetModel]):
    return dict(walk(dataset_spec(model)))


def _index(name: str) -> tuple[str, str]:
    """(output, index suffix) of an index view."""
    for suffix in INDEX_COLUMNS:
        if name.endswith(f'_{suffix}_view'):
            return name.removesuffix(f'_{suffix}_view'), suffix
    raise AssertionError(name)


def test_the_generated_view_names():
    expected = [
        *(f'evidence_{s}_view' for s in SOURCES),
        *(f'evidence_{s}_by_disease_and_target_view' for s in SOURCES),
        *(f'evidence_{s}_by_variant_view' for s in VARIANT_SOURCES),
    ]
    assert len(expected) == 49
    assert sorted(BY_NAME) == sorted(expected)
    assert sorted(EVIDENCE) == [f'evidence_{s}' for s in SOURCES]


def test_by_variant_exists_for_each_output_with_a_variant_column():
    assert sorted(n for n, m in EVIDENCE.items() if 'variantId' in m.model_fields) == [
        f'evidence_{s}' for s in VARIANT_SOURCES
    ]


def test_every_generated_view_is_discovered_under_its_own_class_name():
    views = discover_datasets(VIEWS_PACKAGE)
    for model in EVIDENCE_VIEWS:
        assert views[model.dataset_name] is model
        assert getattr(evidence_views, model.__name__) is model
        assert model.__module__ == f'{VIEWS_PACKAGE}.evidence_views'
    assert len({m.__name__ for m in EVIDENCE_VIEWS}) == len(EVIDENCE_VIEWS)
    assert BY_NAME['evidence_eva_by_disease_and_target_view'].__name__ == 'EvidenceEvaByDiseaseAndTargetView'


@pytest.mark.parametrize('name', sorted(BY_NAME))
def test_a_view_loads_for_its_outputs_products(name):
    output = name.removesuffix('_view') if name in {f'evidence_{s}_view' for s in SOURCES} else _index(name)[0]
    assert BY_NAME[name].products == EVIDENCE[output].products


def test_the_ppp_only_views():
    assert sorted(n for n, m in BY_NAME.items() if m.products == ('ppp',)) == sorted(
        f'evidence_{s}_{kind}view' for s in PPP_ONLY for kind in ('', 'by_disease_and_target_')
    )
    assert all(m.products == ('platform', 'ppp') for n, m in BY_NAME.items() if not n.startswith(PPP_ONLY, 9))


@pytest.mark.parametrize('source', SOURCES)
def test_a_whole_copy_is_its_output(source):
    model, output = BY_NAME[f'evidence_{source}_view'], EVIDENCE[f'evidence_{source}']
    assert issubclass(model, output)
    assert _nodes(model) == _nodes(output)
    assert dataset_spec(model).primary_key == ('id',)
    assert model.clickhouse is not None
    assert (model.clickhouse.engine, model.clickhouse.order_by) == ('rocksdb', ())
    model.model_validate(model.example)


@pytest.mark.parametrize('source', SOURCES)
def test_a_whole_copy_serves_its_outputs_clickhouse_types(source):
    model, output = BY_NAME[f'evidence_{source}_view'], EVIDENCE[f'evidence_{source}']
    assert model.clickhouse is not None
    assert dict(model.clickhouse.column_types) == {p: n.clickhouse for p, n in _nodes(output).items() if n.clickhouse}


INDEX_VIEWS = sorted(n for n in BY_NAME if n.endswith(('_by_disease_and_target_view', '_by_variant_view')))


@pytest.mark.parametrize('name', INDEX_VIEWS)
def test_an_index_view_is_its_columns_with_non_null_keys(name):
    output_name, suffix = _index(name)
    model, output = BY_NAME[name], EVIDENCE[output_name]
    assert _columns(model) == INDEX_COLUMNS[suffix]
    nodes, source = _nodes(model), _nodes(output)
    for column in INDEX_COLUMNS[suffix]:
        expected = source[column]
        if column in INDEX_KEYS[suffix]:
            expected = dataclasses.replace(expected, nullable=False)
        assert nodes[column] == expected, column
        assert not nodes[column].nullable
    assert dataset_spec(model).primary_key == ('id',)
    serving = model.clickhouse
    assert serving is not None
    assert (serving.engine, serving.order_by) == ('mergetree', INDEX_ORDER[suffix])
    whole = BY_NAME[f'{output_name}_view'].clickhouse
    assert whole is not None
    assert dict(serving.column_types) == {p: t for p, t in whole.column_types.items() if p in INDEX_COLUMNS[suffix]}
    model.model_validate(model.example)


@pytest.mark.parametrize('name', sorted(SPLITS))
def test_a_clinical_split_is_its_columns_with_a_non_null_key(name):
    output_name, columns, key, order_by = SPLITS[name]
    model, output = SPLIT_VIEWS[name], OUTPUTS[output_name]
    assert model.dataset_name == name
    assert _columns(model) == columns
    nodes, source = _nodes(model), _nodes(output)
    assert {p: n for p, n in nodes.items() if p != key} == {p: source[p] for p in nodes if p != key}
    assert nodes[key] == dataclasses.replace(source[key], nullable=False)
    assert dataset_spec(model).primary_key == ('id',)
    assert model.products == output.products
    serving = model.clickhouse
    assert serving is not None
    assert (serving.engine, serving.order_by) == ('mergetree', order_by)
    assert dict(serving.column_types) == {p: n.clickhouse for p, n in source.items() if n.clickhouse and p in nodes}
    model.model_validate(model.example)
