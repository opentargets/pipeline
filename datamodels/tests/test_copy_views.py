"""The views that serve an output from ClickHouse: each is the output's columns, with the same types."""

import pytest

from ot_datamodels import DatasetModel
from ot_datamodels.discovery import VIEWS_PACKAGE, discover_datasets
from ot_datamodels.spec import dataset_spec, walk

#: view -> the output it serves
COPIES = {
    'baseline_expression_view': 'baseline_expression',
    'biosample_view': 'biosample',
    'clinical_report_view': 'clinical_report',
    'disease_hpo_view': 'disease_hpo',
    'enhancer_to_gene_view': 'enhancer_to_gene',
    'go_view': 'go',
    'interaction_evidence_view': 'interaction_evidence',
    'literature_entity_lut_view': 'literature_entity_lut',
    'literature_vector_view': 'literature_vector',
    'otar_view': 'otar',
    'pharmacogenomics_view': 'pharmacogenomics',
    'so_view': 'so',
    'target_essentiality_view': 'target_essentiality',
}

OUTPUTS = discover_datasets()
ALL_VIEWS = discover_datasets(VIEWS_PACKAGE)
COPY_VIEWS = {name: ALL_VIEWS[name] for name in COPIES}


def _nodes(model: type[DatasetModel]):
    return dict(walk(dataset_spec(model)))


def _columns(model: type[DatasetModel]) -> list[str]:
    return [f.name for f in dataset_spec(model).fields]


@pytest.mark.parametrize(('view', 'output'), sorted(COPIES.items()))
def test_a_copy_view_has_its_outputs_columns_with_their_types(view: str, output: str) -> None:
    model, source = COPY_VIEWS[view], OUTPUTS[output]
    assert set(_columns(model)) <= set(_columns(source))
    nodes, source_nodes = _nodes(model), _nodes(source)
    assert nodes == {path: source_nodes[path] for path in nodes}


@pytest.mark.parametrize(('view', 'output'), sorted(COPIES.items()))
def test_a_copy_view_is_one_column_projection_or_the_whole_output(view: str, output: str) -> None:
    model, source = COPY_VIEWS[view], OUTPUTS[output]
    if view == 'literature_entity_lut_view':
        assert not issubclass(model, source)
        assert model.products == source.products
    else:
        assert issubclass(model, source)
        assert _columns(model) == _columns(source)
        assert model.products == source.products


@pytest.mark.parametrize('view', sorted(COPIES))
def test_a_copy_view_is_named_after_its_output(view: str) -> None:
    model = COPY_VIEWS[view]
    assert model.dataset_name == view == f'{COPIES[view]}_view'
    assert model.__module__ == f'{VIEWS_PACKAGE}.{view}'
    model.model_validate(model.example)


def test_literature_entity_lut_view_columns_are_served_in_the_order_of_the_old_table() -> None:
    assert _columns(COPY_VIEWS['literature_entity_lut_view']) == [
        'pmid',
        'pmcid',
        'keywordId',
        'relevance',
        'date',
        'year',
        'month',
    ]


def test_literature_vector_view_copies_all_four_columns() -> None:
    assert _columns(COPY_VIEWS['literature_vector_view']) == ['category', 'word', 'norm', 'vector']


def test_otar_view_is_for_ppp_only() -> None:
    assert COPY_VIEWS['otar_view'].products == ('ppp',)


def test_the_served_dictionaries_read_the_source_columns_of_the_view() -> None:
    biosample = COPY_VIEWS['biosample_view'].clickhouse
    assert biosample is not None
    (names,) = biosample.dictionaries
    assert (names.name, names.key, names.attributes) == ('biosample_names', 'id', ('name',))
    assert names.source_columns == {'id': 'biosampleId', 'name': 'biosampleName'}
    so = COPY_VIEWS['so_view'].clickhouse
    assert so is not None
    assert [(d.name, d.key, d.attributes) for d in so.dictionaries] == [('sequence_ontology', 'id', ('label',))]
