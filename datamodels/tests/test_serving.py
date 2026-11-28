"""Every view declares how it is served in ClickHouse, on its model, and the declaration names its columns."""

from pathlib import Path

import pytest

from ot_datamodels import Serving, views
from ot_datamodels.discovery import VIEWS_PACKAGE, discover_datasets
from ot_datamodels.spec import dataset_spec, walk

VIEWS = discover_datasets(VIEWS_PACKAGE)


def _columns(name: str) -> list[str]:
    return [f.name for f in dataset_spec(VIEWS[name]).fields]


def test_there_are_views():
    assert len(VIEWS) == 91


@pytest.mark.parametrize('name', sorted(VIEWS))
def test_every_view_declares_its_serving(name):
    assert isinstance(VIEWS[name].__dict__.get('clickhouse'), Serving)


@pytest.mark.parametrize('name', sorted(VIEWS))
def test_the_engine_has_its_keys(name):
    serving = VIEWS[name].clickhouse
    assert serving is not None
    if serving.engine == 'mergetree':
        assert serving.order_by
        assert set(serving.order_by) <= set(_columns(name))
    else:
        assert serving.order_by == ()
        assert len(dataset_spec(VIEWS[name]).primary_key) == 1


@pytest.mark.parametrize('name', sorted(VIEWS))
def test_dictionaries_name_existing_columns(name):
    serving = VIEWS[name].clickhouse
    assert serving is not None
    for d in serving.dictionaries:
        assert set(d.source_columns) <= {d.key, *d.attributes}
        assert {d.source_columns.get(c, c) for c in (d.key, *d.attributes)} <= set(_columns(name))


def test_the_served_dictionaries():
    found = {name: [d.name for d in m.clickhouse.dictionaries] for name, m in VIEWS.items() if m.clickhouse}
    assert {name: d for name, d in found.items() if d} == {
        'biosample_view': ['biosample_names'],
        'disease_view': ['disease_names'],
        'so_view': ['sequence_ontology'],
        'target_view': ['target_symbols'],
    }


@pytest.mark.parametrize('name', sorted(VIEWS))
def test_column_types_name_primitive_values(name):
    serving = VIEWS[name].clickhouse
    assert serving is not None
    primitives = {path for path, node in walk(dataset_spec(VIEWS[name])) if node.kind == 'primitive'}
    assert set(serving.column_types) <= primitives
    assert all(t.strip() for t in serving.column_types.values())


def test_no_output_is_served_from_its_model():
    assert [name for name, m in discover_datasets().items() if m.clickhouse is not None] == []


@pytest.mark.parametrize(
    ('fields', 'problem'),
    [
        ({'engine': 'mergetree'}, 'needs an order_by'),
        ({'engine': 'rocksdb', 'order_by': ('id',)}, 'takes no order_by'),
        ({'engine': 'log'}, 'unknown engine'),
    ],
)
def test_serving_refuses_a_wrong_key(fields, problem):
    with pytest.raises(ValueError, match=problem):
        Serving(**fields)


#: the outputs pos still serves from `datasets.yaml`, whose values keep their `ClickHouseType`; PR 3 removes it
CLICKHOUSE_TYPED_OUTPUTS = frozenset({
    'clinical_indication', 'clinical_target', 'evidence_cancer_biomarkers', 'evidence_cancer_gene_census',
    'evidence_clingen', 'evidence_clinical_precedence', 'evidence_crispr', 'evidence_crispr_screen', 'evidence_encore',
    'evidence_europepmc', 'evidence_eva', 'evidence_eva_somatic', 'evidence_expression_atlas',
    'evidence_gene2phenotype', 'evidence_gene_burden', 'evidence_genomics_england', 'evidence_gwas_credible_sets',
    'evidence_impc', 'evidence_intogen', 'evidence_orphanet', 'evidence_ot_crispr', 'evidence_ot_crispr_validation',
    'evidence_reactome', 'evidence_uniprot_literature', 'evidence_uniprot_variants',
})  # fmt: skip


def test_only_the_yaml_served_outputs_carry_clickhouse_types():
    """A served view declares its ClickHouse types in `Serving.column_types`."""
    typed = {
        name
        for name, model in discover_datasets().items()
        if any(node.clickhouse is not None for _, node in walk(dataset_spec(model)))
    }
    assert typed <= CLICKHOUSE_TYPED_OUTPUTS


def test_views_import_no_clickhouse_type():
    for path in sorted(Path(views.__file__).parent.glob('*.py')):
        assert 'ClickHouseType' not in path.read_text(), path.name
