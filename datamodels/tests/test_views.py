"""Tests for the view models: their shape against pts's schemas, and the columns derived from outputs."""

import dataclasses
from typing import get_args

import polars as pl
import pytest
from pts_view_schemas import ASSOCIATION_OTF, ASSOCIATION_TIMESERIES, CODING_VARIANT, TARGET_VIEW

from ot_datamodels import DatasetModel, to_polars_schema
from ot_datamodels import vocabularies as v
from ot_datamodels.discovery import VIEWS_PACKAGE, discover_datasets, discover_models
from ot_datamodels.outputs.association_by_datasource_direct import AssociationByDatasourceDirect
from ot_datamodels.outputs.association_by_datasource_indirect import AssociationByDatasourceIndirect
from ot_datamodels.outputs.association_overall_direct import AssociationOverallDirect
from ot_datamodels.outputs.association_overall_indirect import AssociationOverallIndirect
from ot_datamodels.outputs.chemical_probe import ChemicalProbe
from ot_datamodels.outputs.drug_molecule import DrugMolecule
from ot_datamodels.outputs.evidence_eva import EvidenceEva
from ot_datamodels.outputs.homology import Homology
from ot_datamodels.outputs.pharmacogenomics import Pharmacogenomics
from ot_datamodels.outputs.target import Target
from ot_datamodels.outputs.target_safety_event import TargetSafetyEvent
from ot_datamodels.outputs.target_tractability import TargetTractability
from ot_datamodels.outputs.transcript import Transcript
from ot_datamodels.outputs.variant import Variant
from ot_datamodels.spec import Node, dataset_spec, walk
from ot_datamodels.views.association_otf import AssociationOtf
from ot_datamodels.views.association_timeseries import AssociationTimeseries
from ot_datamodels.views.coding_variant import CodingVariant
from ot_datamodels.views.drug_view import DrugView
from ot_datamodels.views.target_view import TargetView

VIEWS: dict[str, type[DatasetModel]] = {
    'association_otf': AssociationOtf,
    'association_timeseries': AssociationTimeseries,
    'coding_variant': CodingVariant,
    'drug_view': DrugView,
    'target_view': TargetView,
}

#: the frozen pre-piece-3 pts schema of each view (`pts_view_schemas`)
PTS_SCHEMAS = {
    'association_otf': ASSOCIATION_OTF,
    'association_timeseries': ASSOCIATION_TIMESERIES,
    'coding_variant': CODING_VARIANT,
    'target_view': TARGET_VIEW,
}

#: columns whose dtype differs from the frozen pre-piece-3 baseline: Enums the baseline holds as strings, and
#: the bool `homologues[].isHighConfidence` it holds as '1'/'0'
CHANGED: dict[str, set[str]] = {
    'association_otf': {'datasourceId', 'datatypeId'},
    'association_timeseries': {'aggregationType'},
    'coding_variant': {'datasources'},
    'target_view': {
        'genomicLocation',
        'go',
        'constraint',
        'homologues',
        'tractability',
        'safetyLiabilities',
        'canonicalTranscript',
    },
}

#: the primary key of each view
PRIMARY_KEYS = {
    'association_otf': ('rowId',),
    'association_timeseries': (),
    'coding_variant': (),
    'drug_view': ('id',),
    'target_view': ('id',),
}

ASSOCIATION_INPUTS = [
    AssociationOverallDirect,
    AssociationOverallIndirect,
    AssociationByDatasourceDirect,
    AssociationByDatasourceIndirect,
]
EVIDENCE = sorted(
    (m for n, m in discover_datasets().items() if n.startswith('evidence_')), key=lambda m: m.dataset_name
)


def _nodes(model: type[DatasetModel]) -> dict[str, Node]:
    return dict(walk(dataset_spec(model)))


#: every derived value of a view: (view, path) -> (output, path, nullability the view sets, or None)
DERIVED: dict[tuple[str, str], tuple[type[DatasetModel], str, bool | None]] = {
    ('association_otf', 'rowId'): (EvidenceEva, 'id', None),
    ('association_otf', 'diseaseId'): (EvidenceEva, 'diseaseId', None),
    ('association_otf', 'targetId'): (EvidenceEva, 'targetId', None),
    ('association_otf', 'rowScore'): (EvidenceEva, 'score', None),
    ('association_otf', 'noveltyDirect'): (AssociationOverallDirect, 'currentNovelty', True),
    ('association_otf', 'noveltyIndirect'): (AssociationOverallIndirect, 'currentNovelty', True),
    ('association_timeseries', 'diseaseId'): (AssociationOverallDirect, 'diseaseId', None),
    ('association_timeseries', 'targetId'): (AssociationOverallDirect, 'targetId', None),
    **{
        ('association_timeseries', c): (AssociationOverallDirect, f'timeseries[]/{c}', None)
        for c in ('year', 'associationScore', 'novelty', 'yearlyEvidenceCount')
    },
    ('coding_variant', 'targetId'): (Variant, 'transcriptConsequences[]/targetId', None),
    ('coding_variant', 'variantId'): (Variant, 'variantId', None),
    ('coding_variant', 'variantEffect'): (Variant, 'variantEffect[]/score', True),
    ('coding_variant', 'variantFunctionalConsequenceIds'): (
        Variant,
        'transcriptConsequences[]/variantFunctionalConsequenceIds',
        None,
    ),
    **{('drug_view', f.name): (DrugMolecule, f.name, None) for f in dataset_spec(DrugMolecule).fields},
    ('drug_view', 'pharmacogenomicsIds[]'): (Pharmacogenomics, 'pharmacogenomicsId', None),
    **{('target_view', f.name): (Target, f.name, None) for f in dataset_spec(Target).fields},
    **{
        ('target_view', f'homologues[]/{c}'): (Homology, c, None)
        for c in (
            'speciesId',
            'speciesName',
            'homologyType',
            'targetGeneId',
            'isHighConfidence',
            'targetGeneSymbol',
            'queryPercentageIdentity',
            'targetPercentageIdentity',
            'priority',
        )
    },
    ('target_view', 'tractability[]/modality'): (TargetTractability, 'modality', None),
    ('target_view', 'tractability[]/id'): (TargetTractability, 'category', None),
    ('target_view', 'tractability[]/value'): (TargetTractability, 'value', None),
    **{
        ('target_view', f'safetyLiabilities[]/{c}'): (TargetSafetyEvent, c, None)
        for c in ('event', 'eventId', 'effects', 'biosamples', 'datasource', 'literature', 'url', 'studies')
    },
    **{
        ('target_view', f'chemicalProbes[]/{c}'): (ChemicalProbe, c, None)
        for c in (
            'targetFromSourceId',
            'drugFromSourceId',
            'drugId',
            'mechanismOfAction',
            'origin',
            'control',
            'isHighQuality',
            'probesDrugsScore',
            'scoreInCells',
            'scoreInOrganisms',
            'urls',
        )
    },
    ('target_view', 'chemicalProbes[]/id'): (ChemicalProbe, 'drugFromSource', None),
    ('target_view', 'transcriptIds[]'): (Transcript, 'transcriptId', None),
    ('target_view', 'transcripts[]/transcriptId'): (Transcript, 'transcriptId', None),
    ('target_view', 'transcripts[]/biotype'): (Transcript, 'biotype', None),
    ('target_view', 'transcripts[]/translationId'): (Transcript, 'proteinId', None),
    ('target_view', 'transcripts[]/isEnsemblCanonical'): (Transcript, 'isEnsemblCanonical', None),
    ('target_view', 'canonicalTranscript/id'): (Transcript, 'transcriptId', None),
    **{
        ('target_view', f'canonicalTranscript/{c}'): (Transcript, c, None)
        for c in ('chromosome', 'start', 'end', 'strand')
    },
}


def _as_pts_writes(
    dtype: pl.DataType | pl.datatypes.DataTypeClass, path: str = ''
) -> pl.DataType | pl.datatypes.DataTypeClass:
    """`dtype` as the frozen pre-piece-3 baseline holds it: every Enum a string, `isHighConfidence` a string."""
    if isinstance(dtype, pl.Enum) or path.endswith('/isHighConfidence'):
        return pl.String()
    if isinstance(dtype, pl.List):
        return pl.List(_as_pts_writes(dtype.inner, f'{path}[]'))
    if isinstance(dtype, pl.Struct):
        return pl.Struct([pl.Field(f.name, _as_pts_writes(f.dtype, f'{path}/{f.name}')) for f in dtype.fields])
    return dtype


@pytest.mark.parametrize('name', sorted(PTS_SCHEMAS))
def test_view_matches_the_pre_piece_3_pts_schema(name: str) -> None:
    schema, pts = to_polars_schema(VIEWS[name]), PTS_SCHEMAS[name]

    assert list(schema) == list(pts)
    assert {c for c in schema if schema[c] != pts[c]} == CHANGED[name]
    assert all(_as_pts_writes(schema[c], c) == pts[c] for c in CHANGED[name])


@pytest.mark.parametrize('name', sorted(VIEWS))
def test_primary_key(name: str) -> None:
    assert dataset_spec(VIEWS[name]).primary_key == PRIMARY_KEYS[name]


@pytest.mark.parametrize('name', sorted(VIEWS))
def test_dataset_name_and_example(name: str) -> None:
    model = VIEWS[name]
    assert model.dataset_name == name
    assert model.__module__ == f'{VIEWS_PACKAGE}.{name}'
    model.model_validate(model.example)


@pytest.mark.parametrize(('view', 'path'), sorted(DERIVED), ids=[f'{v}:{p}' for v, p in sorted(DERIVED)])
def test_derived_value_is_its_output_s(view: str, path: str) -> None:
    output, output_path, nullable = DERIVED[view, path]
    source = _nodes(output)[output_path]
    expected = source if nullable is None else dataclasses.replace(source, nullable=nullable)

    assert _nodes(VIEWS[view])[path] == expected


def test_target_view_extends_target() -> None:
    assert issubclass(TargetView, Target)
    target = dataset_spec(Target).fields
    assert dataset_spec(TargetView).fields[: len(target)] == target


def test_homologue_high_confidence_is_homology_s_bool() -> None:
    assert _nodes(TargetView)['homologues[]/isHighConfidence'].dtype == pl.Boolean()


@pytest.mark.parametrize('evidence', EVIDENCE, ids=lambda m: m.dataset_name)
def test_otf_columns_are_every_evidence_s(evidence: type[DatasetModel]) -> None:
    """The view derives from one evidence model; every other has the same columns, each its own datasource."""
    otf, nodes = _nodes(AssociationOtf), _nodes(evidence)
    assert [otf[c] for c in ('rowId', 'diseaseId', 'targetId', 'rowScore')] == [
        nodes[c] for c in ('id', 'diseaseId', 'targetId', 'score')
    ]
    assert otf['datasourceId'].dtype == nodes['datasourceId'].dtype
    assert otf['datatypeId'].dtype == nodes['datatypeId'].dtype


@pytest.mark.parametrize('association', ASSOCIATION_INPUTS, ids=lambda m: m.dataset_name)
def test_timeseries_columns_are_every_input_s(association: type[DatasetModel]) -> None:
    """The view derives from one association; the others agree but for by-datasource's Int32 yearlyEvidenceCount."""
    view, nodes = _nodes(AssociationTimeseries), _nodes(association)
    assert [view[c] for c in ('diseaseId', 'targetId')] == [nodes[c] for c in ('diseaseId', 'targetId')]
    for c in ('year', 'associationScore', 'novelty'):
        assert view[c] == nodes[f'timeseries[]/{c}'], c
    count = nodes['timeseries[]/yearlyEvidenceCount']
    if count.dtype == pl.Int32():
        # Int32 caps it below UInt32's max, so it needs no `le`
        count = dataclasses.replace(count, dtype=pl.Int64(), constraints=view['yearlyEvidenceCount'].constraints)
    assert view['yearlyEvidenceCount'] == count
    assert view['aggregationType'].dtype == nodes['aggregationType'].dtype
    assert set(nodes['aggregationType'].constraints.allowed or ()) <= set(
        view['aggregationType'].constraints.allowed or ()
    )


def test_association_timeseries_is_ppp_only() -> None:
    assert AssociationTimeseries.products == ('ppp',)
    nodes = _nodes(AssociationTimeseries)
    assert nodes['aggregationValue'].dtype == pl.String()
    assert nodes['partitionKey'].dtype == pl.String()


def test_coding_variant_datasource_is_its_vocabulary() -> None:
    node = _nodes(CodingVariant)['datasources[]/datasourceId']
    assert node.dtype == pl.Enum(list(get_args(v.CodingVariantDatasource)))


def test_coding_variant_datasource_holds_every_source_the_view_reads() -> None:
    datasets = discover_datasets()
    evidence = ('evidence_eva', 'evidence_eva_somatic', 'evidence_uniprot_variants', 'evidence_gwas_credible_sets')
    sources = {value for name in evidence for value in _nodes(datasets[name])['datasourceId'].constraints.allowed or ()}
    sources |= {*get_args(v.PharmacogenomicsDatasource), 'mol_qtl'}

    assert sources <= set(get_args(v.CodingVariantDatasource))


#: the ClickHouseType markers declared on view columns, from the hand-written landing SQL before piece 3
MARKERS = {
    ('coding_variant', 'datasources[]/datasourceCount'): 'UInt32',
    ('coding_variant', 'datasources[]/datasourceNiceName'): 'LowCardinality(String)',
    ('coding_variant', 'referenceAminoAcid'): 'LowCardinality(String)',
    ('coding_variant', 'alternateAminoAcid'): 'LowCardinality(String)',
}


#: every value a view declares rather than derives
DECLARED: dict[str, set[str]] = {
    'association_otf': {'datasourceId', 'datatypeId', 'diseaseData', 'targetData'},
    'association_timeseries': {'aggregationType', 'aggregationValue', 'isDirect', 'partitionKey'},
    'coding_variant': {
        'therapeuticAreas',
        'therapeuticAreas[]',
        'diseases',
        'diseases[]',
        'datasources',
        'datasources[]',
        'datasources[]/datasourceCount',
        'datasources[]/datasourceId',
        'datasources[]/datasourceNiceName',
        'uniprotAccessions',
        'uniprotAccessions[]',
        'referenceAminoAcid',
        'aminoAcidPosition',
        'alternateAminoAcid',
    },
    'drug_view': {'pharmacogenomicsIds'},
    'target_view': {
        'homologues',
        'homologues[]',
        'tractability',
        'tractability[]',
        'safetyLiabilities',
        'safetyLiabilities[]',
        'chemicalProbes',
        'chemicalProbes[]',
        'transcriptIds',
        'transcripts',
        'transcripts[]',
        'transcripts[]/uniprotId',
        'transcripts[]/isUniprotReviewed',
        'transcripts[]/alphafoldId',
        'transcripts[]/uniprotIsoformId',
        'canonicalTranscript',
        'canonicalExons',
        'canonicalExons[]',
        'alternativeGenes',
        'alternativeGenes[]',
    },
}


def _under_derived(name: str, path: str) -> bool:
    return any(path.startswith((f'{p}/', f'{p}[]')) for n, p in DERIVED if n == name)


@pytest.mark.parametrize('name', sorted(VIEWS))
def test_every_view_value_is_derived_or_declared(name: str) -> None:
    derived = {p for n, p in DERIVED if n == name}
    paths = {p for p, _ in walk(dataset_spec(VIEWS[name])) if not _under_derived(name, p)}

    assert not derived & DECLARED[name]
    assert paths == derived | DECLARED[name]


def test_declared_markers() -> None:
    declared = {
        (name, path): node.clickhouse
        for name, model in VIEWS.items()
        for path, node in walk(dataset_spec(model))
        if node.clickhouse is not None
        and not any(path == p or path.startswith((f'{p}/', f'{p}[]')) for n, p in DERIVED if n == name)
    }
    assert declared == MARKERS


def test_discover_models_is_the_outputs_and_the_views() -> None:
    outputs, models = discover_datasets(), discover_models()

    assert not set(outputs) & set(VIEWS)
    assert models == outputs | VIEWS


def test_pharmacogenomics_id_leads_and_keys_the_output() -> None:
    spec = dataset_spec(Pharmacogenomics)
    assert spec.fields[0].name == 'pharmacogenomicsId'
    assert spec.primary_key == ('pharmacogenomicsId',)
    Pharmacogenomics.model_validate(Pharmacogenomics.example)


def test_drug_view_is_drug_molecule_plus_the_ids() -> None:
    drug = [f.name for f in dataset_spec(DrugMolecule).fields]
    assert [f.name for f in dataset_spec(DrugView).fields] == [*drug, 'pharmacogenomicsIds']
    assert discover_models()['drug_view'] is DrugView
