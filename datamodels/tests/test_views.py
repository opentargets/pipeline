"""Tests for the view models: their shape against pts's schemas, and the columns derived from outputs."""

import dataclasses
from typing import get_args

import polars as pl
import pytest
from pts_view_schemas import ASSOCIATION_TIMESERIES, CODING_VARIANT, TARGET_VIEW

from ot_datamodels import DatasetModel, to_polars_schema
from ot_datamodels import vocabularies as v
from ot_datamodels.discovery import VIEWS_PACKAGE, discover_datasets, discover_models
from ot_datamodels.outputs.association_by_datasource_direct import AssociationByDatasourceDirect
from ot_datamodels.outputs.association_by_datasource_indirect import AssociationByDatasourceIndirect
from ot_datamodels.outputs.association_overall_direct import AssociationOverallDirect
from ot_datamodels.outputs.association_overall_indirect import AssociationOverallIndirect
from ot_datamodels.outputs.chemical_probe import ChemicalProbe
from ot_datamodels.outputs.disease import Disease
from ot_datamodels.outputs.drug_molecule import DrugMolecule
from ot_datamodels.outputs.evidence_eva import EvidenceEva
from ot_datamodels.outputs.homology import Homology
from ot_datamodels.outputs.interaction import Interaction
from ot_datamodels.outputs.interaction_evidence import InteractionEvidence
from ot_datamodels.outputs.pharmacogenomics import Pharmacogenomics
from ot_datamodels.outputs.study import Study
from ot_datamodels.outputs.target import Target
from ot_datamodels.outputs.target_safety_event import TargetSafetyEvent
from ot_datamodels.outputs.target_tractability import TargetTractability
from ot_datamodels.outputs.transcript import Transcript
from ot_datamodels.outputs.variant import Variant
from ot_datamodels.spec import Constraints, Node, dataset_spec, walk
from ot_datamodels.views.association_otf_disease import AssociationOtfDisease
from ot_datamodels.views.association_otf_target import AssociationOtfTarget
from ot_datamodels.views.association_timeseries import AssociationTimeseries
from ot_datamodels.views.coding_variant import CodingVariant
from ot_datamodels.views.disease_view import DiseaseView
from ot_datamodels.views.drug_view import DrugView
from ot_datamodels.views.interaction_view import InteractionView
from ot_datamodels.views.target_view import TargetView

VIEWS: dict[str, type[DatasetModel]] = {
    'association_otf_disease': AssociationOtfDisease,
    'association_otf_target': AssociationOtfTarget,
    'association_timeseries': AssociationTimeseries,
    'coding_variant': CodingVariant,
    'disease_view': DiseaseView,
    'drug_view': DrugView,
    'interaction_view': InteractionView,
    'target_view': TargetView,
}

#: the frozen pre-piece-3 pts schema of each view (`pts_view_schemas`)
PTS_SCHEMAS = {
    'association_timeseries': ASSOCIATION_TIMESERIES,
    'coding_variant': CODING_VARIANT,
    'target_view': TARGET_VIEW,
}

#: columns whose dtype differs from the frozen pre-piece-3 baseline: Enums the baseline holds as strings, and
#: the bool `homologues[].isHighConfidence` it holds as '1'/'0'
CHANGED: dict[str, set[str]] = {
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
    'association_otf_disease': ('rowId',),
    'association_otf_target': ('rowId',),
    'association_timeseries': (),
    'coding_variant': (),
    'disease_view': ('id',),
    'drug_view': ('id',),
    'interaction_view': ('targetA',),
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
    ('association_otf_disease', 'rowId'): (EvidenceEva, 'id', None),
    ('association_otf_disease', 'A'): (EvidenceEva, 'diseaseId', None),
    ('association_otf_disease', 'B'): (EvidenceEva, 'targetId', None),
    ('association_otf_disease', 'rowScore'): (EvidenceEva, 'score', None),
    ('association_otf_disease', 'noveltyDirect'): (AssociationOverallDirect, 'currentNovelty', True),
    ('association_otf_disease', 'noveltyIndirect'): (AssociationOverallIndirect, 'currentNovelty', True),
    ('association_otf_target', 'rowId'): (EvidenceEva, 'id', None),
    ('association_otf_target', 'A'): (EvidenceEva, 'targetId', None),
    ('association_otf_target', 'B'): (EvidenceEva, 'diseaseId', None),
    ('association_otf_target', 'rowScore'): (EvidenceEva, 'score', None),
    ('association_otf_target', 'noveltyDirect'): (AssociationOverallDirect, 'currentNovelty', True),
    ('association_otf_target', 'noveltyIndirect'): (AssociationOverallIndirect, 'currentNovelty', True),
    ('association_otf_target', 'indirect'): (Disease, 'ancestors', None),
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
    **{
        ('disease_view', c): (Disease, c, None)
        for c in (
            'id',
            'name',
            'therapeuticAreas',
            'description',
            'dbXRefs',
            'obsoleteTerms',
            'parents',
            'children',
            'ancestors',
            'descendants',
        )
    },
    ('disease_view', 'isTherapeuticArea'): (Disease, 'ontology/isTherapeuticArea', None),
    ('disease_view', 'studyIds[]'): (Study, 'studyId', None),
    ('disease_view', 'indirectStudyIds[]'): (Study, 'studyId', None),
    **{('drug_view', f.name): (DrugMolecule, f.name, None) for f in dataset_spec(DrugMolecule).fields},
    ('drug_view', 'pharmacogenomicsIds[]'): (Pharmacogenomics, 'pharmacogenomicsId', None),
    ('interaction_view', 'targetA'): (Interaction, 'targetA', None),
    **{
        ('interaction_view', f'interactions[]/{c}'): (Interaction, c, None)
        for c in ('targetA', 'intA', 'targetB', 'intB', 'scoring', 'count', 'sourceDatabase')
    },
    ('interaction_view', 'interactions[]/speciesA/taxonId'): (Interaction, 'speciesA/taxonId', None),
    **{
        ('interaction_view', f'interactions[]/evidences[]/{c}'): (InteractionEvidence, c, None)
        for c in (
            'evidenceScore',
            'expansionMethodMiIdentifier',
            'expansionMethodShortName',
            'hostOrganismScientificName',
            'interactionDetectionMethodMiIdentifier',
            'interactionDetectionMethodShortName',
            'interactionIdentifier',
            'interactionTypeMiIdentifier',
            'interactionTypeShortName',
            'participantDetectionMethodA',
            'participantDetectionMethodB',
            'pubmedId',
        )
    },
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
    """The views derive from one evidence model; every other has the same columns, each its own datasource."""
    nodes = _nodes(evidence)
    sides = ((AssociationOtfDisease, 'diseaseId', 'targetId'), (AssociationOtfTarget, 'targetId', 'diseaseId'))
    for model, a, b in sides:
        otf = _nodes(model)
        assert [otf[c] for c in ('rowId', 'A', 'B', 'rowScore')] == [nodes[c] for c in ('id', a, b, 'score')]
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
    ('interaction_view', 'interactions[]/speciesB/taxonId'): 'UInt32',
    ('interaction_view', 'interactions[]/evidences[]/hostOrganismTaxId'): 'UInt32',
}


#: every value a view declares rather than derives
DECLARED: dict[str, set[str]] = {
    'association_otf_disease': {'datasourceId', 'datatypeId', 'searchA', 'searchB', 'indirect', 'indirect[]'},
    'association_otf_target': {'datasourceId', 'datatypeId', 'searchA', 'searchB', 'isMeasurement'},
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
    'disease_view': {
        'synonyms',
        'synonyms[]',
        'synonyms[]/relation',
        'synonyms[]/terms',
        'synonyms[]/terms[]',
        'studyIds',
        'indirectStudyIds',
    },
    'drug_view': {'pharmacogenomicsIds'},
    'interaction_view': {
        'interactions',
        'interactions[]',
        'interactions[]/intABiologicalRole',
        'interactions[]/intBBiologicalRole',
        'interactions[]/speciesA',
        'interactions[]/speciesA/mnemonic',
        'interactions[]/speciesA/scientificName',
        'interactions[]/speciesB',
        'interactions[]/speciesB/mnemonic',
        'interactions[]/speciesB/scientificName',
        'interactions[]/speciesB/taxonId',
        'interactions[]/evidences',
        'interactions[]/evidences[]',
        'interactions[]/evidences[]/hostOrganismTaxId',
        'interactions[]/evidences[]/interactionResources',
        'interactions[]/evidences[]/interactionResources/databaseVersion',
        'interactions[]/evidences[]/interactionResources/sourceDatabase',
    },
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


def test_disease_view_is_the_served_disease_columns() -> None:
    assert [f.name for f in dataset_spec(DiseaseView).fields] == [
        'id',
        'name',
        'therapeuticAreas',
        'description',
        'dbXRefs',
        'obsoleteTerms',
        'synonyms',
        'parents',
        'children',
        'ancestors',
        'descendants',
        'isTherapeuticArea',
        'studyIds',
        'indirectStudyIds',
    ]
    assert discover_models()['disease_view'] is DiseaseView
    synonyms = _nodes(DiseaseView)
    assert synonyms['synonyms[]/relation'].dtype == pl.String()
    assert synonyms['synonyms[]/terms[]'].nullable


OTF_COMMON = ['rowId', 'A', 'B', 'datatypeId', 'datasourceId', 'rowScore', 'searchA', 'searchB']


def test_association_otf_views_are_the_served_columns() -> None:
    disease = [f.name for f in dataset_spec(AssociationOtfDisease).fields]
    target = [f.name for f in dataset_spec(AssociationOtfTarget).fields]
    assert disease == [*OTF_COMMON, 'noveltyDirect', 'noveltyIndirect', 'indirect']
    assert target == [*OTF_COMMON, 'isMeasurement', 'noveltyDirect', 'noveltyIndirect', 'indirect']
    models = discover_models()
    assert models['association_otf_disease'] is AssociationOtfDisease
    assert models['association_otf_target'] is AssociationOtfTarget


def test_association_otf_views_types() -> None:
    disease, target = _nodes(AssociationOtfDisease), _nodes(AssociationOtfTarget)
    assert disease['indirect'].dtype == pl.List(pl.String())
    assert target['indirect'].dtype == pl.List(pl.String())
    assert target['indirect[]'].clickhouse == 'LowCardinality(String)'
    assert disease['indirect[]'].clickhouse is None
    assert target['isMeasurement'].dtype == pl.Boolean()
    assert not target['isMeasurement'].nullable
    for nodes in (disease, target):
        assert nodes['searchA'].nullable
        assert nodes['searchB'].nullable


INTERACTION_ROW = [
    'targetA',
    'intA',
    'targetB',
    'intB',
    'intABiologicalRole',
    'intBBiologicalRole',
    'scoring',
    'count',
    'sourceDatabase',
    'speciesA',
    'speciesB',
    'evidences',
]
INTERACTION_EVIDENCE = [
    'evidenceScore',
    'expansionMethodMiIdentifier',
    'expansionMethodShortName',
    'hostOrganismScientificName',
    'hostOrganismTaxId',
    'interactionDetectionMethodMiIdentifier',
    'interactionDetectionMethodShortName',
    'interactionIdentifier',
    'interactionResources',
    'interactionTypeMiIdentifier',
    'interactionTypeShortName',
    'participantDetectionMethodA',
    'participantDetectionMethodB',
    'pubmedId',
]


def test_interaction_view_is_the_served_interaction_columns() -> None:
    nodes = _nodes(InteractionView)

    assert [f.name for f in dataset_spec(InteractionView).fields] == ['targetA', 'interactions']
    assert [f.name for f in nodes['interactions[]'].fields] == INTERACTION_ROW
    for species in ('speciesA', 'speciesB'):
        assert [f.name for f in nodes[f'interactions[]/{species}'].fields] == ['mnemonic', 'scientificName', 'taxonId']
    assert [f.name for f in nodes['interactions[]/evidences[]'].fields] == INTERACTION_EVIDENCE
    resources = nodes['interactions[]/evidences[]/interactionResources']
    assert [f.name for f in resources.fields] == ['databaseVersion', 'sourceDatabase']
    assert nodes['interactions'].constraints.min_length == 1
    assert nodes['interactions[]/evidences'].constraints.min_length == 1
    assert discover_models()['interaction_view'] is InteractionView


#: the values interaction_view declares in a served type, by the output value each serves
INTERACTION_SERVED_AS: dict[str, tuple[type[DatasetModel], str]] = {
    'interactions[]/intABiologicalRole': (Interaction, 'intABiologicalRole'),
    'interactions[]/intBBiologicalRole': (Interaction, 'intBBiologicalRole'),
    'interactions[]/speciesA/mnemonic': (Interaction, 'speciesA/mnemonic'),
    'interactions[]/speciesA/scientificName': (Interaction, 'speciesA/scientificName'),
    'interactions[]/speciesB/mnemonic': (Interaction, 'speciesB/mnemonic'),
    'interactions[]/speciesB/scientificName': (Interaction, 'speciesB/scientificName'),
    'interactions[]/speciesB/taxonId': (Interaction, 'speciesB/taxonId'),
    'interactions[]/evidences[]/hostOrganismTaxId': (InteractionEvidence, 'hostOrganismTaxId'),
    'interactions[]/evidences[]/interactionResources/databaseVersion': (
        InteractionEvidence,
        'interactionResources/databaseVersion',
    ),
    'interactions[]/evidences[]/interactionResources/sourceDatabase': (
        InteractionEvidence,
        'interactionResources/sourceDatabase',
    ),
}


@pytest.mark.parametrize('path', sorted(INTERACTION_SERVED_AS))
def test_interaction_view_declared_values_are_their_output_s_as_served(path: str) -> None:
    """The output value without LowCardinality and Enum (a String keeping the allowed values); taxon ids Nullable
    UInt32, IntAct's negative pseudo-taxids wrapped (David, 2026-11-24)."""
    output, output_path = INTERACTION_SERVED_AS[path]
    source = _nodes(output)[output_path]
    expected = dataclasses.replace(source, clickhouse=None)
    if isinstance(source.dtype, pl.Enum):
        expected = dataclasses.replace(expected, dtype=pl.String())
    if path.endswith(('/taxonId', '/hostOrganismTaxId')):
        expected = dataclasses.replace(
            expected, nullable=True, constraints=Constraints(ge=0, le=2**32 - 1), clickhouse='UInt32'
        )

    assert _nodes(InteractionView)[path] == expected


def test_interaction_view_source_database_stays_the_enum() -> None:
    nodes = _nodes(InteractionView)
    assert nodes['interactions[]/sourceDatabase'].dtype == pl.Enum(list(get_args(v.InteractionSource)))
    assert nodes['interactions[]/evidences[]/interactionResources/sourceDatabase'].dtype == pl.String()
