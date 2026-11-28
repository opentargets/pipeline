"""pos.clickhouse_ddl: ClickHouse landing DDL generated from the dataset models."""

import datetime
import re
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal, get_args

import polars as pl
import pytest
from ot_datamodels import ClickHouseDictionary, Serving
from ot_datamodels.base import ClickHouseType, DatasetModel, PolarsDtype, PrimaryKey, StructModel
from ot_datamodels.discovery import VIEWS_PACKAGE, discover_datasets
from ot_datamodels.outputs.study import Study
from ot_datamodels.outputs.target import Target
from ot_datamodels.outputs.variant import Variant
from ot_datamodels.spec import dataset_spec
from ot_datamodels.views.association_otf_disease_view import AssociationOtfDiseaseView
from ot_datamodels.views.association_otf_target_view import AssociationOtfTargetView
from ot_datamodels.views.coding_variant_by_target_view import CodingVariantByTargetView
from ot_datamodels.views.coding_variant_by_variant_view import CodingVariantByVariantView
from ot_datamodels.views.colocalisation_by_credible_set_view import ColocalisationByCredibleSetView
from ot_datamodels.views.credible_set_locus_view import CredibleSetLocusView
from ot_datamodels.views.credible_set_view import CredibleSetView
from ot_datamodels.views.disease_phenotype_by_disease_view import DiseasePhenotypeByDiseaseView
from ot_datamodels.views.disease_view import DiseaseView
from ot_datamodels.views.drug_mechanism_of_action_by_drug_view import DrugMechanismOfActionByDrugView
from ot_datamodels.views.drug_warning_by_drug_view import DrugWarningByDrugView
from ot_datamodels.views.interaction_view import InteractionView
from ot_datamodels.views.l2g_prediction_by_credible_set_view import L2gPredictionByCredibleSetView
from ot_datamodels.views.mouse_phenotype_by_target_view import MousePhenotypeByTargetView
from ot_datamodels.views.pharmacovigilance_by_drug_view import PharmacovigilanceByDrugView
from ot_datamodels.views.study_credible_set_view import StudyCredibleSetView
from ot_datamodels.views.study_view import StudyView
from ot_datamodels.views.target_by_region_view import TargetByRegionView
from ot_datamodels.views.target_credible_set_view import TargetCredibleSetView
from ot_datamodels.views.target_prioritisation_view import TargetPrioritisationView
from ot_datamodels.views.target_view import TargetView
from ot_datamodels.views.transcript_view import TranscriptView
from ot_datamodels.views.variant_credible_set_view import VariantCredibleSetView
from ot_datamodels.views.variant_view import VariantView
from ot_datamodels.vocabularies import (
    ChemblActionType,
    ChemblTargetType,
    ChemblWarningType,
    Chromosome,
    CredibleSetQualityControl,
    Datasource,
    Datatype,
    FineMappingMethod,
    HpoAspect,
    HpoEvidenceCode,
    HpoResource,
    HpoSex,
    InteractionSource,
    MechanismCuration,
    StudyType,
    TranscriptBiotype,
    VariantQualityControl,
)

from pos import clickhouse_ddl
from pos.clickhouse_ddl import (
    CHType,
    column_type,
    columns,
    create_table,
    dictionary_sql,
    drop_dictionary_sql,
    input_dir,
    landing_ddl,
    landing_table,
    model_type,
    parse_type,
    render,
    rocksdb_key,
    served_sql,
    served_views,
    table_sql,
    uniqueness_sql,
)

ROOT = Path(__file__).parents[1]


class Foo(StructModel):
    a: str
    b: int


class Bar(StructModel):
    a: str


class Toy(DatasetModel):
    dataset_name = 'toy'
    dataset_title = 'Toy'
    dataset_description = 'Toy.'
    example = {}

    marked: Annotated[int, ClickHouseType('UInt32')]
    marked_nullable: Annotated[int | None, ClickHouseType('UInt32')]
    literal: Literal['a', 'b']
    enum: Annotated[Literal['x', 'y'], PolarsDtype(pl.Enum)]
    flag: bool
    day: datetime.date
    score: float | None
    tags: list[str]
    maybe_tags: list[str | None]
    nested: list[list[int]]
    struct: Foo
    nullable_struct: Foo | None
    low_cardinality: Annotated[str | None, ClickHouseType('LowCardinality(String)')]
    bars: list[Bar | None]


def _rendered(name: str) -> str:
    node = next(f.node for f in dataset_spec(Toy).fields if f.name == name)
    return render(column_type(node))


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('marked', 'UInt32'),
        ('marked_nullable', 'Nullable(UInt32)'),
        ('literal', 'String'),
        ('enum', "Enum8('x' = 1, 'y' = 2)"),
        ('flag', 'Bool'),
        ('day', 'Date'),
        ('score', 'Nullable(Float64)'),
        ('tags', 'Array(String)'),
        ('maybe_tags', 'Array(Nullable(String))'),
        ('nested', 'Array(Array(Int64))'),
        ('struct', 'Tuple(`a` String, `b` Int64)'),
        ('nullable_struct', 'Tuple(`a` Nullable(String), `b` Nullable(Int64))'),
    ],
)
def test_column_type(name, expected):
    assert _rendered(name) == expected


def test_low_cardinality_nullable_order():
    assert _rendered('low_cardinality') == 'LowCardinality(Nullable(String))'


def test_list_of_nullable_structs():
    assert _rendered('bars') == 'Array(Tuple(`a` Nullable(String)))'


def test_columns_in_model_order():
    assert [name for name, _ in columns(Toy)] == [f.name for f in dataset_spec(Toy).fields]


class MemberA(DatasetModel):
    dataset_name = 'member_a'
    dataset_title = 'A'
    dataset_description = 'A.'
    example = {}

    id: str
    s: Bar
    l: list[str]  # noqa: E741
    only_a: int


class Quoted(DatasetModel):
    dataset_name = 'quoted'
    dataset_title = 'Q'
    dataset_description = 'Q.'
    example = {}

    label: Annotated[Literal["it's", 'back\\slash'], PolarsDtype(pl.Enum)]


def test_enum_labels_round_trip():
    t = columns(Quoted)[0][1]
    assert render(t) == r"Enum8('it\'s' = 1, 'back\\slash' = 2)"
    assert parse_type(render(t)) == t
    assert model_type(dataset_spec(Quoted).fields[0].node) == t


BigLiteral = Literal[tuple(f'v{i}' for i in range(128))]  # ty: ignore[invalid-type-form]


class BigEnum(StructModel):
    a: Annotated[BigLiteral, PolarsDtype(pl.Enum)]


class BigEnumDataset(DatasetModel):
    dataset_name = 'big_enum'
    dataset_title = 'B'
    dataset_description = 'B.'
    example = {}

    s: BigEnum


def test_enum_too_large_names_the_path():
    with pytest.raises(ValueError, match=r'^s/a: 128 Enum categories'):
        columns(BigEnumDataset)


def test_pos_ddl_prints_one_dataset(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr('sys.argv', ['pos-ddl', 'disease_view'])
    clickhouse_ddl.main()
    out = capsys.readouterr().out
    assert out.startswith('CREATE TABLE IF NOT EXISTS `disease_view_log` (')


def test_pos_ddl_all_prints_every_served_view_of_the_product(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr('sys.argv', ['pos-ddl', '--all'])
    clickhouse_ddl.main()
    platform = capsys.readouterr().out
    monkeypatch.setattr('sys.argv', ['pos-ddl', '--all', '--product', 'ppp'])
    clickhouse_ddl.main()
    ppp = capsys.readouterr().out
    assert platform.splitlines() == [f'{landing_ddl(m)};' for m in served_views('platform').values()]
    assert ppp.splitlines() == [f'{landing_ddl(m)};' for m in served_views('ppp').values()]
    assert '`evidence_eva_view_log`' in platform
    assert '`evidence_encore_view_log`' not in platform
    assert '`evidence_encore_view_log`' in ppp


def test_create_table_quotes_identifiers():
    sql = create_table('toy_log', [('id', column_type(dataset_spec(MemberA).fields[0].node))])
    assert sql == 'CREATE TABLE IF NOT EXISTS `toy_log` (`id` String) ENGINE = Log'


def test_every_view_is_served():
    assert set(served_views()) == set(discover_datasets(VIEWS_PACKAGE))


def test_a_served_view_lands_from_its_folder_and_builds_its_table_and_dictionaries():
    assert (landing_table(DiseaseView), input_dir(DiseaseView)) == ('disease_view_log', 'view/disease_view')
    statements = served_sql(DiseaseView)
    assert statements[0] == 'DROP DICTIONARY IF EXISTS `disease_names` SYNC'
    assert statements[1:4] == table_sql(DiseaseView)
    assert statements[2].endswith('ENGINE = EmbeddedRocksDB() PRIMARY KEY `id`')
    assert statements[4].startswith('CREATE OR REPLACE DICTIONARY `disease_names` (`id` String, `name` String)')
    assert statements[4].endswith("SOURCE(CLICKHOUSE(TABLE 'disease_view')) LAYOUT(COMPLEX_KEY_HASHED()) LIFETIME(0)")
    assert statements[5:] == ['DROP TABLE `disease_view_log` SYNC']
    by_region = table_sql(TargetByRegionView)
    assert by_region[1].endswith('ENGINE = MergeTree() ORDER BY (`chromosome`, `start`, `end`)')
    assert by_region[3] == 'OPTIMIZE TABLE `target_by_region_view` FINAL'


class Typed(DatasetModel):
    dataset_name = 'typed_view'
    dataset_title = 'Typed'
    dataset_description = 'Typed.'
    example = {}
    clickhouse = Serving(
        engine='mergetree',
        order_by=('count',),
        column_types={
            'count': 'UInt32',
            'names[]': 'LowCardinality(String)',
            'nested/b': 'UInt8',
            'bars[]/a': 'LowCardinality(String)',
        },
    )

    count: int
    names: list[str]
    nested: Foo | None
    bars: list[Bar | None]
    marked: Annotated[int, ClickHouseType('UInt32')]


def test_column_types_set_the_view_s_primitive_types():
    assert [(name, render(t)) for name, t in columns(Typed)] == [
        ('count', 'UInt32'),
        ('names', 'Array(LowCardinality(String))'),
        ('nested', 'Tuple(`a` Nullable(String), `b` Nullable(UInt8))'),
        ('bars', 'Array(Tuple(`a` LowCardinality(Nullable(String))))'),
        ('marked', 'UInt32'),
    ]


@pytest.mark.parametrize('path', ['missing', 'names', 'nested', 'bars[]', 'nested/c'])
def test_a_column_types_path_that_is_no_primitive_fails(monkeypatch, path):
    monkeypatch.setattr(Typed, 'clickhouse', Serving(engine='rocksdb', column_types={path: 'UInt8'}))
    with pytest.raises(ValueError, match=rf"typed_view: column_types paths \['{re.escape(path)}'\] are not primitive"):
        columns(Typed)


def test_an_output_declaring_clickhouse_fails_the_served_views(monkeypatch):
    served = SimpleNamespace(clickhouse=DiseaseView.clickhouse)
    monkeypatch.setattr(clickhouse_ddl, 'discover_datasets', lambda: {'disease': served, 'target': Target})
    with pytest.raises(ValueError, match=r"output models \['disease'\] declare clickhouse"):
        served_views()


def test_served_views_follow_the_product():
    assert set(served_views()) - set(served_views('platform')) == {
        'association_timeseries_view',
        'otar_view',
        *(f'evidence_{s}_{kind}view' for s in ('encore', 'ot_crispr', 'ot_crispr_validation')
          for kind in ('', 'by_disease_and_target_')),
    }  # fmt: skip
    assert set(served_views('ppp')) == set(served_views())


def _type_at(cols: Sequence[tuple[str, CHType]], path: str) -> CHType | None:
    """The type at `a`, `a/b` or `a[]/b` (`a[]` is the element); a Tuple of `cols` for ''; None if absent."""
    t: CHType | None = CHType('Tuple', fields=tuple(cols))
    for part in path.split('/') if path else []:
        name, lists = part, 0
        while name.endswith('[]'):
            name, lists = name[:-2], lists + 1
        t = dict(t.fields).get(name) if t is not None else None
        for _ in range(lists):
            t = t.element if t is not None else None
    return t


#: The `targets` landing types the markers on the `target` model keep.
TARGET_VIEW_TYPES = {
    'genomicLocation/start': 'UInt32',
    'genomicLocation/end': 'UInt32',
    'genomicLocation/strand': "Enum8('-1' = -1, '0' = 0, '1' = 1)",
    'constraint[]/exp': 'Nullable(Float64)',
    'constraint[]/oe': 'Nullable(Float64)',
    'constraint[]/oeLower': 'Nullable(Float64)',
    'constraint[]/oeUpper': 'Nullable(Float64)',
    'constraint[]/score': 'Nullable(Float64)',
    'constraint[]/obs': 'Nullable(UInt32)',
    'constraint[]/upperBin': 'Nullable(UInt32)',
    'constraint[]/upperBin6': 'Nullable(UInt32)',
    'constraint[]/upperRank': 'Nullable(UInt32)',
    'hallmarks/attributes[]/pmid': 'Nullable(UInt32)',
    'hallmarks/cancerHallmarks[]/pmid': 'UInt32',
    'targetClass[]/id': 'UInt32',
}


@pytest.mark.parametrize(('path', 'expected'), TARGET_VIEW_TYPES.items())
def test_target_view_keeps_landing_types(path, expected):
    t = _type_at(columns(TargetView), path)
    assert t is not None
    assert render(t) == expected


def test_disease_view_landing_is_the_served_disease():
    assert ', '.join(f'{n} {render(t, quote_fields=False)}' for n, t in columns(DiseaseView)) == (
        'id String, name String, therapeuticAreas Array(LowCardinality(String)), description Nullable(String), '
        'dbXRefs Array(String), obsoleteTerms Array(String), '
        'synonyms Array(Tuple(relation String, terms Array(Nullable(String)))), '
        'parents Array(LowCardinality(String)), children Array(String), ancestors Array(LowCardinality(String)), '
        'descendants Array(String), isTherapeuticArea Bool, studyIds Array(String), indirectStudyIds Array(String)'
    )


def _enum8(vocabulary) -> str:
    """The Enum8 pos generates for a vocabulary: its values numbered from 1 in order."""
    return f'Enum8({", ".join(f"'{v}' = {i}" for i, v in enumerate(get_args(vocabulary), 1))})'


#: the columns of the served associations_otf_* tables as aotf.sql built them (read off system.columns), but
#: for target isMeasurement, UInt8 there
AOTF_SERVED = {
    'rowId': 'String',
    'A': 'LowCardinality(String)',
    'B': 'LowCardinality(String)',
    'datatypeId': _enum8(Datatype),
    'datasourceId': _enum8(Datasource),
    'rowScore': 'Float64',
    'searchA': 'Nullable(String)',
    'searchB': 'Nullable(String)',
    'noveltyDirect': 'Nullable(Float64)',
    'noveltyIndirect': 'Nullable(Float64)',
}


@pytest.mark.parametrize(
    ('model', 'extra'),
    [
        (AssociationOtfDiseaseView, {'indirect': 'Array(String)'}),
        (AssociationOtfTargetView, {'isMeasurement': 'Bool', 'indirect': 'Array(LowCardinality(String))'}),
    ],
    ids=['disease', 'target'],
)
def test_aotf_views_land_the_served_columns(model, extra):
    served = {**AOTF_SERVED, **extra}
    order = ['rowId', 'A', 'B', 'datatypeId', 'datasourceId', 'rowScore', 'searchA', 'searchB']
    if 'isMeasurement' in extra:
        order.append('isMeasurement')
    order += ['noveltyDirect', 'noveltyIndirect', 'indirect']
    landed = {n: render(t, quote_fields=False) for n, t in columns(model)}
    assert list(landed) == order
    assert landed == served


_DETECTION = 'Array(Tuple(miIdentifier Nullable(String), shortName Nullable(String)))'
#: the served interaction table as interaction.sql built it (read off system.columns): no LowCardinality survives
#: groupArray, the speciesA names and interactionResources.sourceDatabase are String, taxon ids UInt32
INTERACTION_SERVED = (
    'targetA String, interactions Array(Tuple('
    'targetA String, intA String, targetB Nullable(String), intB String, '
    'intABiologicalRole String, intBBiologicalRole String, scoring Nullable(Float64), count UInt8, '
    f'sourceDatabase {_enum8(InteractionSource)}, '
    'speciesA Tuple(mnemonic String, scientificName String, taxonId UInt32), '
    'speciesB Tuple(mnemonic String, scientificName Nullable(String), taxonId Nullable(UInt32)), '
    'evidences Array(Tuple('
    'evidenceScore Nullable(Float64), expansionMethodMiIdentifier Nullable(String), '
    'expansionMethodShortName Nullable(String), hostOrganismScientificName Nullable(String), '
    'hostOrganismTaxId Nullable(UInt32), interactionDetectionMethodMiIdentifier String, '
    'interactionDetectionMethodShortName String, interactionIdentifier Nullable(String), '
    'interactionResources Tuple(databaseVersion String, sourceDatabase String), '
    'interactionTypeMiIdentifier Nullable(String), interactionTypeShortName Nullable(String), '
    f'participantDetectionMethodA {_DETECTION}, participantDetectionMethodB {_DETECTION}, '
    'pubmedId Nullable(String)))))'
)


def test_interaction_view_lands_the_served_interaction():
    assert ', '.join(f'{n} {render(t, quote_fields=False)}' for n, t in columns(InteractionView)) == INTERACTION_SERVED


#: `proteinCodingCoords` as protein_coding_coords.sql served it (read off system.columns): groupArray strips
#: LowCardinality, and the CAST turned the landing Enum8 datasourceId back into a String
PROTEIN_CODING_COORDS = (
    'Array(Tuple(targetId String, uniprotAccessions Array(String), aminoAcidPosition Int32, '
    'alternateAminoAcid String, referenceAminoAcid String, variantFunctionalConsequenceIds Array(String), '
    'variantEffect Nullable(Float64), variantId String, diseases Array(String), '
    'datasources Array(Tuple(datasourceCount UInt32, datasourceId String, datasourceNiceName String)), '
    'therapeuticAreas Array(String)))'
)


@pytest.mark.parametrize(
    ('model', 'key'),
    [(CodingVariantByVariantView, 'variantId'), (CodingVariantByTargetView, 'targetId')],
    ids=['variant', 'target'],
)
def test_coding_variant_views_land_the_served_columns(model, key):
    landed = {n: render(t, quote_fields=False) for n, t in columns(model)}
    assert landed == {key: 'String', 'proteinCodingCoords': PROTEIN_CODING_COORDS}
    assert list(landed) == [key, 'proteinCodingCoords']


#: The small compute tables as their declarations serve them: the top-level columns as before, each item the output
#: row (or the fields served) in output order and types (David, 2026-11-24, decision 4). Against the scripts'
#: tables (system.columns): disease_hpo evidence keeps `qualifier`, aspect/evidenceType/resource are Enum8 (were
#: Nullable(String)/String), sex Nullable(Enum8) (Nullable(String)), bioCuration String (Nullable); mouse phenotypes
#: in output order (alphabetical), targetInModelEnsemblId String (Nullable); drug warnings with chemblIds first,
#: country String and id UInt32 (Nullable), warningType Enum8 (String); mechanism rows (mechanismOfAction, actionType,
#: references, targetName, targets, curation) (references last), actionType Enum8 and targetName String (Nullable),
#: curation Enum8 (String), references (source, ids, urls) ((ids, source, urls)); adverse events (name, count, logLR,
#: critval, meddraCode) ((count, critval, name, logLR, meddraCode)).
SMALL_COMPUTE_SERVED = {
    DiseasePhenotypeByDiseaseView: (
        'disease String, phenotypes Array(Tuple(disease String, phenotype String, evidence Array(Tuple('
        f'aspect {_enum8(HpoAspect)}, bioCuration String, diseaseFromSourceId String, diseaseFromSource String, '
        f'diseaseName String, evidenceType {_enum8(HpoEvidenceCode)}, frequency Nullable(String), '
        'modifiers Array(String), onset Array(String), qualifier Nullable(String), qualifierNot Bool, '
        f'references Array(String), sex Nullable({_enum8(HpoSex)}), resource {_enum8(HpoResource)}))))'
    ),
    MousePhenotypeByTargetView: (
        'targetFromSourceId String, mouse_phenotypes Array(Tuple(targetInModel String, targetInModelMgiId String, '
        'targetInModelEnsemblId String, targetFromSourceId String, modelPhenotypeId String, '
        'modelPhenotypeLabel String, biologicalModels Array(Tuple(allelicComposition String, '
        'geneticBackground String, id Nullable(String), literature Array(String))), '
        'modelPhenotypeClasses Array(Tuple(id String, label String))))'
    ),
    DrugWarningByDrugView: (
        'chemblId String, drugWarnings Array(Tuple(chemblIds Array(String), toxicityClass Nullable(String), '
        'country String, description Nullable(String), id UInt32, '
        'references Array(Tuple(id String, source String, url String)), '
        f'warningType {_enum8(ChemblWarningType)}, year Nullable(UInt16), efoTerm Nullable(String), '
        'efoId Nullable(String), efoIdForWarningClass Nullable(String)))'
    ),
    DrugMechanismOfActionByDrugView: (
        f'chemblId String, rows Array(Tuple(mechanismOfAction String, actionType {_enum8(ChemblActionType)}, '
        'references Array(Tuple(source String, ids Array(String), urls Array(String))), targetName String, '
        f'targets Array(String), curation {_enum8(MechanismCuration)})), '
        f'uniqueTargetTypes Array({_enum8(ChemblTargetType)}), uniqueActionTypes Array({_enum8(ChemblActionType)})'
    ),
    PharmacovigilanceByDrugView: (
        'chembl_id String, adverse_events Array(Tuple(name String, count UInt32, logLR Float64, critval Float64, '
        'meddraCode String)), criticalValue Float64'
    ),
}


@pytest.mark.parametrize('model', list(SMALL_COMPUTE_SERVED), ids=lambda m: m.dataset_name)
def test_small_compute_views_land_the_served_columns(model):
    landed = ', '.join(f'{n} {render(t, quote_fields=False)}' for n, t in columns(model))
    assert landed == SMALL_COMPUTE_SERVED[model]


_STRAND = "Enum8('-1' = -1, '0' = 0, '1' = 1)"
#: the served tables of the target group as targets.sql, transcript.sql and target_prioritisation.sql built them
#: (read off system.columns): no LowCardinality survives groupArray, transcript flags are Nullable(String);
#: targets_by_region.target is Tuple(id String) (David, 2026-11-24); biotype is the transcript's Enum8
TARGET_GROUP_SERVED = {
    TargetByRegionView: (f'chromosome {_enum8(Chromosome)}, start UInt32, end UInt32, target Tuple(id String)'),
    TranscriptView: (
        f'targetId String, transcripts Array(Tuple(transcriptId String, biotype {_enum8(TranscriptBiotype)}, '
        'proteinId Nullable(String), uniprotSwissprotIds Array(String), uniprotTremblIds Array(String), '
        'uniprotIsoformIds Array(String), '
        'alphafoldIds Array(String), dbXrefs Array(Tuple(id String, source String)), isEnsemblCanonical Bool, '
        f'chromosome {_enum8(Chromosome)}, start UInt32, end UInt32, strand {_STRAND}, '
        'transcriptionStartSite UInt32, flags Array(Tuple(label Nullable(String), value Nullable(String))), '
        f'exons Array(Tuple(exonId String, chromosome {_enum8(Chromosome)}, start UInt32, end UInt32, '
        f'strand {_STRAND}))))'
    ),
    TargetPrioritisationView: 'targetId String, items Array(Tuple(key String, value String))',
}


@pytest.mark.parametrize('model', list(TARGET_GROUP_SERVED), ids=lambda m: m.dataset_name)
def test_target_group_views_land_the_served_columns(model):
    assert ', '.join(f'{n} {render(t, quote_fields=False)}' for n, t in columns(model)) == TARGET_GROUP_SERVED[model]


def test_target_view_lands_target_then_the_id_lists():
    """`targets`: the target columns as before, then the two lists targets.sql joined on."""
    landed = [(n, render(t, quote_fields=False)) for n, t in columns(TargetView)]
    assert landed[-2:] == [('credibleSetIds', 'Array(String)'), ('pharmacogenomicsIds', 'Array(UInt64)')]
    assert [n for n, _ in landed[: len(dataset_spec(Target).fields)]] == [f.name for f in dataset_spec(Target).fields]
    assert len(landed) == 31


@pytest.mark.parametrize(
    ('view', 'output', 'lists'),
    [
        (StudyView, Study, [('credibleSetIds', 'Array(String)')]),
        (VariantView, Variant, [('credibleSetIds', 'Array(String)'), ('pharmacogenomicsIds', 'Array(UInt64)')]),
    ],
    ids=['studies', 'variants'],
)
def test_genetics_b_views_land_the_output_then_the_id_lists(view, output, lists):
    """`studies` and `variants`: the output's landing columns, then the lists studies.sql and variants.sql joined
    on; `studies` without `credible_sets_by_study.studyId` (David, 2026-11-25). The output's columns take the
    view's `column_types`."""
    landed = [(n, render(t, quote_fields=False)) for n, t in columns(view)]
    types = view.clickhouse.column_types
    base = [
        (f.name, render(clickhouse_ddl._type(f.node, False, f.name, types), quote_fields=False))
        for f in dataset_spec(output).fields
    ]
    assert landed == [*base, *lists]


#: the served colocalisation and l2g_predictions as colocalisation.sql and l2g_predictions.sql built them (design
#: "Served DDL", read off system.columns): groupArray stripped LowCardinality from colocalisationMethod and
#: features.name, which stay String; rightStudyType and chromosome stay Enum8
GENETICS_A_SERVED = {
    ColocalisationByCredibleSetView: (
        'credibleSetId String, colocalisation Array(Tuple(credibleSetId String, otherCredibleSetId String, '
        f'rightStudyType {_enum8(StudyType)}, chromosome {_enum8(Chromosome)}, colocalisationMethod String, '
        'numberColocalisingVariants UInt32, h3 Float64, h4 Float64, clpp Float64, '
        'betaRatioSignAverage Nullable(Float64)))'
    ),
    L2gPredictionByCredibleSetView: (
        'credibleSetId String, l2g_predictions Array(Tuple(credibleSetId String, geneId String, score Float64, '
        'features Array(Tuple(name String, value Float64, shapValue Float64)), shapBaseValue Float64))'
    ),
}


@pytest.mark.parametrize('model', list(GENETICS_A_SERVED), ids=lambda m: m.dataset_name)
def test_genetics_a_views_land_the_served_columns(model):
    assert ', '.join(f'{n} {render(t, quote_fields=False)}' for n, t in columns(model)) == GENETICS_A_SERVED[model]


_CHR, _ST, _FMM = _enum8(Chromosome), _enum8(StudyType), _enum8(FineMappingMethod)
_LEAD_STATS = 'pValueMantissa Nullable(Float32), pValueExponent Nullable(Int32), beta Nullable(Float64)'
_TOP_L2G = (
    'topL2GGeneId Nullable(String), topL2GApprovedSymbol LowCardinality(Nullable(String)), '
    'topL2GScore Nullable(Float64)'
)
_DISEASES = 'Array(Tuple(id LowCardinality(String), name LowCardinality(String)))'
#: the served credible_sets, credible_sets_locus and read models as credible_sets.sql and the three read-model
#: scripts built them (design "Served DDL", read off system.columns): isTransQtl UInt8 (has()); any() stripped
#: LowCardinality from variant_credible_sets.condition; the LEFT JOIN defaults keep target_credible_sets.biosampleName
#: non-null and variant_credible_sets.biosampleName nullable. Both qualityControls are their outputs' Enum8s.
GENETICS_C_SERVED = {
    CredibleSetView: (
        f'credibleSetId String, studyId String, studyType {_ST}, chromosome {_CHR}, leadVariantId String, '
        'leadPosition UInt32, leadPValueMantissa Nullable(Float32), leadPValueExponent Nullable(Int32), '
        'leadBeta Nullable(Float64), leadStandardError Nullable(Float64), '
        'leadEffectAlleleFrequency Nullable(Float32), leadSampleSize Nullable(UInt32), locusStart UInt32, '
        f'locusEnd UInt32, locusSize UInt32, fineMappingMethod {_FMM}, logBayesFactor Nullable(Float64), '
        'confidence UInt8, purityMeanR2 Nullable(Float64), purityMinR2 Nullable(Float64), '
        f'qualityControls Array({_enum8(CredibleSetQualityControl)}), isTransQtl UInt8'
    ),
    CredibleSetLocusView: (
        'credibleSetId String, locus Array(Tuple(variantId String, posteriorProbability Float32, '
        'logBayesFactor Nullable(Float32), pValueMantissa Nullable(Float32), pValueExponent Nullable(Int32), '
        'effectAlleleFrequency Nullable(Float32), sampleSize Nullable(UInt32), reportedBeta Nullable(Float64), '
        'reportedStandardError Nullable(Float64), r2Overall Nullable(Float32), conditionalBeta Nullable(Float32), '
        'conditionalStandardError Nullable(Float32), rescaledAbsBeta Nullable(Float32), '
        'rescaledStandardError Nullable(Float32), rescaledDirection Nullable(Int8), '
        f'qualityControls Array({_enum8(VariantQualityControl)})))'
    ),
    StudyCredibleSetView: (
        f'studyId String, credibleSetId String, leadVariantId String, leadChromosome {_CHR}, leadPosition UInt32, '
        f'leadReferenceAllele String, leadAlternateAllele String, {_LEAD_STATS}, fineMappingMethod {_FMM}, '
        f'confidence UInt8, locusSize UInt32, {_TOP_L2G}'
    ),
    TargetCredibleSetView: (
        f'targetId String, credibleSetId String, {_LEAD_STATS}, fineMappingMethod {_FMM}, confidence UInt8, '
        f'isTransQtl UInt8, leadVariantId String, leadChromosome {_CHR}, leadPosition UInt32, '
        f'leadReferenceAllele String, leadAlternateAllele String, studyId String, studyType {_ST}, '
        'condition LowCardinality(Nullable(String)), biosampleId Nullable(String), '
        'biosampleName LowCardinality(String), locusSize UInt32'
    ),
    VariantCredibleSetView: (
        f'studyType {_ST}, variantId String, credibleSetId String, posteriorProbability Float32, locusSize UInt32, '
        f'{_LEAD_STATS}, fineMappingMethod {_FMM}, confidence UInt8, isTransQtl UInt8, leadVariantId String, '
        f'leadChromosome {_CHR}, leadPosition UInt32, leadReferenceAllele String, leadAlternateAllele String, '
        'studyId String, condition Nullable(String), studyTargetId Nullable(String), '
        'studyTargetApprovedSymbol LowCardinality(Nullable(String)), biosampleId Nullable(String), '
        'biosampleName LowCardinality(Nullable(String)), traitFromSource LowCardinality(String), '
        f'diseases {_DISEASES}, diseaseTherapeuticAreas {_DISEASES}, {_TOP_L2G}'
    ),
}


@pytest.mark.parametrize('model', list(GENETICS_C_SERVED), ids=lambda m: m.dataset_name)
def test_genetics_c_views_land_the_served_columns(model):
    assert ', '.join(f'{n} {render(t, quote_fields=False)}' for n, t in columns(model)) == GENETICS_C_SERVED[model]


# served tables

STR = CHType('String')


KEYED_NAMES = ClickHouseDictionary('keyed_names', key='id', attributes=('name',))


class Keyed(DatasetModel):
    dataset_name = 'keyed_view'
    dataset_title = 'Keyed'
    dataset_description = 'Keyed.'
    example = {}
    clickhouse = Serving(engine='rocksdb', dictionaries=(KEYED_NAMES,))

    id: Annotated[str, PrimaryKey()]
    name: str


class Ordered(DatasetModel):
    dataset_name = 'ordered_view'
    dataset_title = 'Ordered'
    dataset_description = 'Ordered.'
    example = {}
    clickhouse = Serving(engine='mergetree', order_by=('target', 'id'))

    id: str
    target: str | None
    score: float


class TwoKeys(DatasetModel):
    dataset_name = 'two_keys_view'
    dataset_title = 'Two keys'
    dataset_description = 'Two keys.'
    example = {}
    clickhouse = Serving(engine='rocksdb')

    a: Annotated[str, PrimaryKey()]
    b: Annotated[str, PrimaryKey()]


def test_rocksdb_table_sql_keys_on_the_primary_key():
    assert table_sql(Keyed) == [
        'DROP TABLE IF EXISTS `keyed_view` SYNC',
        'CREATE TABLE `keyed_view` (`id` String, `name` String) ENGINE = EmbeddedRocksDB() PRIMARY KEY `id`',
        'INSERT INTO `keyed_view` SELECT `id`, `name` FROM `keyed_view_log`',
    ]


def test_rocksdb_needs_one_primary_key_column():
    assert rocksdb_key(Keyed) == 'id'
    with pytest.raises(ValueError, match=r"two_keys_view: rocksdb needs one PrimaryKey column, found \('a', 'b'\)"):
        table_sql(TwoKeys)


def test_mergetree_nullable_order_by_allows_nullable_key_and_is_optimised():
    assert table_sql(Ordered) == [
        'DROP TABLE IF EXISTS `ordered_view` SYNC',
        (
            'CREATE TABLE `ordered_view` (`id` String, `target` Nullable(String), `score` Float64) '
            'ENGINE = MergeTree() ORDER BY (`target`, `id`) SETTINGS allow_nullable_key = 1'
        ),
        'INSERT INTO `ordered_view` SELECT `id`, `target`, `score` FROM `ordered_view_log`',
        'OPTIMIZE TABLE `ordered_view` FINAL',
    ]


def test_mergetree_without_a_nullable_order_by_has_no_settings(monkeypatch):
    monkeypatch.setattr(Ordered, 'clickhouse', Serving(engine='mergetree', order_by=('score', 'id')))
    assert table_sql(Ordered)[1].endswith('ENGINE = MergeTree() ORDER BY (`score`, `id`)')


def test_a_model_without_serving_has_no_table():
    with pytest.raises(ValueError, match='toy is not a served view'):
        table_sql(Toy)


def test_served_sql_rebuilds_the_dictionaries_over_the_table_then_drops_the_landing():
    assert served_sql(Keyed) == [
        'DROP DICTIONARY IF EXISTS `keyed_names` SYNC',
        *table_sql(Keyed),
        dictionary_sql(KEYED_NAMES, 'keyed_view', columns(Keyed)),
        'DROP TABLE `keyed_view_log` SYNC',
    ]
    assert served_sql(Ordered) == [*table_sql(Ordered), 'DROP TABLE `ordered_view_log` SYNC']


def test_landing_ddl_is_a_log_table_of_the_columns():
    assert landing_ddl(Ordered) == (
        'CREATE TABLE IF NOT EXISTS `ordered_view_log` (`id` String, `target` Nullable(String), `score` Float64) '
        'ENGINE = Log'
    )


def test_dictionary_sql_renames_through_expression():
    d = ClickHouseDictionary(
        'biosample_names',
        key='id',
        attributes=('name',),
        source_columns={'id': 'biosampleId', 'name': 'biosampleName'},
    )
    cols = [('biosampleId', STR), ('biosampleName', STR), ('other', STR)]
    assert dictionary_sql(d, 'biosample', cols) == (
        'CREATE OR REPLACE DICTIONARY `biosample_names` '
        '(`id` String EXPRESSION `biosampleId`, `name` String EXPRESSION `biosampleName`) PRIMARY KEY `id` '
        "SOURCE(CLICKHOUSE(TABLE 'biosample')) LAYOUT(COMPLEX_KEY_HASHED()) LIFETIME(0)"
    )


def test_dictionary_sql_identity_columns_have_no_expression():
    d = ClickHouseDictionary('d', key='id', attributes=('name',))
    assert dictionary_sql(d, 't', [('id', STR), ('name', STR)]) == (
        'CREATE OR REPLACE DICTIONARY `d` (`id` String, `name` String) PRIMARY KEY `id` '
        "SOURCE(CLICKHOUSE(TABLE 't')) LAYOUT(COMPLEX_KEY_HASHED()) LIFETIME(0)"
    )


def test_dictionary_sql_drops_low_cardinality():
    d = ClickHouseDictionary('sequence_ontology', key='id', attributes=('label',))
    low = CHType('String', low_cardinality=True)
    assert dictionary_sql(d, 'sequence_ontology_source', [('id', low), ('label', low)]).startswith(
        'CREATE OR REPLACE DICTIONARY `sequence_ontology` (`id` String, `label` String) PRIMARY KEY `id` '
    )


def test_drop_dictionary_sql():
    d = ClickHouseDictionary('biosample_names', key='id', attributes=('name',))
    assert drop_dictionary_sql(d) == 'DROP DICTIONARY IF EXISTS `biosample_names` SYNC'


def test_uniqueness_sql_reads_the_key_in_the_landing_table():
    assert uniqueness_sql(Keyed) == (
        "SELECT throwIf(count() != uniqExact(tuple(`id`)), 'duplicate id in keyed_view') FROM `keyed_view_log`"
    )


def test_table_sql_rerun_keeps_one_copy_of_the_rows(clickhouse_database):
    client = clickhouse_database
    client.command(landing_ddl(Ordered))
    client.command("INSERT INTO ordered_view_log VALUES ('a', 't1', 1), ('b', NULL, 2)")
    for _ in range(2):
        for statement in table_sql(Ordered):
            client.command(statement)
    assert client.command('SELECT count() FROM ordered_view') == 2


def test_served_sql_rerun_reads_the_rebuilt_table(clickhouse_database):
    client = clickhouse_database
    for name in ('nx', 'ny'):
        client.command(landing_ddl(Keyed))
        client.insert('keyed_view_log', [['x', name]])
        for statement in served_sql(Keyed):
            client.command(statement)
        assert client.command("SELECT dictGet('keyed_names', 'name', tuple('x'))") == name
    assert client.command('EXISTS TABLE keyed_view_log') == 0
