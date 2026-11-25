"""pos.clickhouse_ddl: ClickHouse landing DDL generated from the dataset models."""

import datetime
from pathlib import Path
from typing import Annotated, Literal, get_args

import polars as pl
import pytest
import yaml
from ot_datamodels.base import ClickHouseType, DatasetModel, PolarsDtype, PrimaryKey, StructModel
from ot_datamodels.discovery import VIEWS_PACKAGE, discover_datasets, discover_models
from ot_datamodels.outputs.study import Study
from ot_datamodels.outputs.target import Target
from ot_datamodels.outputs.variant import Variant
from ot_datamodels.spec import dataset_spec
from ot_datamodels.views.association_otf_disease import AssociationOtfDisease
from ot_datamodels.views.association_otf_target import AssociationOtfTarget
from ot_datamodels.views.coding_variant_by_target import CodingVariantByTarget
from ot_datamodels.views.coding_variant_by_variant import CodingVariantByVariant
from ot_datamodels.views.colocalisation_by_credible_set import ColocalisationByCredibleSet
from ot_datamodels.views.disease_phenotype_by_disease import DiseasePhenotypeByDisease
from ot_datamodels.views.disease_view import DiseaseView
from ot_datamodels.views.drug_mechanism_of_action_by_drug import DrugMechanismOfActionByDrug
from ot_datamodels.views.drug_warning_by_drug import DrugWarningByDrug
from ot_datamodels.views.interaction_view import InteractionView
from ot_datamodels.views.l2g_prediction_by_credible_set import L2gPredictionByCredibleSet
from ot_datamodels.views.mouse_phenotype_by_target import MousePhenotypeByTarget
from ot_datamodels.views.pharmacovigilance_by_drug import PharmacovigilanceByDrug
from ot_datamodels.views.study_view import StudyView
from ot_datamodels.views.target_by_region import TargetByRegion
from ot_datamodels.views.target_prioritisation_view import TargetPrioritisationView
from ot_datamodels.views.target_view import TargetView
from ot_datamodels.views.transcript_view import TranscriptView
from ot_datamodels.views.variant_view import VariantView
from ot_datamodels.vocabularies import (
    ChemblActionType,
    ChemblTargetType,
    ChemblWarningType,
    Chromosome,
    Datasource,
    Datatype,
    HpoAspect,
    HpoEvidenceCode,
    HpoResource,
    HpoSex,
    InteractionSource,
    MechanismCuration,
    StudyType,
)

from pos import clickhouse_ddl
from pos.clickhouse_ddl import (
    CHType,
    canonical,
    column_type,
    columns,
    create_table,
    ddl_for,
    declarations,
    declared_sql,
    dictionary_sql,
    drop_dictionary_sql,
    entry_problems,
    landing_columns,
    member_table,
    members,
    model_type,
    parse_table,
    parse_type,
    render,
    rocksdb_key,
    table_sql,
    type_at,
    union_columns,
    uniqueness_sql,
)
from pos.clickhouse_tables import ClickHouseDictionary, ClickHouseTable, entry_tables

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


class MemberB(DatasetModel):
    dataset_name = 'member_b'
    dataset_title = 'B'
    dataset_description = 'B.'
    example = {}

    id: str
    only_b: float | None


class ClashA(DatasetModel):
    dataset_name = 'clash_a'
    dataset_title = 'A'
    dataset_description = 'A.'
    example = {}

    x: int


class ClashB(DatasetModel):
    dataset_name = 'clash_b'
    dataset_title = 'B'
    dataset_description = 'B.'
    example = {}

    x: str


def test_union_absent_struct_and_list():
    union = {name: render(t) for name, t in union_columns([MemberA, MemberB])}
    assert list(union) == ['id', 's', 'l', 'only_a', 'only_b']
    assert union == {
        'id': 'String',
        's': 'Tuple(`a` Nullable(String))',
        'l': 'Array(String)',
        'only_a': 'Nullable(Int64)',
        'only_b': 'Nullable(Float64)',
    }


def test_union_type_clash_raises():
    with pytest.raises(ValueError, match=r'^column x:') as info:
        union_columns([ClashA, ClashB])
    assert 'clash_a' in str(info.value)
    assert 'clash_b' in str(info.value)


class NullableBar(DatasetModel):
    dataset_name = 'nullable_bar'
    dataset_title = 'N'
    dataset_description = 'N.'
    example = {}

    s: Bar | None


class PlainBar(DatasetModel):
    dataset_name = 'plain_bar'
    dataset_title = 'P'
    dataset_description = 'P.'
    example = {}

    s: Bar


def test_union_nested_field_nullability():
    union = union_columns([PlainBar, NullableBar])
    assert [(n, render(t)) for n, t in union] == [('s', 'Tuple(`a` Nullable(String))')]


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


class PppOnly(DatasetModel):
    dataset_name = 'toy_ppp'
    dataset_title = 'PPP'
    dataset_description = 'PPP.'
    products = ('ppp',)
    example = {}

    id: str


def test_members_filters_product(monkeypatch):
    models = {m.dataset_name: m for m in (MemberB, MemberA, PppOnly, Toy)}
    monkeypatch.setattr(clickhouse_ddl, 'discover_models', lambda: models)
    assert members('*_*', 'platform') == [MemberA, MemberB]
    assert members('*_*', 'ppp') == [MemberA, MemberB, PppOnly]


def test_landing_columns(monkeypatch):
    models = {m.dataset_name: m for m in (MemberB, MemberA, PppOnly, Toy)}
    monkeypatch.setattr(clickhouse_ddl, 'discover_models', lambda: models)
    glob = {'input_dir': 'output', 'models': 'member_*'}
    assert landing_columns(glob, 'platform') == union_columns([MemberA, MemberB])
    assert landing_columns({'input_dir': 'output/x', 'model': 'toy_ppp'}, 'platform') == columns(PppOnly)
    assert landing_columns({'input_dir': 'view/toy', 'preload_script': 'toy.sql'}, 'platform') is None
    assert landing_columns({'input_dir': 'output/toy'}, 'platform') is None


def test_ddl_for(monkeypatch):
    models = {m.dataset_name: m for m in (MemberB, MemberA, PppOnly, Toy)}
    monkeypatch.setattr(clickhouse_ddl, 'discover_models', lambda: models)
    one = {'table': 't_log', 'model': 'toy_ppp', 'engine': 'MergeTree ORDER BY id'}
    assert ddl_for(one, 'platform') == [('t_log', create_table('t_log', columns(PppOnly), 'MergeTree ORDER BY id'))]
    union = {'table': 'u_log', 'models': 'member_*'}
    assert ddl_for(union, 'platform') == [
        ('member_a_log', create_table('member_a_log', columns(MemberA))),
        ('member_b_log', create_table('member_b_log', columns(MemberB))),
        ('u_log', create_table('u_log', union_columns([MemberA, MemberB]))),
    ]
    assert ddl_for({'table': 'v', 'preload_script': 'v.sql'}, 'platform') == []
    with pytest.raises(ValueError, match=r"^p_log: models 'toy_\*' matches no model for platform$"):
        ddl_for({'table': 'p_log', 'models': 'toy_*'}, 'platform')


def test_pos_ddl_prints_one_dataset(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr('sys.argv', ['pos-ddl', 'disease'])
    clickhouse_ddl.main()
    out = capsys.readouterr().out
    assert out.startswith('CREATE TABLE IF NOT EXISTS `disease_log` (')


def test_pos_ddl_evidence_follows_the_product(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr('sys.argv', ['pos-ddl', 'evidence', '--product', 'ppp'])
    clickhouse_ddl.main()
    ppp = capsys.readouterr().out
    monkeypatch.setattr('sys.argv', ['pos-ddl', 'evidence'])
    clickhouse_ddl.main()
    platform = capsys.readouterr().out
    assert '`assays`' in ppp
    assert '`assays`' not in platform
    assert '`evidence_europepmc_log`' in platform
    assert platform.count('CREATE TABLE') == len(members('evidence_*', 'platform')) + 1


def test_pos_ddl_all_prints_every_modelled_entry(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr('sys.argv', ['pos-ddl', '--all'])
    clickhouse_ddl.main()
    out = capsys.readouterr().out
    assert '`disease_log`' in out
    assert '`evidence_log`' in out
    assert '`targets_log`' in out


def test_type_at():
    cols = columns(Toy)
    assert type_at(cols, 'tags') == dict(cols)['tags']
    assert type_at(cols, 'tags[]') == CHType('String')
    assert type_at(cols, 'nested[][]') == CHType('Int64')
    assert type_at(cols, 'struct/b') == CHType('Int64')
    assert type_at(cols, 'nullable_struct/a') == CHType('String', nullable=True)
    assert type_at(cols, 'bars[]/a') == CHType('String', nullable=True)
    assert type_at(cols, '') == CHType('Tuple', fields=tuple(cols))
    assert type_at(cols, 'absent') is None
    assert type_at(cols, 'struct/absent') is None
    assert type_at(cols, 'absent[]/a') is None


def test_canonical_numbers_enums():
    numbered = parse_type("Enum8('a' = 1, 'b' = 2)")
    assert canonical(parse_type("Enum('a', 'b')")) == numbered
    assert canonical(numbered) == numbered
    assert canonical(parse_type("Enum8('b' = 1, 'a' = 2)")) != numbered
    assert canonical(parse_type("Enum('a' = 5, 'b')")) == parse_type("Enum8('a' = 5, 'b' = 6)")
    assert canonical(parse_type("Enum('a' = 200)")) == parse_type("Enum16('a' = 200)")
    nested = canonical(parse_type("Array(Tuple(x Nullable(Enum('a', 'b'))))"))
    assert nested == parse_type("Array(Tuple(x Nullable(Enum8('a' = 1, 'b' = 2))))")


def test_create_table_quotes_identifiers():
    sql = create_table('toy_log', [('id', column_type(dataset_spec(MemberA).fields[0].node))])
    assert sql == 'CREATE TABLE IF NOT EXISTS `toy_log` (`id` String) ENGINE = Log'


def test_every_entry_round_trips():
    entries = yaml.safe_load((ROOT / 'config' / 'datasets.yaml').read_text())['clickhouse']
    models = discover_models()
    checked = 0
    for name, entry in entries.items():
        if 'model' not in entry:
            continue
        cols = columns(models[entry['model']])
        assert list(parse_table(create_table(entry['table'], cols)).items()) == cols, name
        checked += 1
    assert checked > 20
    served = ('aotf_disease', 'aotf_target', 'association_time_series', 'colocalisation', 'disease', 'disease_hpo',
              'drug', 'drug_warnings', 'interaction', 'l2g_predictions', 'mechanism_of_action', 'mouse_phenotypes',
              'pharmacovigilance', 'protein_coding_coords_by_target', 'protein_coding_coords_by_variant', 'studies',
              'target_prioritisation', 'targets', 'targets_by_region', 'transcript', 'variants')  # fmt: skip
    views = {entries[n]['model'] for n in served}
    assert views == set(discover_datasets(VIEWS_PACKAGE))
    for product in ('platform', 'ppp'):
        found = members(entries['evidence']['models'], product)
        assert found, product
        tables = [(member_table(m), columns(m)) for m in found]
        tables.append((entries['evidence']['table'], union_columns(found)))
        for table, cols in tables:
            assert list(parse_table(create_table(table, cols)).items()) == cols, (product, table)


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
    t = type_at(columns(TargetView), path)
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
    return f"Enum8({', '.join(f"'{v}' = {i}" for i, v in enumerate(get_args(vocabulary), 1))})"


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
        (AssociationOtfDisease, {'indirect': 'Array(String)'}),
        (AssociationOtfTarget, {'isMeasurement': 'Bool', 'indirect': 'Array(LowCardinality(String))'}),
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
    [(CodingVariantByVariant, 'variantId'), (CodingVariantByTarget, 'targetId')],
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
    DiseasePhenotypeByDisease: (
        'disease String, phenotypes Array(Tuple(disease String, phenotype String, evidence Array(Tuple('
        f'aspect {_enum8(HpoAspect)}, bioCuration String, diseaseFromSourceId String, diseaseFromSource String, '
        f'diseaseName String, evidenceType {_enum8(HpoEvidenceCode)}, frequency Nullable(String), '
        'modifiers Array(String), onset Array(String), qualifier Nullable(String), qualifierNot Bool, '
        f'references Array(String), sex Nullable({_enum8(HpoSex)}), resource {_enum8(HpoResource)}))))'
    ),
    MousePhenotypeByTarget: (
        'targetFromSourceId String, mouse_phenotypes Array(Tuple(targetInModel String, targetInModelMgiId String, '
        'targetInModelEnsemblId String, targetFromSourceId String, modelPhenotypeId String, '
        'modelPhenotypeLabel String, biologicalModels Array(Tuple(allelicComposition String, '
        'geneticBackground String, id Nullable(String), literature Array(String))), '
        'modelPhenotypeClasses Array(Tuple(id String, label String))))'
    ),
    DrugWarningByDrug: (
        'chemblId String, drugWarnings Array(Tuple(chemblIds Array(String), toxicityClass Nullable(String), '
        'country String, description Nullable(String), id UInt32, '
        'references Array(Tuple(id String, source String, url String)), '
        f'warningType {_enum8(ChemblWarningType)}, year Nullable(UInt16), efoTerm Nullable(String), '
        'efoId Nullable(String), efoIdForWarningClass Nullable(String)))'
    ),
    DrugMechanismOfActionByDrug: (
        f'chemblId String, rows Array(Tuple(mechanismOfAction String, actionType {_enum8(ChemblActionType)}, '
        'references Array(Tuple(source String, ids Array(String), urls Array(String))), targetName String, '
        f'targets Array(String), curation {_enum8(MechanismCuration)})), '
        f'uniqueTargetTypes Array({_enum8(ChemblTargetType)}), uniqueActionTypes Array({_enum8(ChemblActionType)})'
    ),
    PharmacovigilanceByDrug: (
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
#: targets_by_region.target is Tuple(id String) (David, 2026-11-24)
TARGET_GROUP_SERVED = {
    TargetByRegion: (
        f'chromosome {_enum8(Chromosome)}, start UInt32, end UInt32, target Tuple(id String)'
    ),
    TranscriptView: (
        'targetId String, transcripts Array(Tuple(transcriptId String, biotype String, proteinId Nullable(String), '
        'uniprotSwissprotIds Array(String), uniprotTremblIds Array(String), uniprotIsoformIds Array(String), '
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
    on; `studies` without `credible_sets_by_study.studyId` (David, 2026-11-25)."""
    landed = [(n, render(t, quote_fields=False)) for n, t in columns(view)]
    base = [(n, render(t, quote_fields=False)) for n, t in columns(output)]
    assert landed == [*base, *lists]


#: the served colocalisation and l2g_predictions as colocalisation.sql and l2g_predictions.sql built them (design
#: "Served DDL", read off system.columns): groupArray stripped LowCardinality from colocalisationMethod and
#: features.name, which stay String; rightStudyType and chromosome stay Enum8
GENETICS_A_SERVED = {
    ColocalisationByCredibleSet: (
        'credibleSetId String, colocalisation Array(Tuple(credibleSetId String, otherCredibleSetId String, '
        f'rightStudyType {_enum8(StudyType)}, chromosome {_enum8(Chromosome)}, colocalisationMethod String, '
        'numberColocalisingVariants UInt32, h3 Float64, h4 Float64, clpp Float64, '
        'betaRatioSignAverage Nullable(Float64)))'
    ),
    L2gPredictionByCredibleSet: (
        'credibleSetId String, l2g_predictions Array(Tuple(credibleSetId String, geneId String, score Float64, '
        'features Array(Tuple(name String, value Float64, shapValue Float64)), shapBaseValue Float64))'
    ),
}


@pytest.mark.parametrize('model', list(GENETICS_A_SERVED), ids=lambda m: m.dataset_name)
def test_genetics_a_views_land_the_served_columns(model):
    assert ', '.join(f'{n} {render(t, quote_fields=False)}' for n, t in columns(model)) == GENETICS_A_SERVED[model]


# declared tables

STR = CHType('String')
LANDING = [
    ('id', STR),
    ('drugId', STR),
    ('targetId', CHType('String', nullable=True)),
    ('score', CHType('Float64', nullable=True)),
    ('tags', CHType('Array', element=STR)),
]


def test_rocksdb_table_sql_keys_on_the_given_key():
    table = ClickHouseTable('gene_ontology', 'rocksdb')
    assert table_sql(table, [('id', STR), ('name', STR)], 'gene_ontology_log', key='id') == [
        'DROP TABLE IF EXISTS `gene_ontology` SYNC',
        (
            'CREATE TABLE `gene_ontology` (`id` String, `name` String) '
            'ENGINE = EmbeddedRocksDB() PRIMARY KEY `id`'
        ),
        'INSERT INTO `gene_ontology` SELECT `id`, `name` FROM `gene_ontology_log`',
    ]


def test_rocksdb_table_sql_needs_a_key():
    with pytest.raises(ValueError, match='gene_ontology: rocksdb needs a key'):
        table_sql(ClickHouseTable('gene_ontology', 'rocksdb'), [('id', STR)], 'l')


def test_mergetree_nullable_order_by_allows_nullable_key_and_is_optimised():
    table = ClickHouseTable('t', 'mergetree', order_by=('targetId', 'id'), columns=('id', 'targetId'))
    assert table_sql(table, LANDING, 't_log') == [
        'DROP TABLE IF EXISTS `t` SYNC',
        (
            'CREATE TABLE `t` (`id` String, `targetId` Nullable(String)) '
            'ENGINE = MergeTree() ORDER BY (`targetId`, `id`) SETTINGS allow_nullable_key = 1'
        ),
        'INSERT INTO `t` SELECT `id`, `targetId` FROM `t_log`',
        'OPTIMIZE TABLE `t` FINAL',
    ]


def test_mergetree_without_a_nullable_order_by_has_no_settings():
    table = ClickHouseTable('t', 'mergetree', order_by=('id',), columns=('id',))
    assert table_sql(table, LANDING, 'l')[1] == 'CREATE TABLE `t` (`id` String) ENGINE = MergeTree() ORDER BY (`id`)'


def test_mergetree_where_and_columns_order():
    table = ClickHouseTable(
        'clinical_target',
        'mergetree',
        order_by=('targetId', 'id'),
        columns=('targetId', 'drugId', 'id'),
        where='targetId IS NOT NULL',
    )
    assert table_sql(table, LANDING, 'clinical_target_log') == [
        'DROP TABLE IF EXISTS `clinical_target` SYNC',
        (
            'CREATE TABLE `clinical_target` (`targetId` Nullable(String), `drugId` String, `id` String) '
            'ENGINE = MergeTree() ORDER BY (`targetId`, `id`) SETTINGS allow_nullable_key = 1'
        ),
        (
            'INSERT INTO `clinical_target` SELECT `targetId`, `drugId`, `id` FROM `clinical_target_log` '
            'WHERE targetId IS NOT NULL'
        ),
        'OPTIMIZE TABLE `clinical_target` FINAL',
    ]


def test_no_columns_keeps_every_landing_column():
    table = ClickHouseTable('t', 'rocksdb')
    assert table_sql(table, LANDING, 'l', key='id')[2] == (
        'INSERT INTO `t` SELECT `id`, `drugId`, `targetId`, `score`, `tags` FROM `l`'
    )


def test_dictionary_sql_renames_through_expression():
    d = ClickHouseDictionary(
        'biosample_names',
        'biosample',
        key='id',
        attributes=('name',),
        source_columns={'id': 'biosampleId', 'name': 'biosampleName'},
    )
    cols = [('biosampleId', STR), ('biosampleName', STR), ('other', STR)]
    assert dictionary_sql(d, cols) == (
        'CREATE OR REPLACE DICTIONARY `biosample_names` '
        '(`id` String EXPRESSION `biosampleId`, `name` String EXPRESSION `biosampleName`) PRIMARY KEY `id` '
        "SOURCE(CLICKHOUSE(TABLE 'biosample')) LAYOUT(COMPLEX_KEY_HASHED()) LIFETIME(0)"
    )


def test_dictionary_sql_identity_columns_have_no_expression():
    d = ClickHouseDictionary('d', 't', key='id', attributes=('name',))
    assert dictionary_sql(d, [('id', STR), ('name', STR)]) == (
        'CREATE OR REPLACE DICTIONARY `d` (`id` String, `name` String) PRIMARY KEY `id` '
        "SOURCE(CLICKHOUSE(TABLE 't')) LAYOUT(COMPLEX_KEY_HASHED()) LIFETIME(0)"
    )


def test_dictionary_sql_drops_low_cardinality():
    d = ClickHouseDictionary('sequence_ontology', 'sequence_ontology_source', key='id', attributes=('label',))
    low = CHType('String', low_cardinality=True)
    assert dictionary_sql(d, [('id', low), ('label', low)]).startswith(
        'CREATE OR REPLACE DICTIONARY `sequence_ontology` (`id` String, `label` String) PRIMARY KEY `id` '
    )


def test_drop_dictionary_sql():
    d = ClickHouseDictionary('biosample_names', 'biosample', key='id', attributes=('name',))
    assert drop_dictionary_sql(d) == 'DROP DICTIONARY IF EXISTS `biosample_names` SYNC'


def test_uniqueness_sql_reads_the_key_through_the_filter():
    table = ClickHouseTable('biosample', 'rocksdb', where='biosampleId != 3')
    assert uniqueness_sql(table, 'biosampleId', 'biosample_log') == (
        "SELECT throwIf(count() != uniqExact(tuple(`biosampleId`)), 'duplicate biosampleId in biosample') "
        'FROM `biosample_log` WHERE biosampleId != 3'
    )


class Keyed(DatasetModel):
    dataset_name = 'keyed'
    dataset_title = 'Keyed'
    dataset_description = 'Keyed.'
    example = {}
    products = ('platform', 'ppp')

    id: Annotated[str, PrimaryKey()]


class KeyedPppOnly(DatasetModel):
    dataset_name = 'keyed_ppp_only'
    dataset_title = 'PPP only'
    dataset_description = 'PPP only.'
    example = {}
    products = ('ppp',)

    id: Annotated[str, PrimaryKey()]


class OtherKey(DatasetModel):
    dataset_name = 'other_key'
    dataset_title = 'Other key'
    dataset_description = 'Other key.'
    example = {}

    gene: Annotated[str, PrimaryKey()]


class TwoKeys(DatasetModel):
    dataset_name = 'two_keys'
    dataset_title = 'Two keys'
    dataset_description = 'Two keys.'
    example = {}

    a: Annotated[str, PrimaryKey()]
    b: Annotated[str, PrimaryKey()]


KEYED = {
    'table': 'keyed_log',
    'model': 'keyed',
    'tables': [{'name': 'keyed', 'engine': 'rocksdb'}],
    'dictionaries': [{'name': 'keyed_names', 'source': 'keyed', 'key': 'id', 'attributes': ['id']}],
}
PPP_ONLY = {'table': 'ppp_only_log', 'model': 'keyed_ppp_only', 'tables': [{'name': 'ppp_only', 'engine': 'rocksdb'}]}


def test_declarations_follow_the_model_products(monkeypatch):
    monkeypatch.setattr(clickhouse_ddl, 'discover_models', lambda: {'keyed': Keyed, 'keyed_ppp_only': KeyedPppOnly})
    tables, dictionaries = declarations(KEYED, 'platform')
    assert [t.name for t in tables] == ['keyed']
    assert [d.name for d in dictionaries] == ['keyed_names']
    assert declarations(PPP_ONLY, 'platform') == ([], [])
    assert [t.name for t in declarations(PPP_ONLY, 'ppp')[0]] == ['ppp_only']


def test_a_models_entry_declares_for_every_product():
    entry = {'table': 'u_log', 'models': 'keyed*', 'tables': [{'name': 'u', 'engine': 'rocksdb'}]}
    assert declarations(entry, 'platform') == declarations(entry, 'ppp') == ([ClickHouseTable('u', 'rocksdb')], [])


def test_rocksdb_key_is_the_one_primary_key_the_models_share():
    assert rocksdb_key([Keyed]) == 'id'
    assert rocksdb_key([Keyed, KeyedPppOnly]) == 'id'
    with pytest.raises(ValueError, match='disagree'):
        rocksdb_key([Keyed, OtherKey])
    with pytest.raises(ValueError, match='one PrimaryKey column'):
        rocksdb_key([TwoKeys])


def test_declared_sql_keys_rocksdb_on_the_model_primary_key(monkeypatch):
    monkeypatch.setattr(clickhouse_ddl, 'discover_models', lambda: {'keyed': Keyed})
    statements = declared_sql(KEYED, 'platform')
    assert 'ENGINE = EmbeddedRocksDB() PRIMARY KEY `id`' in statements[2]
    assert statements[0] == 'DROP DICTIONARY IF EXISTS `keyed_names` SYNC'
    assert statements[-2].startswith('CREATE OR REPLACE DICTIONARY `keyed_names`')
    assert statements[-1] == 'DROP TABLE `keyed_log` SYNC'


def test_entry_tables_take_only_the_table_fields():
    entry = {'table': 'e_log', 'tables': [{'name': 'e', 'engine': 'rocksdb', 'key': 'id'}]}
    with pytest.raises(ValueError, match=r"unknown ClickHouseTable fields \['key'\]"):
        entry_tables(entry)


def test_entry_problems_check_a_yaml_declaration_against_its_members(monkeypatch):
    monkeypatch.setattr(clickhouse_ddl, 'discover_models', lambda: {'keyed': Keyed, 'other_key': OtherKey})
    good = {'table': 'u_log', 'models': 'keyed', 'tables': [{'name': 'u', 'engine': 'rocksdb'}]}
    assert entry_problems(good, 'platform') == []
    bad = {
        'table': 'u_log',
        'models': '*',
        'tables': [{'name': 'u', 'engine': 'mergetree', 'order_by': ['nope'], 'where': 'zip IS NULL'}],
    }
    assert entry_problems(bad, 'platform') == [
        "u_log.u: where 'zip' is not a column of the model",
        "u_log.u: order_by 'nope' is not a column of the table",
    ]
    disagree = {'table': 'u_log', 'models': '*', 'tables': [{'name': 'u', 'engine': 'rocksdb'}]}
    assert entry_problems(disagree, 'platform') == [
        "u_log: members disagree on PrimaryKey: keyed ('id',), other_key ('gene',)",
        'u_log.u: rocksdb needs one PrimaryKey column on the model, it has none',
    ]


@pytest.mark.parametrize('product', ['platform', 'ppp'])
def test_yaml_declarations_are_valid(product):
    entries = yaml.safe_load((ROOT / 'config' / 'datasets.yaml').read_text())['clickhouse']
    declared = [e for e in entries.values() if 'tables' in e or 'dictionaries' in e]
    assert declared
    assert [p for e in declared for p in entry_problems(e, product)] == []


def test_table_sql_rerun_keeps_one_copy_of_the_rows(clickhouse_database):
    client = clickhouse_database
    table = ClickHouseTable('ddl_rerun', 'mergetree', order_by=('id',), where='id != 3')
    cols = [('id', CHType('Int64')), ('name', STR)]
    client.command('DROP TABLE IF EXISTS ddl_rerun_log SYNC')
    client.command('CREATE TABLE ddl_rerun_log (id Int64, name String) ENGINE = Log')
    client.command("INSERT INTO ddl_rerun_log VALUES (1, 'a'), (2, 'b'), (3, 'c')")
    try:
        for _ in range(2):
            for statement in table_sql(table, cols, 'ddl_rerun_log'):
                client.command(statement)
        assert client.command('SELECT count() FROM ddl_rerun') == 2
    finally:
        client.command('DROP TABLE IF EXISTS ddl_rerun SYNC')
        client.command('DROP TABLE IF EXISTS ddl_rerun_log SYNC')


def test_dictionary_sql_rerun_reads_the_rebuilt_table(clickhouse_database):
    client = clickhouse_database
    table = ClickHouseTable('ddl_dict_src', 'rocksdb')
    cols = [('k', STR), ('v', STR)]
    d = ClickHouseDictionary(
        'ddl_dict', 'ddl_dict_src', key='id', attributes=('name',), source_columns={'id': 'k', 'name': 'v'}
    )
    client.command('DROP TABLE IF EXISTS ddl_dict_log SYNC')
    client.command('CREATE TABLE ddl_dict_log (k String, v String) ENGINE = Log')
    client.command("INSERT INTO ddl_dict_log VALUES ('x', 'nx')")
    try:
        for _ in range(2):
            client.command(drop_dictionary_sql(d))
            for statement in table_sql(table, cols, 'ddl_dict_log', key='k'):
                client.command(statement)
            client.command(dictionary_sql(d, cols))
        assert client.command("SELECT dictGet('ddl_dict', 'name', tuple('x'))") == 'nx'
    finally:
        client.command('DROP DICTIONARY IF EXISTS ddl_dict')
        client.command('DROP TABLE IF EXISTS ddl_dict_src SYNC')
        client.command('DROP TABLE IF EXISTS ddl_dict_log SYNC')
