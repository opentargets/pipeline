"""The croissant record sets and file sets are built from the dataset models."""

import json
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, ClassVar, Literal

import mlcroissant as mlc
import polars as pl
import pytest
from loguru import logger
from ot_datamodels import Bioregistry, DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from ot_datamodels.discovery import discover_datasets
from ot_datamodels.spec import dataset_spec, walk
from pydantic import Field

from pos.croissant import schema
from pos.croissant.constants import TYPE_DICT
from pos.croissant.crumbs.metadata import PlatformOutputMetadata
from pos.croissant.crumbs.record_sets import PlatformOutputRecordSets, data_type


class Location(StructModel):
    chromosome: str = Field(description='Chromosome')
    start: Annotated[int, PolarsDtype(pl.Int32)] = Field(description='Start position')


class Xref(StructModel):
    id: Annotated[str, Bioregistry('pubmed')] = Field(description='Cross-reference id')
    source: str = Field(description='Cross-reference source')


class Thing(DatasetModel):
    dataset_name: ClassVar[str] = 'thing'
    dataset_title: ClassVar[str] = 'Thing'
    dataset_description: ClassVar[str] = 'Things for the tests'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target', 'Disease')
    example: ClassVar[dict] = {
        'id': 'T1',
        'version': 1,
        'kind': 'a',
        'score': 0.5,
        'released': '2026-12-10',
        'flag': True,
        'otherIds': ['O1'],
        'tags': ['x'],
        'location': {'chromosome': '1', 'start': 10},
        'xrefs': [{'id': '1', 'source': 's'}],
        'grid': [[1]],
    }

    id: Annotated[str, Bioregistry('ensembl'), PrimaryKey()] = Field(description='Thing id')
    version: Annotated[int, PrimaryKey(), PolarsDtype(pl.Int16)] = Field(description='Thing version')
    kind: Annotated[Literal['a', 'b'], PolarsDtype(pl.Enum)] = Field(description='Kind of thing')
    score: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(description='Score')
    released: date | None = Field(description='Release date')
    flag: bool = Field(description='Flag')
    otherIds: list[Annotated[str, Reference('other', 'id'), Bioregistry('ensembl')]] = Field(  # noqa: N815
        description='Other ids'
    )
    tags: list[str] = Field(description='Tags')
    location: Location = Field(description='Location')
    xrefs: list[Xref] = Field(description='Cross-references')
    grid: list[list[Annotated[int, PolarsDtype(pl.Int8)]]] = Field(description='Grid')


class Other(DatasetModel):
    dataset_name: ClassVar[str] = 'other'
    dataset_title: ClassVar[str] = 'Other'
    dataset_description: ClassVar[str] = 'Others'
    example: ClassVar[dict] = {'id': 'O1', 'thingId': 'T1'}

    id: Annotated[str, PrimaryKey()] = Field(description='Other id')
    thingId: Annotated[str, Reference('thing', 'id')] = Field(description='Thing it belongs to')  # noqa: N815


MODELS = {'thing': Thing, 'other': Other}


@pytest.fixture
def release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A release folder with `thing` (flat), `other` (hive-partitioned) and `stray` (no model)."""
    monkeypatch.setattr(schema, 'discover_datasets', lambda: MODELS)
    output = tmp_path / 'output'
    for folder in ('thing', 'other/source=a', 'other/source=b', 'stray'):
        (output / folder).mkdir(parents=True)
    (output / 'thing' / 'part-00000.parquet').write_text('')
    return [str(output / name) for name in ('thing', 'other', 'stray')]


@pytest.fixture
def warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    handler = logger.add(lambda m: messages.append(m.record['message']), level='WARNING')
    yield messages
    logger.remove(handler)


def _metadata(datasets: list[str]) -> PlatformOutputMetadata:
    return PlatformOutputMetadata(
        datasets=datasets,
        ftp_location='https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.12/output/',
        gcp_location='gs://open-targets-data-releases/26.12/output/',
        aws_location=None,
        version='26.12',
        date_published=datetime(2026, 12, 10),
    )


def _fields(record_set: mlc.RecordSet) -> dict[str, mlc.Field]:
    out: dict[str, mlc.Field] = {}
    pending = list(record_set.fields)
    while pending:
        field = pending.pop(0)
        out[field.id] = field
        pending.extend(field.sub_fields or [])
    return out


def _thing() -> mlc.RecordSet:
    return PlatformOutputRecordSets().add_datasets([dataset_spec(Thing)]).get_metadata()[0]


def test_record_set_is_the_model() -> None:
    record_set = _thing()
    assert record_set.id == record_set.name == 'thing'
    assert record_set.description == 'Things for the tests [Target, Disease]'
    assert record_set.key == ['thing/id', 'thing/version']
    assert [f.name for f in record_set.fields] == list(Thing.model_fields)


def test_field_types_follow_the_polars_dtypes() -> None:
    fields = _fields(_thing())
    assert fields['thing/id'].data_types == [mlc.DataType.TEXT]
    assert fields['thing/version'].data_types == [mlc.DataType.INT16]
    assert fields['thing/score'].data_types == [mlc.DataType.FLOAT32]
    assert fields['thing/flag'].data_types == [mlc.DataType.BOOL]
    assert fields['thing/released'].data_types == [mlc.DataType.DATE]
    assert fields['thing/location/start'].data_types == [mlc.DataType.INT32]


def test_enum_is_text() -> None:
    assert _fields(_thing())['thing/kind'].data_types == [mlc.DataType.TEXT]


def test_every_model_dtype_has_a_croissant_type() -> None:
    for model in discover_datasets().values():
        for path, node in walk(dataset_spec(model)):
            if node.kind == 'primitive':
                assert data_type(node) in TYPE_DICT.values(), f'{model.dataset_name}/{path}'


def test_descriptions_carry_the_bioregistry_prefix() -> None:
    fields = _fields(_thing())
    assert fields['thing/id'].description == 'Thing id [bioregistry:ensembl]'
    assert fields['thing/otherIds'].description == 'Other ids [bioregistry:ensembl]'
    assert fields['thing/xrefs/id'].description == 'Cross-reference id [bioregistry:pubmed]'
    assert fields['thing/flag'].description == 'Flag'


def test_list_struct_and_list_of_list_shapes(warnings: list[str]) -> None:
    fields = _fields(_thing())
    assert fields['thing/tags'].repeated is True
    assert fields['thing/tags'].data_types == [mlc.DataType.TEXT]
    assert [s.id for s in fields['thing/location'].sub_fields] == ['thing/location/chromosome', 'thing/location/start']
    assert fields['thing/location'].data_types == []
    assert fields['thing/location/start'].source.extract.column == 'location/start'
    assert fields['thing/xrefs'].repeated is True
    assert [s.name for s in fields['thing/xrefs'].sub_fields] == ['id', 'source']
    assert fields['thing/xrefs/source'].source.file_set == 'thing-fileset'
    assert fields['thing/grid'].repeated is True
    assert fields['thing/grid'].data_types == [mlc.DataType.INT8]
    assert any('grid is of type array of array' in m for m in warnings)


def test_foreign_keys_come_from_references() -> None:
    fields = _fields(_thing())
    assert fields['thing/otherIds'].references.field == 'other/id'
    other = PlatformOutputRecordSets().add_datasets([dataset_spec(Other)]).get_metadata()[0]
    assert _fields(other)['other/thingId'].references.field == 'thing/id'
    assert not fields['thing/tags'].references.field


def test_file_sets_name_the_model_and_follow_the_partitions(release: list[str]) -> None:
    jsonld = _metadata(release).to_json()
    filesets = {node['@id']: node for node in jsonld['distribution'] if node['@type'] == 'cr:FileSet'}
    assert filesets['thing-fileset']['name'] == 'Thing'
    assert filesets['thing-fileset']['includes'] == 'thing/*.parquet'
    assert filesets['other-fileset']['name'] == 'Other'
    assert filesets['other-fileset']['includes'] == 'other/**/*.parquet'
    assert filesets['other-fileset']['description'].endswith('partitioned by source')


def test_a_folder_with_no_model_is_skipped_with_a_warning(release: list[str], warnings: list[str]) -> None:
    jsonld = _metadata(release).to_json()
    assert [rs['@id'] for rs in jsonld['recordSet']] == ['thing', 'other']
    assert 'stray-fileset' not in {node['@id'] for node in jsonld['distribution']}
    assert schema.NO_MODEL_WARNING.format(dataset='stray') in warnings


def test_the_release_validates_with_mlcroissant(release: list[str], tmp_path: Path) -> None:
    path = tmp_path / 'croissant.json'
    path.write_text(json.dumps(_metadata(release).to_json(), default=str))
    reloaded = mlc.Dataset(str(path)).metadata
    assert not reloaded.ctx.issues.errors


def test_a_field_without_a_description_fails() -> None:
    class Bare(DatasetModel):
        dataset_name: ClassVar[str] = 'bare'
        dataset_title: ClassVar[str] = 'Bare'
        dataset_description: ClassVar[str] = 'Bare'
        example: ClassVar[dict] = {'id': 'x'}

        id: str

    with pytest.raises(ValueError, match='bare/id has no description'):
        PlatformOutputRecordSets().add_datasets([dataset_spec(Bare)])


# Every output model: the descriptor a full release gets.


@pytest.fixture(scope='module')
def corpus() -> dict[str, mlc.RecordSet]:
    specs = [dataset_spec(model) for _, model in sorted(discover_datasets().items())]
    return {rs.id: rs for rs in PlatformOutputRecordSets().add_datasets(specs).get_metadata()}


def test_every_output_has_a_record_set(corpus: dict[str, mlc.RecordSet]) -> None:
    assert set(corpus) == set(discover_datasets())


def test_every_field_has_a_description(corpus: dict[str, mlc.RecordSet]) -> None:
    empty = [fid for rs in corpus.values() for fid, f in _fields(rs).items() if not f.description]
    assert not empty
    assert not [fid for rs in corpus.values() for fid, f in _fields(rs).items() if 'PLACEHOLDER' in str(f.description)]


def test_keys_are_the_model_primary_keys(corpus: dict[str, mlc.RecordSet]) -> None:
    for name, model in discover_datasets().items():
        expected = [f'{name}/{column}' for column in dataset_spec(model).primary_key]
        assert (corpus[name].key or []) == expected, name


def test_foreign_keys_point_at_existing_fields(corpus: dict[str, mlc.RecordSet]) -> None:
    ids = {fid for rs in corpus.values() for fid in _fields(rs)}
    references = {
        fid: f.references.field for rs in corpus.values() for fid, f in _fields(rs).items() if f.references.field
    }
    assert references
    assert {fid: ref for fid, ref in references.items() if ref not in ids} == {}


def test_references_to_a_credible_set_point_at_its_id(corpus: dict[str, mlc.RecordSet]) -> None:
    for field in (
        'credible_set_variant/credibleSetId',
        'colocalisation/leftLocusId',
        'colocalisation/rightLocusId',
        'l2g_prediction/credibleSetId',
        'evidence_gwas_credible_sets/studyLocusId',
    ):
        assert _fields(corpus[field.split('/')[0]])[field].references.field == 'credible_set/id'


def test_a_full_release_validates_with_mlcroissant(tmp_path: Path) -> None:
    output = tmp_path / 'output'
    for name in discover_datasets():
        (output / name).mkdir(parents=True)
    path = tmp_path / 'croissant.json'
    datasets = sorted(str(p) for p in output.iterdir())
    path.write_text(json.dumps(_metadata(datasets).to_json(), default=str))
    reloaded = mlc.Dataset(str(path)).metadata
    assert not reloaded.ctx.issues.errors
    assert len(reloaded.record_sets) == len(discover_datasets())
