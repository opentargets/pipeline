"""Tests for the interaction transformer."""

import json
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from pts.transformers.interaction import (
    INTACT_SCHEMA,
    aggregate,
    divide,
    explode,
    generate_interactions,
    get_code,
    interaction,
    interaction_id,
    to_camel_case,
    transform_ensembl_protein,
    transform_human_mapping,
    transform_intact,
    transform_rnacentral,
    transform_string,
    with_interaction_id,
)
from pts.transformers.utils.dataset import scan_dataset

SPECIES = pl.Struct({'mnemonic': pl.String, 'scientificName': pl.String, 'taxonId': pl.Int64})
HUMAN = {'mnemonic': 'human', 'scientificName': 'Homo sapiens', 'taxonId': 9606}
MOUSE = {'mnemonic': 'mouse', 'scientificName': 'Mus musculus', 'taxonId': 10090}
STRING_COLUMNS = [
    'protein1',
    'protein2',
    'neighborhood',
    'fusion',
    'cooccurence',
    'coexpression',
    'experimental',
    'database',
    'textmining',
    'combined_score',
    'homology',
]


def organism(mnemonic: str = 'human', name: str = 'Homo sapiens', taxon: int = 9606) -> dict:
    return {'mnemonic': mnemonic, 'scientific_name': name, 'taxon_id': taxon}


def interactor(id_: str, org: dict | None = None) -> dict:
    return {'biological_role': 'enzyme', 'id': id_, 'id_source': 'uniprotkb', 'organism': org or organism()}


def intact_record(a: dict, b: dict | None, score: float | None, evidence: list | None, source='intact') -> dict:
    return {
        'interaction': {'causal_interaction': None, 'evidence': evidence, 'interaction_score': score},
        'interactorA': a,
        'interactorB': b,
        'source_info': {'database_version': '1', 'source_database': source},
    }


def intact_evidence(identifier: str) -> dict:
    return {'interaction_identifier': identifier, 'pubmed_id': '1', 'participant_detection_method_A': []}


def intact_frame(records: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(records, schema=INTACT_SCHEMA)


def key_frame(**overrides) -> pl.DataFrame:
    row = {
        'sourceDatabase': 'intact',
        'targetA': 'ENSG00000001',
        'intA': 'P11111',
        'intABiologicalRole': 'enzyme',
        'targetB': 'ENSG00000002',
        'intB': 'P22222',
        'intBBiologicalRole': 'enzyme target',
        'speciesA': HUMAN,
        'speciesB': HUMAN,
    } | overrides
    return pl.DataFrame([row], schema_overrides={'speciesA': SPECIES, 'speciesB': SPECIES, 'targetB': pl.String})


def string_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        [{c: r.get(c, '0') for c in STRING_COLUMNS} for r in rows], schema=dict.fromkeys(STRING_COLUMNS, pl.String)
    )


# ---------------------------------------------------------------------------
# polars vs spark traps
# ---------------------------------------------------------------------------


def test_divide_matches_ieee_division() -> None:
    values = [288, 348, 1, 999, 1000, 0, None]
    out = pl.DataFrame({'x': values}).select(divide(pl.col('x'), 1000))['x'].to_list()
    assert out == [None if v is None else v / 1000 for v in values]


def test_camel_case_renames_nested_structs_without_nulling_them() -> None:
    df = pl.DataFrame([
        {
            'host_organism_tissue': {'full_name': 'liver', 'xrefs': ['a']},
            'participant_detection_method_A': [{'mi_identifier': 'MI:1', 'short_name': 'x'}],
        }
    ])
    out = to_camel_case(df)
    assert out.columns == ['hostOrganismTissue', 'participantDetectionMethodA']
    assert out.row(0, named=True) == {
        'hostOrganismTissue': {'fullName': 'liver', 'xrefs': ['a']},
        'participantDetectionMethodA': [{'miIdentifier': 'MI:1', 'shortName': 'x'}],
    }


def test_interactions_handle_a_multi_chunk_frame_with_missing_interactor_b() -> None:
    first = intact_frame([intact_record(interactor('P1'), None, 0.5, [intact_evidence('e1')])])
    second = intact_frame([
        intact_record(interactor('P2'), interactor('P3', organism('mouse', 'Mus musculus', 10090)), 0.5, []),
        intact_record(interactor('P4', organism('mouse', 'Mus musculus', 10090)), {'id': 'P5'}, 0.5, []),
    ])
    df = pl.concat([first, second], rechunk=False)
    assert df.n_chunks() > 1
    out = generate_interactions(df, pl.DataFrame(schema={'gene_id': pl.String, 'mapped_id': pl.String}), pl.lit(1))
    forward = {r['intA']: r for r in out.to_dicts() if r['intA'] in {'P1', 'P2', 'P4'}}
    assert forward['P1']['intB'] == 'P1'
    assert forward['P1']['speciesB'] == HUMAN
    assert forward['P2']['speciesB'] == MOUSE
    assert forward['P4']['intB'] == 'P5'
    assert forward['P4']['speciesB'] == MOUSE


def test_explode_drops_null_and_empty_lists_but_keeps_null_elements() -> None:
    df = pl.DataFrame({'k': [1, 2, 3, 4], 'v': [[1, 2], [], None, [None]]})
    out = explode(df, 'v')
    assert out.rows() == [(1, 1), (1, 2), (4, None)]


def test_intact_without_evidence_is_dropped() -> None:
    df = intact_frame([
        intact_record(interactor('P1'), interactor('P1'), 0.5, [intact_evidence('e1')]),
        intact_record(interactor('P2'), interactor('P2'), 0.5, []),
        intact_record(interactor('P3'), interactor('P3'), 0.5, None),
    ])
    out = transform_intact(df, pl.DataFrame(schema={'gene_id': pl.String, 'mapped_id': pl.String}))
    assert out['intA'].to_list() == ['P1']


def test_ensembl_protein_without_a_match_is_an_empty_string() -> None:
    gtf = pl.DataFrame({
        '_c2': ['CDS', 'CDS', 'exon'],
        '_c8': [
            'gene_id "ENSG00000000001"; protein_id "ENSP00000000001";',
            'gene_id "ENSG00000000002";',
            'gene_id "ENSG00000000003"; protein_id "ENSP00000000003";',
        ],
    })
    out = transform_ensembl_protein(gtf).sort('gene_id')
    assert out.rows() == [('ENSG00000000001', 'ENSP00000000001'), ('ENSG00000000002', '')]


def test_get_code_strips_spaces_only() -> None:
    df = pl.DataFrame({'x': ['  URS123-2_992 ', 'P1_x', '\tP2-1', '', None]})
    assert df.select(get_code(pl.col('x')))['x'].to_list() == ['URS123', 'P1', '\tP2', '', None]


def test_human_mapping_collects_ensembl_ids_without_nulls() -> None:
    df = pl.DataFrame({
        '_c0': ['P1', None, 'P2', 'BRCA1', None],
        '_c1': ['Ensembl', 'Ensembl', 'Ensembl', 'Gene_Name', 'Ensembl'],
        '_c2': ['ENSG1', 'ENSG1', 'ENSG1', 'ENSG1', 'ENSG2'],
    })
    out = {r['id']: r['mapping_list'] for r in transform_human_mapping(df).to_dicts()}
    assert out == {'ENSG1': ['P1', 'P2'], 'ENSG2': []}


def test_intact_struct_fields_are_alphabetical(tmp_path: Path) -> None:
    def check(dtype: pl.DataType) -> None:
        if isinstance(dtype, pl.Struct):
            names = [f.name for f in dtype.fields]
            assert names == sorted(names)
            for f in dtype.fields:
                check(f.dtype)
        elif isinstance(dtype, pl.List):
            check(dtype.inner)

    for dtype in INTACT_SCHEMA.values():
        check(dtype)

    record = {
        'source_info': {'source_database': 'intact', 'database_version': '1'},
        'interactorB': {'organism': {'taxon_id': 9606, 'mnemonic': 'human'}, 'id': 'P2'},
        'interactorA': {'id': 'P1', 'biological_role': 'enzyme'},
        'interaction': {'interaction_score': 0.5, 'evidence': [{'pubmed_id': '7', 'expansion_method_short_name': 's'}]},
    }
    path = tmp_path / 'intact.json'
    path.write_text(json.dumps(record) + '\n')
    row = scan_dataset(str(path), format='ndjson', schema=INTACT_SCHEMA).collect().row(0, named=True)
    assert row['source_info'] == {'database_version': '1', 'source_database': 'intact'}
    assert row['interactorB']['organism'] == {'mnemonic': 'human', 'scientific_name': None, 'taxon_id': 9606}
    assert row['interaction']['evidence'][0]['pubmed_id'] == '7'
    assert row['interaction']['evidence'][0]['expansion_method_short_name'] == 's'


# ---------------------------------------------------------------------------
# behaviour
# ---------------------------------------------------------------------------


def test_rnacentral_maps_columns() -> None:
    df = pl.DataFrame({f'_c{i}': [v] for i, v in enumerate(['URS1', '9606', 'x', 'y', 'z', 'ENSG1'])})
    assert transform_rnacentral(df).row(0, named=True) == {'gene_id': 'ENSG1', 'mapped_id': 'URS1'}


def test_string_filters_scores_and_keeps_positive_channels() -> None:
    raw = string_frame([
        {
            'protein1': '9606.ENSP1',
            'protein2': '9606.ENSP2',
            'combined_score': '500',
            'experimental': '288',
            'textmining': '100',
        },
        {'protein1': '9606.ENSP3', 'protein2': '9606.ENSP4', 'combined_score': '200', 'experimental': '100'},
        {'protein1': '10090.ENSP5', 'protein2': '9606.ENSP6', 'combined_score': '900', 'experimental': '100'},
    ])
    mapping = pl.DataFrame({'gene_id': ['ENSG1', 'ENSG2'], 'mapped_id': ['ENSP1', 'ENSP2']})
    out = transform_string(raw, mapping, score_threshold=400, version='12.0')
    assert out.height == 2
    assert set(out['intA']) == {'ENSP1'}
    assert set(out['targetB']) == {'ENSG2'}
    assert set(out['interactionScore']) == {0.5}
    assert out['interactionResources'].to_list()[0] == {'database_version': '12.0', 'source_database': 'string'}
    channels = {e['interaction_detection_method_short_name']: e for e in out['evidences'].to_list()}
    assert set(channels) == {'experimental', 'textmining'}
    assert channels['experimental']['evidence_score'] == 288
    assert channels['experimental']['interaction_detection_method_mi_identifier'] == 'MI:0591'


def test_bidirectional_sources_add_the_reverse_direction_except_self_interactions() -> None:
    df = intact_frame([
        intact_record(interactor('P1'), interactor('P2'), 0.5, [intact_evidence('e1')]),
        intact_record(interactor('P3'), interactor('P3'), 0.5, [intact_evidence('e2')]),
        intact_record(interactor('P4'), interactor('P5'), 0.5, [intact_evidence('e3')], source='biogrid'),
    ])
    mapping = pl.DataFrame({'gene_id': ['ENSG1', 'ENSG2'], 'mapped_id': ['P1', 'P2']})
    out = transform_intact(df, mapping)
    pairs = sorted((r['intA'], r['intB'], r['targetA'], r['targetB']) for r in out.to_dicts())
    assert pairs == [
        ('P1', 'P2', 'ENSG1', 'ENSG2'),
        ('P2', 'P1', 'ENSG2', 'ENSG1'),
        ('P3', 'P3', None, None),
        ('P4', 'P5', None, None),
    ]


def test_aggregate_counts_evidence_and_takes_the_maximum_score() -> None:
    key = key_frame()
    rows = pl.concat([key, key, key]).with_columns(interactionScore=pl.Series([0.44, 0.74, None]))
    out = aggregate(rows)
    assert out.height == 1
    assert out.row(0, named=True)['count'] == 3
    assert out.row(0, named=True)['scoring'] == 0.74


def test_interaction_id_is_a_signed_64_bit_integer() -> None:
    out = with_interaction_id(key_frame())
    assert out.schema['interactionId'] == pl.Int64


@pytest.mark.parametrize(
    'overrides',
    [{'speciesA': MOUSE}, {'targetB': None}, {'targetB': ''}, {'intBBiologicalRole': 'enzyme'}],
)
def test_interaction_id_distinguishes_every_key_column(overrides: dict) -> None:
    base = with_interaction_id(key_frame())['interactionId'][0]
    assert with_interaction_id(key_frame(**overrides))['interactionId'][0] != base


def test_interaction_id_is_pinned() -> None:
    """XXH64 (seed 0) of the key columns joined by \\x1f, nulls as \\x00, structs flattened."""
    df = pl.concat([key_frame(), key_frame(sourceDatabase='string', targetB=None, speciesB=MOUSE)])
    assert df.select(interaction_id(df.schema))['interactionId'].to_list() == [
        -4777012063784892725,
        3524803122290032628,
    ]


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def test_interaction_end_to_end(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    src.mkdir()
    pl.DataFrame(
        {
            'id': ['ENSG1', 'ENSG2'],
            'proteinIds': [[{'id': 'P1', 'source': 'uniprot'}], [{'id': 'P2', 'source': 'uniprot'}]],
            'dbXrefs': [[{'id': '1', 'source': 'HGNC'}], None],
        },
    ).write_parquet(src / 'target.parquet')
    (src / 'rna.tsv').write_text('URS1\t9606\tx\ty\tz\tENSG2\n')
    (src / 'hm.dat').write_text('P1\tEnsembl\tENSG1\nP2\tEnsembl\tENSG2\n')
    (src / 'gtf').write_text(
        '#!genome\n1\tens\tCDS\t1\t2\t.\t+\t0\tgene_id "ENSG00000000001"; protein_id "ENSP00000000001";\n'
        '1\tens\tCDS\t1\t2\t.\t+\t0\tgene_id "ENSG00000000002"; protein_id "ENSP00000000002";\n'
    )
    records = [
        intact_record(interactor('P1'), interactor('P2'), 0.5, [intact_evidence('e1'), intact_evidence('e2')]),
        intact_record(interactor('P1'), interactor('Q9', organism()), 0.4, [intact_evidence('e3')]),
    ]
    (src / 'intact.json').write_text('\n'.join(json.dumps(r) for r in records) + '\n')
    (src / 'string.txt').write_text(
        ' '.join(STRING_COLUMNS) + '\n9606.ENSP00000000001 9606.ENSP00000000002 0 0 0 0 288 0 100 500 0\n'
    )
    dst = tmp_path / 'dst'
    interaction(
        {
            'targets': str(src / 'target.parquet'),
            'rnacentral': str(src / 'rna.tsv'),
            'humanmapping': str(src / 'hm.dat'),
            'ensproteins': str(src / 'gtf'),
            'intact': str(src / 'intact.json'),
            'strings': str(src / 'string.txt'),
        },
        {
            'interactions': str(dst / 'interaction'),
            'interactions_evidence': str(dst / 'interaction_evidence'),
            'interactions_unmatched': str(dst / 'unmatched'),
        },
        {'string_version': '12.0'},
        None,
    )
    interactions = pl.read_parquet(dst / 'interaction')
    evidences = pl.read_parquet(dst / 'interaction_evidence')
    unmatched = pl.read_parquet(dst / 'unmatched')

    by_pair = {(r['sourceDatabase'], r['intA'], r['intB']): r for r in interactions.to_dicts()}
    assert set(by_pair) == {
        ('intact', 'P1', 'P2'),
        ('intact', 'P2', 'P1'),
        ('intact', 'P1', 'Q9'),
        ('string', 'ENSP00000000001', 'ENSP00000000002'),
    }
    assert by_pair['intact', 'P1', 'P2']['count'] == 2
    assert by_pair['string', 'ENSP00000000001', 'ENSP00000000002']['count'] == 2
    assert by_pair['string', 'ENSP00000000001', 'ENSP00000000002']['scoring'] == 0.5
    assert unmatched['intB'].to_list() == ['Q9']

    assert evidences.height == interactions['count'].sum()
    assert evidences.join(interactions, on='interactionId', how='anti').height == 0
    string_scores = evidences.filter(pl.col('interactionResources').struct.field('sourceDatabase') == 'string')
    assert sorted(string_scores['evidenceScore'].to_list()) == [0.1, 0.288]

    schema = pq.read_schema(next((dst / 'interaction').glob('*.parquet')))
    assert {f.name for f in schema if not f.nullable} == {'interactionId', 'count', 'speciesA', 'speciesB'}
    schema = pq.read_schema(next((dst / 'interaction_evidence').glob('*.parquet')))
    assert {f.name for f in schema if not f.nullable} == {'interactionId'}
    assert evidences.columns[:7] == [
        'interactionResources',
        'interactionScore',
        'interactionDetectionMethodShortName',
        'interactionDetectionMethodMiIdentifier',
        'evidenceScore',
        'interactionIdentifier',
        'pubmedId',
    ]
