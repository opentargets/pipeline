"""Tests for the target_tractability transformer."""

from pathlib import Path

import polars as pl

from pts.transformers.target_tractability import build_tractability, target_tractability

SM = 'SM_B1_existence_sm_tractability_bucket'
AB = 'AB_B1_existence_ab_tractability_bucket'


def tract(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={'ensembl_gene_id': pl.String, SM: pl.String, AB: pl.String}, orient='row')


def targets(ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame({'id': ids}, schema={'id': pl.String})


def test_build_tractability_output_columns():
    result = build_tractability(tract([('ENSG1', '0', '0')]), targets(['ENSG1']))
    assert result.schema == {'targetId': pl.String, 'modality': pl.String, 'category': pl.String, 'value': pl.Boolean}
    assert result.columns == ['targetId', 'modality', 'category', 'value']


def test_build_tractability_one_row_per_assessment():
    assert build_tractability(tract([('ENSG1', '0', '0')]), targets(['ENSG1'])).height == 2


def test_build_tractability_value_true_when_bucket_is_1():
    result = build_tractability(tract([('ENSG1', '1', '0')]), targets(['ENSG1']))
    assert result.filter(modality='SM')['value'].to_list() == [True]


def test_build_tractability_value_false_when_bucket_is_0():
    result = build_tractability(tract([('ENSG1', '0', '1')]), targets(['ENSG1']))
    assert result.filter(modality='SM')['value'].to_list() == [False]


def test_build_tractability_drops_unknown_target():
    assert build_tractability(tract([('ENSG9', '0', '0')]), targets(['ENSG1'])).height == 0


def test_build_tractability_keeps_known_target():
    result = build_tractability(tract([('ENSG1', '0', '0'), ('ENSG9', '0', '0')]), targets(['ENSG1']))
    assert result.height == 2
    assert set(result['targetId']) == {'ENSG1'}


def test_build_tractability_modality_extracted_from_column_name():
    result = build_tractability(tract([('ENSG1', '0', '0')]), targets(['ENSG1']))
    assert set(result['modality']) == {'SM', 'AB'}


def test_build_tractability_bucket_id_extracted_from_column_name():
    result = build_tractability(tract([('ENSG1', '0', '0')]), targets(['ENSG1']))
    assert 'bucket' in set(result['category'])


def test_value_is_numeric_one_else_false():
    cases = [
        ('1', True), ('1.0', True), ('0', False), ('', False), ('x', False), (None, False),
        (' 1', True), ('1 ', True), ('\t1', True), ('2', False), ('1.5', True), ('01', True),
        ('1e0', False), ('+1', True), ('-1', False), ('.5', False), ('1.', True), ('1.5.5', False),
        ('\x7f1', True), ('1\x7f', True), ('4294967297', False), ('9223372036854775808', False),
    ]  # fmt: skip
    ids = [f'G{i}' for i in range(len(cases))]
    raw = pl.DataFrame({'ensembl_gene_id': ids, 'SM_B1_Approved Drug': [c for c, _ in cases]})
    got = build_tractability(raw, pl.DataFrame({'id': ids}))
    values = dict(zip(got['targetId'], got['value'], strict=True))
    assert [values[i] for i in ids] == [v for _, v in cases]


def test_target_tractability_null_and_empty_lists():
    raw = pl.DataFrame({'ensembl_gene_id': ['ENSG1', None], 'other': ['a', 'b']})
    assert build_tractability(raw, targets(['ENSG1'])).height == 0
    assert build_tractability(tract([(None, '1', '1')]), targets(['ENSG1'])).height == 0
    assert build_tractability(tract([('ENSG1', None, None)]), targets([])).height == 0
    got = build_tractability(tract([('ENSG1', None, None)]), targets(['ENSG1']))
    assert got['value'].to_list() == [False, False]


def test_build_tractability_duplicate_target_ids_fan_out():
    assert build_tractability(tract([('ENSG1', '1', '1')]), targets(['ENSG1', 'ENSG1'])).height == 4


def test_target_tractability_end_to_end_is_deterministic(tmp_path: Path):
    tsv = tmp_path / 'tractability.tsv'
    tsv.write_text(f'ensembl_gene_id\t{SM}\t{AB}\tnote\nENSG2\t1\t0\tx\nENSG1\t0\t1.0\ty\nENSG9\t1\t1\tz\n')
    target = tmp_path / 'target'
    target.mkdir()
    a = pl.DataFrame({'id': ['ENSG1']})
    b = pl.DataFrame({'id': ['ENSG2']})
    pl.concat([a, b], rechunk=False).write_parquet(target / 'part.parquet')
    source = {'tractability': str(tsv), 'target': str(target)}

    outputs = []
    for name in ('one', 'two'):
        destination = tmp_path / name
        target_tractability(source, str(destination), None, None)
        outputs.append(sorted((p.name, p.read_bytes()) for p in destination.glob('*.parquet')))
    assert outputs[0] == outputs[1]

    result = pl.read_parquet(tmp_path / 'one')
    assert result.rows() == [
        ('ENSG1', 'SM', 'bucket', False),
        ('ENSG1', 'AB', 'bucket', True),
        ('ENSG2', 'SM', 'bucket', True),
        ('ENSG2', 'AB', 'bucket', False),
    ]


def test_build_tractability_keeps_bucket_order_within_target():
    raw = pl.DataFrame({
        'ensembl_gene_id': ['ENSG2', 'ENSG1'],
        'SM_B1_Approved Drug': ['1', '0'],
        'SM_B2_Advanced Clinical': ['0', '1'],
        'AB_B1_Approved Drug': ['0', '0'],
        'PR_B1_Literature': ['1', '1'],
    })
    got = build_tractability(raw, targets(['ENSG1', 'ENSG2']))
    buckets = [('SM', 'Approved Drug'), ('SM', 'Advanced Clinical'), ('AB', 'Approved Drug'), ('PR', 'Literature')]
    assert got.select('targetId', 'modality', 'category').rows() == [
        (t, m, c) for t in ('ENSG1', 'ENSG2') for m, c in buckets
    ]


def test_build_tractability_multi_chunk_input():
    raw = pl.concat([tract([('ENSG1', '1', '0')]), tract([('ENSG2', '0', '1')])], rechunk=False)
    one = build_tractability(raw, targets(['ENSG1', 'ENSG2']))
    assert one.equals(build_tractability(raw.rechunk(), targets(['ENSG1', 'ENSG2'])))
