"""FASTA ingestion produces a bounded, base-level reference dataset."""

from unittest.mock import Mock

import polars as pl
import pytest
from otter.config.model import Config

from pts.transformers.reference import reference


@pytest.fixture
def doctest_namespace():
    return {}


def test_fasta_to_parquet_preserves_positions_across_lines_and_batches(tmp_path):
    source = tmp_path / 'reference.fasta'
    source.write_text('>chr1 description\nacG\nTN\n\n>chrM\ntA\n>chr19_KI270938v1_alt\nGC\n')
    destination = tmp_path / 'reference'

    reference(str(source), str(destination), {'batch_size': 2}, Mock(spec=Config))

    parts = sorted(destination.glob('*.parquet'))
    assert len(parts) > 1
    assert all(pl.read_parquet(part).height <= 2 for part in parts)
    result = pl.read_parquet(parts).sort('chr', 'pos')
    assert result.schema == {'chr': pl.String, 'pos': pl.Int64, 'ref': pl.String}
    assert result.rows() == [
        ('1', 1, 'A'),
        ('1', 2, 'C'),
        ('1', 3, 'G'),
        ('1', 4, 'T'),
        ('1', 5, 'N'),
        ('19_KI270938v1_alt', 1, 'G'),
        ('19_KI270938v1_alt', 2, 'C'),
        ('MT', 1, 'T'),
        ('MT', 2, 'A'),
    ]


@pytest.mark.parametrize(
    'contents',
    [
        '',
        '>chr1\n',
        'ACGT\n',
        '>\nAC\n',
        '>chr1\nAC\n>1\nGT\n',
        '>chr1\nAC?T\n',
        '>chr1\n>chr2\nAC\n',
    ],
)
def test_invalid_fasta_is_rejected(tmp_path, contents):
    source = tmp_path / 'reference.fasta'
    source.write_text(contents)
    with pytest.raises(ValueError):
        reference(str(source), str(tmp_path / 'reference'), {}, Mock(spec=Config))


def test_existing_reference_is_not_overwritten(tmp_path):
    source = tmp_path / 'reference.fasta'
    source.write_text('>chr1\nAC\n')
    destination = tmp_path / 'reference'
    reference(str(source), str(destination), {}, Mock(spec=Config))
    with pytest.raises(ValueError, match='already exists'):
        reference(str(source), str(destination), {}, Mock(spec=Config))
    assert pl.read_parquet(destination / '*.parquet')['ref'].to_list() == ['A', 'C']
