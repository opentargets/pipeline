"""Tests for GERP and AlphaMissense as columns of the consequences frame."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl
import pyBigWig  # ty: ignore[unresolved-import] -- compiled module without stubs
import pytest

from pts.transformers.variant import predictions
from pts.transformers.variant.alleles import vep_allele_form
from pts.transformers.variant.predictions import add_gerp_and_alphamissense
from pts.transformers.variant.records import CONSEQUENCES_SCHEMA
from tests.variant_fixtures import consequences

SAMPLE = Path(__file__).parent / 'test_data' / 'alphamissense_sample.tsv'


@pytest.fixture(scope='module')
def tsv(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp('alphamissense') / 'AlphaMissense_hg38.tsv.gz'
    path.write_bytes(gzip.compress(SAMPLE.read_bytes()))
    return str(path)


@pytest.fixture(scope='module')
def bigwig(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A bigWig of contig 1 that scores 945517 to 945519 and the two bases around 946175."""
    path = str(tmp_path_factory.mktemp('gerp') / 'gerp.bw')
    handle = pyBigWig.open(path, 'w')
    handle.addHeader([('1', 1_000_000)])
    scores = {945517: -0.466, 945518: -1.75, 945519: -1.87, 946174: 0.12349, 946175: 0.0625}
    handle.addEntries(
        ['1'] * len(scores),
        [position - 1 for position in scores],
        ends=list(scores),
        values=list(scores.values()),
    )
    handle.close()
    return path


def consequence_rows(*rows: tuple[int, int, str, str, str]) -> pl.DataFrame:
    """One consequence per (row, position, reference, alternate, transcript id), in VEP's coordinates."""
    row, position, reference, alternate, transcript = (list(column) for column in zip(*rows, strict=True))
    alleles = vep_allele_form(
        pl.DataFrame({'position': position, 'referenceAllele': reference, 'alternateAllele': alternate}).lazy()
    ).collect()
    return consequences(
        row=row,
        position=position,
        referenceAllele=reference,
        alternateAllele=alternate,
        transcript_id=transcript,
        **{name: alleles[name].to_list() for name in ('vepStart', 'vepEnd', 'vepReference', 'vepAlternate')},
    )


def effects(found: pl.DataFrame) -> list[tuple]:
    return [
        (tc['transcript_id'], tc['conservation'], tc['am_class'], tc['am_pathogenicity'])
        for tc in found.iter_rows(named=True)
    ]


def test_fill_conservation_and_alphamissense(bigwig: str, tsv: str):
    found = add_gerp_and_alphamissense(
        consequence_rows(
            (0, 945518, 'C', 'A', 'ENST00000327044'),
            (0, 945518, 'C', 'A', 'ENST00000999999'),
            (1, 946175, 'G', 'A', 'ENST00000327044'),
        ),
        bigwig,
        tsv,
        '1',
    )
    assert effects(found) == [
        ('ENST00000327044', -1.75, 'likely_benign', 0.1719),
        ('ENST00000999999', -1.75, None, None),
        ('ENST00000327044', 0.062, 'likely_benign', 0.1618),
    ]


def test_fill_deletion_and_insertion_spans(bigwig: str, tsv: str):
    found = add_gerp_and_alphamissense(
        consequence_rows(
            (0, 945517, 'TCA', 'T', 'ENST00000327044'),
            (1, 945518, 'C', 'CT', 'ENST00000327044'),
        ),
        bigwig,
        tsv,
        '1',
    )
    # the deletion removes 945518 and 945519; the insertion sits between 945518 and 945519
    assert [(c, a) for _, c, a, _ in effects(found)] == [(-1.75, None), (-1.75, None)]


def test_fill_leaves_missing_data_null(bigwig: str, tsv: str):
    found = add_gerp_and_alphamissense(consequence_rows((0, 500, 'C', 'A', 'ENST00000327044')), bigwig, tsv, '1')
    assert effects(found) == [('ENST00000327044', None, None, None)]


def test_fill_chromosome_without_contig_or_rows(bigwig: str, tsv: str):
    found = add_gerp_and_alphamissense(consequence_rows((0, 945518, 'C', 'A', 'ENST00000327044')), bigwig, tsv, 'MT')
    assert effects(found) == [('ENST00000327044', None, None, None)]


def test_gerp_is_computed_once_per_variant(bigwig: str, tsv: str, monkeypatch: pytest.MonkeyPatch):
    spans: list[int] = []
    gerp_max = predictions.gerp_max

    def spy(path: str, chromosome: str, rows: pl.DataFrame) -> pl.Series:
        spans.append(rows.height)
        return gerp_max(path, chromosome, rows)

    monkeypatch.setattr(predictions, 'gerp_max', spy)
    found = add_gerp_and_alphamissense(
        consequence_rows((0, 945518, 'C', 'A', 'ENST00000327044'), (0, 945518, 'C', 'A', 'ENST00000999999')),
        bigwig,
        tsv,
        '1',
    )
    assert spans == [1]
    assert found['conservation'].to_list() == [-1.75, -1.75]


def test_effects_accept_an_empty_consequences_frame():
    found = add_gerp_and_alphamissense(pl.DataFrame(schema=CONSEQUENCES_SCHEMA), '/no/gerp.bw', '/no/am.tsv.gz', '1')
    assert found.height == 0
    assert found.select('conservation', 'am_class', 'am_pathogenicity').schema == pl.Schema({
        'conservation': pl.Float64,
        'am_class': pl.String,
        'am_pathogenicity': pl.Float64,
    })
