"""Tests for running fastVEP on one chromosome."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import stat
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.variant.fastvep import build_cache, cache_path, prepare_gff3, run_fastvep, write_vcf

FIXTURES = Path(__file__).parent / 'test_data' / 'fastvep'
SAMPLE = Path(__file__).parent / 'test_data' / 'fastvep_sample.json'

#: A real fastVEP v0.4.0 binary for `test_run_fastvep_chr22_slice`: the `FASTVEP_BINARY`
#: environment variable, else `fastvep` on the PATH. The test is skipped when neither exists.
FASTVEP_BINARY = os.environ.get('FASTVEP_BINARY') or shutil.which('fastvep')

GFF3 = """\
##gff-version 3
22\tensembl\tgene\t100\t900\t.\t+\t.\tID=gene:ENSG01;biotype=protein_coding
22\tensembl\tmRNA\t100\t900\t.\t+\t.\tID=transcript:ENST01;Parent=gene:ENSG01;tag=basic,Ensembl_canonical
22\tensembl\texon\t100\t400\t.\t+\t.\tParent=transcript:ENST01;Name=ENSE01
22\tensembl\tCDS\t150\t400\t.\t+\t0\tID=CDS:ENSP01;Parent=transcript:ENST01
###
22\thavana\tgene\t300\t2000\t.\t+\t.\tID=gene:ENSG02;biotype=protein_coding
22\thavana\tmRNA\t300\t2000\t.\t+\t.\tID=transcript:ENST02;Parent=gene:ENSG02;tag=basic,readthrough_transcript
22\thavana\tfive_prime_UTR\t300\t349\t.\t+\t.\tParent=transcript:ENST02
22\thavana\texon\t300\t600\t.\t+\t.\tParent=transcript:ENST02;Name=ENSE02
22\thavana\tCDS\t350\t600\t.\t+\t0\tID=CDS:ENSP02;Parent=transcript:ENST02
22\thavana\tmRNA\t300\t1800\t.\t+\t.\tID=transcript:ENST03;Parent=gene:ENSG02;tag=basic
22\thavana\texon\t300\t1800\t.\t+\t.\tParent=transcript:ENST03;Name=ENSE03
###
22\thavana\tgene\t3000\t3500\t.\t+\t.\tID=gene:ENSG04;biotype=artifact
22\thavana\ttranscript\t3000\t3500\t.\t+\t.\tID=transcript:ENST04;Parent=gene:ENSG04;biotype=artifact;tag=basic
22\thavana\texon\t3000\t3500\t.\t+\t.\tParent=transcript:ENST04;Name=ENSE04
"""


def fake_binary(directory: Path, body: str) -> str:
    """A shell script standing in for fastVEP; `$out` is the value of its `-o` argument."""
    script = directory / 'fastvep'
    script.write_text(
        f'#!/bin/sh\nout=""; prev=""\nfor arg in "$@"; do [ "$prev" = "-o" ] && out="$arg"; prev="$arg"; done\n{body}\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def inputs(directory: Path) -> tuple[Path, Path, Path]:
    """A VCF, a GFF3 with its cache and a FASTA; only their paths matter to a fake binary."""
    vcf, gff3, fasta = directory / 'in.vcf', directory / 'in.gff3', directory / 'in.fa'
    for path in (vcf, gff3, fasta):
        path.write_text('')
    cache_path(gff3).write_text('')
    return vcf, gff3, fasta


def test_prepare_gff3_drops_readthrough(tmp_path: Path) -> None:
    source = tmp_path / 'source.gff3.gz'
    with gzip.open(source, 'wt') as handle:
        handle.write(GFF3)

    prepared = prepare_gff3(str(source), tmp_path / 'out')

    kept = prepared.read_text().splitlines()
    assert not [line for line in kept if 'ENST02' in line or 'ENSP02' in line]
    assert kept == [line for line in GFF3.splitlines() if not {'ENST02', 'ENSP02', 'ENST04'} & set(_ids(line))]


def _ids(line: str) -> list[str]:
    return [value.split(':')[-1] for value in line.replace(';', '=').split('=')]


def test_prepare_gff3_drops_artifact_transcripts(tmp_path: Path) -> None:
    source = tmp_path / 'source.gff3'
    source.write_text(GFF3)

    prepared = prepare_gff3(str(source), tmp_path / 'out').read_text()

    assert 'ENST04' not in prepared
    assert 'ENSE04' not in prepared
    assert 'ID=gene:ENSG04;biotype=artifact' in prepared


def test_prepare_gff3_reads_plain_text(tmp_path: Path) -> None:
    source = tmp_path / 'source.gff3'
    source.write_text(GFF3)

    prepared = prepare_gff3(str(source), tmp_path / 'out')

    assert 'readthrough_transcript' not in prepared.read_text()
    assert 'ID=transcript:ENST03' in prepared.read_text()


def test_write_vcf(tmp_path: Path) -> None:
    #                1234567890123
    sequence = b'ACGTTTATCTGCA'
    rows = pl.DataFrame(
        {
            'chromosome': ['1'] * 7,
            'position': [4, 7, 10, 2, 11, 9, 2],
            'rsId': ['rs1', '.', '.', '.', '.', '.', '.'],
            'referenceAllele': ['TTT', 'AT', 'T', 'C', 'GCA', 'CTG', 'C'],
            'alternateAllele': ['TT', 'T', 'TAT', 'G', 'GA', 'CGT', 'A'],
        },
        schema_overrides={'position': pl.Int32},
    )
    path = tmp_path / 'rows.vcf'

    write_vcf(rows, path, sequence)

    assert path.read_text().splitlines() == [
        '##fileformat=VCFv4.2',
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO',
        '1\t2\t6\tC\tA\t.\t.\t.',
        '1\t2\t3\tC\tG\t.\t.\t.',
        # The shared prefix TT is trimmed and the deletion re-padded with the base before it.
        '1\t5\t0\tTT\tT\t.\t.\t.',
        # AT>T shares only a suffix: the padding base is T at position 6, from the sequence.
        '1\t6\t1\tTA\tT\t.\t.\t.',
        '1\t9\t5\tCTG\tCGT\t.\t.\t.',
        '1\t10\t2\tT\tTAT\t.\t.\t.',
        # GCA>GA trims G, then A: C at 12 is deleted, padded with G at 11.
        '1\t11\t4\tGC\tG\t.\t.\t.',
    ]


def test_write_vcf_without_a_sequence(tmp_path: Path) -> None:
    """A contig the FASTA lacks writes indels as given, without a padding base."""
    rows = pl.DataFrame(
        {
            'chromosome': ['1', '1'],
            'position': [4, 7],
            'rsId': ['.', '.'],
            'referenceAllele': ['C', 'AT'],
            'alternateAllele': ['G', 'T'],
        },
        schema_overrides={'position': pl.Int32},
    )
    path = tmp_path / 'rows.vcf'

    write_vcf(rows, path, b'')

    assert path.read_text().splitlines()[2:] == ['1\t4\t0\tC\tG\t.\t.\t.', '1\t7\t1\tAT\tT\t.\t.\t.']


def test_build_cache_next_to_the_gff3(tmp_path: Path) -> None:
    gff3, fasta = tmp_path / 'in.gff3', tmp_path / 'in.fa'
    gff3.write_text(GFF3)
    calls = tmp_path / 'calls.txt'
    binary = fake_binary(tmp_path, f'echo "$@" >> {calls}; printf \'[\\n]\\n\' > "$out"')

    build_cache(gff3, fasta, binary)

    cache = cache_path(gff3)
    assert cache.parent == gff3.parent
    assert cache.exists()
    build, warm = (line.split() for line in calls.read_text().splitlines())
    assert build[:4] == ['cache', '--gff3', str(gff3), '--no-progress']
    assert warm[0] == 'annotate'
    assert warm[warm.index('--fasta') + 1] == str(fasta)
    assert warm[warm.index('--transcript-cache') + 1] == build[build.index('-o') + 1]
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(['calls.txt', 'fastvep', 'in.gff3', cache.name])


@pytest.mark.parametrize('failing', ['cache', 'annotate'])
def test_build_cache_fails_on_nonzero_exit(tmp_path: Path, failing: str) -> None:
    gff3, fasta = tmp_path / 'in.gff3', tmp_path / 'in.fa'
    gff3.write_text(GFF3)
    binary = fake_binary(tmp_path, f'echo partial > "$out"; [ "$1" = {failing} ] && echo broken >&2 && exit 2; true')

    with pytest.raises(RuntimeError, match='broken'):
        build_cache(gff3, fasta, binary)
    assert sorted(path.name for path in tmp_path.iterdir()) == ['fastvep', 'in.gff3']


def test_run_fastvep_passes_flags_threads_and_cache(tmp_path: Path) -> None:
    vcf, gff3, fasta = inputs(tmp_path)
    calls = tmp_path / 'calls.txt'
    binary = fake_binary(tmp_path, f'echo "$RAYON_NUM_THREADS $@" > {calls}; printf \'[\\n]\\n\' > "$out"')
    output = tmp_path / 'out.json'

    assert run_fastvep(vcf, gff3, fasta, output, threads=3, binary=binary) == output

    threads, *arguments = calls.read_text().split()
    assert threads == '3'
    assert arguments[:2] == ['annotate', '--input']
    for flag in ('--pick-allele-gene', '--symbol', '--hgvs', '--canonical', '--no-progress'):
        assert flag in arguments
    assert arguments[arguments.index('--pick-order') + 1] == 'mane_select,canonical'
    assert arguments[arguments.index('--distance') + 1] == '500000'
    assert arguments[arguments.index('--output-format') + 1] == 'json'
    assert arguments[arguments.index('--transcript-cache') + 1] == str(cache_path(gff3))
    assert '--gff3' not in arguments
    assert json.loads(output.read_text()) == []


def test_run_fastvep_fails_on_truncated_output(tmp_path: Path) -> None:
    vcf, gff3, fasta = inputs(tmp_path)
    binary = fake_binary(tmp_path, 'printf \'[{"a":1}\' > "$out"')
    output = tmp_path / 'out.json'

    with pytest.raises(RuntimeError, match='truncated'):
        run_fastvep(vcf, gff3, fasta, output, threads=1, binary=binary)
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith('out')] == []


def test_run_fastvep_fails_on_nonzero_exit(tmp_path: Path) -> None:
    vcf, gff3, fasta = inputs(tmp_path)
    binary = fake_binary(tmp_path, 'printf \'[\\n]\\n\' > "$out"; echo "no such contig" >&2; exit 1')
    output = tmp_path / 'out.json'

    with pytest.raises(RuntimeError, match='no such contig'):
        run_fastvep(vcf, gff3, fasta, output, threads=1, binary=binary)
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith('out')] == []


def test_run_fastvep_reports_stderr_that_is_not_utf8(tmp_path: Path) -> None:
    vcf, gff3, fasta = inputs(tmp_path)
    binary = fake_binary(tmp_path, "printf 'bad \\377 byte\\n' >&2; exit 1")

    with pytest.raises(RuntimeError, match='bad \ufffd byte'):
        run_fastvep(vcf, gff3, fasta, tmp_path / 'out.json', threads=1, binary=binary)


def test_run_fastvep_needs_the_cache(tmp_path: Path) -> None:
    vcf, gff3, fasta = inputs(tmp_path)
    cache_path(gff3).unlink()
    binary = fake_binary(tmp_path, 'printf \'[\\n]\\n\' > "$out"')

    with pytest.raises(FileNotFoundError, match='build_cache'):
        run_fastvep(vcf, gff3, fasta, tmp_path / 'out.json', threads=1, binary=binary)


@pytest.mark.skipif(FASTVEP_BINARY is None, reason='no fastvep binary: set FASTVEP_BINARY or put fastvep on the PATH')
def test_run_fastvep_chr22_slice(tmp_path: Path) -> None:
    assert FASTVEP_BINARY is not None
    gff3, fasta = tmp_path / 'chr22_slice.gff3', tmp_path / 'chr22_slice.fa'
    for path in (gff3, fasta):
        with gzip.open(FIXTURES / f'{path.name}.gz', 'rb') as source, path.open('wb') as target:
            shutil.copyfileobj(source, target)
    vcf = FIXTURES / 'chr22_slice.vcf'
    build_cache(gff3, fasta, FASTVEP_BINARY)
    cache = cache_path(gff3).stat()

    output = run_fastvep(vcf, gff3, fasta, tmp_path / 'out.json', threads=2, binary=FASTVEP_BINARY)

    assert (cache.st_ino, cache.st_mtime_ns) == (cache_path(gff3).stat().st_ino, cache_path(gff3).stat().st_mtime_ns)

    records = json.loads(output.read_text())
    rows = [line for line in vcf.read_text().splitlines() if not line.startswith('#')]
    assert len(records) == len(rows) == 50
    assert [record['id'] for record in records] == [line.split('\t')[2] for line in rows]
    assert records == json.loads(SAMPLE.read_text())
