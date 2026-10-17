"""Tests for the coding_variant transformer."""

from pathlib import Path

import polars as pl
import pytest

from pts.schemas.coding_variant import datasource
from pts.transformers.coding_variant import (
    HARMONIC_SUM_CONSTANT,
    build_view,
    coding_variant,
    harmonic_sum,
    parse_amino_acid_change,
    process_gwas_associations,
    process_qtls,
    process_target_index,
    process_variants,
)

VE = pl.Struct({'method': pl.String, 'score': pl.Float32})
TC = pl.Struct({
    'targetId': pl.String,
    'aminoAcidChange': pl.String,
    'uniprotAccessions': pl.List(pl.String),
    'variantFunctionalConsequenceIds': pl.List(pl.String),
})
PROT = pl.Struct({'id': pl.String, 'source': pl.String})
LOCUS = pl.Struct({'variantId': pl.String, 'posteriorProbability': pl.Float64})
EV = {'datasourceId': pl.String, 'variantId': pl.String, 'score': pl.Float64, 'id': pl.String, 'diseaseId': pl.String}
GWAS = {
    'datasourceId': pl.String,
    'studyLocusId': pl.String,
    'diseaseId': pl.String,
    'score': pl.Float64,
    'id': pl.String,
}
OUTPUT_SCHEMA = {
    'targetId': pl.String,
    'variantId': pl.String,
    'therapeuticAreas': pl.List(pl.String),
    'diseases': pl.List(pl.String),
    'datasources': pl.List(datasource),
    'variantEffect': pl.Float32,
    'variantFunctionalConsequenceIds': pl.List(pl.String),
    'uniprotAccessions': pl.List(pl.String),
    'referenceAminoAcid': pl.String,
    'aminoAcidPosition': pl.Int32,
    'alternateAminoAcid': pl.String,
}


def tc(target, aa, uniprot=None, so=None) -> dict:
    return {
        'targetId': target,
        'aminoAcidChange': aa,
        'uniprotAccessions': uniprot,
        'variantFunctionalConsequenceIds': so,
    }


def ds(count, datasource, name) -> dict:
    return {'datasourceCount': count, 'datasourceId': datasource, 'datasourceNiceName': name}


def evidence(rows) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=EV, orient='row')


def world() -> dict[str, pl.DataFrame]:
    variant = pl.DataFrame(
        {
            'variantId': ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8'],
            'variantEffect': [
                [
                    {'method': 'VEP', 'score': 0.1},
                    {'method': 'AlphaMissense', 'score': 0.75},
                    {'method': 'AlphaMissense', 'score': 0.5},
                ],
                None,
                [],
                [{'method': 'AlphaMissense', 'score': 0.25}],
                None,
                [{'method': 'VEP', 'score': 0.3}],
                None,
                None,
            ],
            'transcriptConsequences': [
                [
                    tc('T1', 'A12V', ['P12345', 'Q1', 'P1'], ['SO:1']),
                    tc('T2', None),
                    tc('T2', 'p.Ala12Val', ['ABC', 'AB', 'AA'], ['SO:2']),
                ],
                [tc('T3', '*12A')],
                [tc('T1', 'a12v', ['P1'])],
                [],
                None,
                [tc('T1', 'R7H', ['P1'])],
                [tc('T4', 'G3S', ['Z9'])],
                [tc('T1', None)],
            ],
        },
        schema={'variantId': pl.String, 'variantEffect': pl.List(VE), 'transcriptConsequences': pl.List(TC)},
    )
    target = pl.DataFrame(
        {
            'id': ['T1', 'T2', 'T3', 'T4'],
            'proteinIds': [
                [
                    {'id': 'P1', 'source': 'uniprot_swissprot'},
                    {'id': 'X', 'source': 'uniprot_trembl'},
                    {'id': 'P12345', 'source': 'uniprot_swissprot'},
                ],
                [{'id': 'X', 'source': 'uniprot_trembl'}],
                None,
                [{'id': 'Z1', 'source': 'uniprot_swissprot'}],
            ],
        },
        schema={'id': pl.String, 'proteinIds': pl.List(PROT)},
    )
    disease = pl.DataFrame(
        {'id': ['D1', 'D2', 'D3', 'D4'], 'therapeuticAreas': [['TA2', 'TA1'], ['TA2'], None, ['TA3']]},
        schema={'id': pl.String, 'therapeuticAreas': pl.List(pl.String)},
    )
    gwas = pl.DataFrame(
        [
            ('gwas_credible_sets', 'SL1', 'D1', 0.75, 'g1'),
            ('gwas_credible_sets', 'SL1', 'D1', 0.25, 'g2'),
            ('gwas_credible_sets', 'SL1', 'D4', 0.5, 'g3'),
            ('gwas_credible_sets', 'SL2', 'D4', 0.5, 'g4'),
            ('gwas_credible_sets', 'SL2', 'D4', 0.5, 'g5'),
        ],
        schema=GWAS,
        orient='row',
    )
    credible_set = pl.DataFrame(
        {
            'studyLocusId': ['SL1', 'SL2', 'SL3', 'SL4', 'SL5', 'SL6'],
            'studyType': ['gwas', 'gwas', 'eqtl', 'pqtl', 'sqtl', None],
            'locus': [
                [{'variantId': 'V1', 'posteriorProbability': 0.5}, {'variantId': 'V9', 'posteriorProbability': 0.5}],
                [{'variantId': 'V1', 'posteriorProbability': 1.0}],
                [
                    {'variantId': 'V7', 'posteriorProbability': 0.5},
                    {'variantId': 'V7', 'posteriorProbability': 0.1},
                    {'variantId': 'V1', 'posteriorProbability': 0.4},
                ],
                [],
                None,
                [{'variantId': 'V3', 'posteriorProbability': 1.0}],
            ],
        },
        schema={'studyLocusId': pl.String, 'studyType': pl.String, 'locus': pl.List(LOCUS)},
    )
    return {
        'variant': variant,
        'target': target,
        'disease': disease,
        'evidence_eva': evidence([
            ('eva', 'V1', 0.5, 'e1', 'D1'),
            ('eva', 'V1', 0.25, 'e2', 'D2'),
            ('eva', 'V6', 0.0, 'e3', 'D1'),
            ('eva', None, 0.9, 'e4', 'D1'),
        ]),
        'evidence_eva_somatic': evidence([
            ('eva_somatic', 'V1', 0.5, 'e5', 'D2'),
            ('eva_somatic', 'V1', 0.25, 'e6', 'D1'),
        ]),
        'evidence_uniprot_variant': evidence([
            ('uniprot_variants', 'V2', 1.0, 'e7', 'D3'),
            ('uniprot_variants', 'V2', 0.5, 'e8', 'D4'),
        ]),
        'evidence_gwas_credible_set': gwas,
        'credible_set': credible_set,
        'pharmacogenetics': pl.DataFrame({'variantId': ['V2', 'V2', None], 'datasourceId': ['clinpgx'] * 3}),
    }


V1_AGG = {
    'variantId': 'V1',
    'therapeuticAreas': ['TA1', 'TA2', 'TA3'],
    'diseases': ['D1', 'D4', 'D2'],
    'datasources': [
        ds(2, 'eva', 'ClinVar'),
        ds(2, 'eva_somatic', 'ClinVar-somatic'),
        ds(2, 'gwas_credible_sets', 'GWAS credible sets'),
        ds(1, 'mol_qtl', 'molQTL'),
    ],
    'variantEffect': 0.75,
}
# spark's rows on world(), with uniprotAccessions, therapeuticAreas and datasources sorted
EXPECTED = [
    V1_AGG
    | {
        'targetId': 'T1',
        'variantFunctionalConsequenceIds': ['SO:1'],
        'uniprotAccessions': ['P1', 'P12345'],
        'referenceAminoAcid': 'A',
        'aminoAcidPosition': 12,
        'alternateAminoAcid': 'V',
    },
    V1_AGG
    | {
        'targetId': 'T2',
        'variantFunctionalConsequenceIds': ['SO:2'],
        'uniprotAccessions': ['AA', 'AB', 'ABC'],
        'referenceAminoAcid': 'p',
        'aminoAcidPosition': 12,
        'alternateAminoAcid': '',
    },
    {
        'targetId': 'T3',
        'variantId': 'V2',
        'therapeuticAreas': ['TA3'],
        'diseases': ['D3', 'D4'],
        'datasources': [ds(2, 'clinpgx', 'clinpgx'), ds(2, 'uniprot_variants', 'UniProt Variants')],
        'variantEffect': None,
        'variantFunctionalConsequenceIds': None,
        'uniprotAccessions': None,
        'referenceAminoAcid': '*',
        'aminoAcidPosition': 12,
        'alternateAminoAcid': 'A',
    },
    {
        'targetId': 'T1',
        'variantId': 'V6',
        'therapeuticAreas': [],
        'diseases': [],
        'datasources': [],
        'variantEffect': None,
        'variantFunctionalConsequenceIds': None,
        'uniprotAccessions': ['P1'],
        'referenceAminoAcid': 'R',
        'aminoAcidPosition': 7,
        'alternateAminoAcid': 'H',
    },
    {
        'targetId': 'T4',
        'variantId': 'V7',
        'therapeuticAreas': [],
        'diseases': [],
        'datasources': [ds(2, 'mol_qtl', 'molQTL')],
        'variantEffect': None,
        'variantFunctionalConsequenceIds': None,
        'uniprotAccessions': [],
        'referenceAminoAcid': 'G',
        'aminoAcidPosition': 3,
        'alternateAminoAcid': 'S',
    },
]


def view(frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    lazy = {name: frame.lazy() for name, frame in frames.items()}
    return build_view(process_variants(lazy.pop('variant')), lazy).collect()


def by_variant(out: pl.DataFrame) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for row in out.iter_rows(named=True):
        rows.setdefault(row['variantId'], []).append(row)
    return rows


def reference_harmonic_sum(scores: list[float]) -> float:
    total = 0.0
    for i, s in enumerate(sorted(scores, reverse=True)):
        total += s / (i + 1) ** 2 / HARMONIC_SUM_CONSTANT
    return total


def test_harmonic_sum_constant_matches_spark() -> None:
    assert sum(1 / ((i + 1) ** 2) for i in range(1000)) == HARMONIC_SUM_CONSTANT


def test_harmonic_sum_matches_reference() -> None:
    scores = [0.3, 0.8, 1.0, 0.7, 0.123456789, 0.987654321, 0.5, 0.5, 0.1 + 0.2]
    got = pl.DataFrame({'s': scores}).select(harmonic_sum(pl.col('s'))).item()
    assert got == reference_harmonic_sum(scores)


def test_harmonic_sum_per_group_matches_reference() -> None:
    groups = {
        'a': [0.3, 0.8, 1.0],
        'b': [0.7, 0.2, 0.9, 0.9, 0.1 + 0.2],
        'c': [0.288],
    }
    frame = pl.DataFrame(
        [(g, s) for g, scores in groups.items() for s in scores] + [('b', None), ('d', None)],
        schema={'g': pl.String, 's': pl.Float64},
        orient='row',
    )
    chunked = pl.concat([frame.slice(0, 4), frame.slice(4)], rechunk=False)
    got = dict(chunked.group_by('g').agg(harmonic_sum(pl.col('s')).alias('h')).iter_rows())
    assert got == {**{g: reference_harmonic_sum(s) for g, s in groups.items()}, 'd': None}


@pytest.mark.parametrize(
    ('aa', 'expected'),
    [
        ('p.Ala12Val', ('p', 12, '')),
        ('A12V', ('A', 12, 'V')),
        ('*12A', ('*', 12, 'A')),
        ('a12v', ('a', 12, '')),
        ('A12z', ('A', 12, 'z')),
        ('AV', ('AV', None, 'AV')),
        ('12', ('', 12, '')),
        ('A99999999999V', ('A', None, 'V')),
        ('-12+', ('-', 12, '+')),
        ('A12V\n', ('A', 12, 'V')),
        ('A\u0661\u0662V', ('A', None, 'V')),
        ('', ('', None, '')),
        (None, (None, None, None)),
    ],
)
def test_parse_amino_acid_change(aa, expected) -> None:
    frame = pl.DataFrame({'aa': [aa]}, schema={'aa': pl.String})
    ref, position, alt = parse_amino_acid_change(pl.col('aa'))
    out = frame.select(ref, position, alt)
    assert out.columns == ['referenceAminoAcid', 'aminoAcidPosition', 'alternateAminoAcid']
    assert out.schema['aminoAcidPosition'] == pl.Int32
    assert out.row(0) == expected


def test_process_variants_keeps_first_alphamissense_and_explodes_consequences() -> None:
    out = process_variants(world()['variant'].lazy()).collect()
    assert out.columns == [
        'variantId',
        'variantEffect',
        'targetId',
        'variantFunctionalConsequenceIds',
        'uniprotAccessions',
        'referenceAminoAcid',
        'aminoAcidPosition',
        'alternateAminoAcid',
    ]
    got = sorted(out.select('variantId', 'targetId', 'variantEffect').rows(), key=lambda r: (r[0], r[1]))
    assert got == [
        ('V1', 'T1', 0.75),
        ('V1', 'T2', 0.75),
        ('V2', 'T3', None),
        ('V3', 'T1', None),
        ('V6', 'T1', None),
        ('V7', 'T4', None),
    ]


def test_process_target_index_keeps_swissprot_only() -> None:
    out = process_target_index(world()['target'].lazy()).collect().sort('targetId')
    # spark: size(null) is -1, so a target with null proteinIds stays with null accessions
    assert out.rows() == [('T1', ['P1', 'P12345']), ('T3', None), ('T4', ['Z1'])]


def test_qtl_variant_ids_survive_case() -> None:
    frames = world()
    variant_ids = process_variants(frames['variant'].lazy()).select('variantId')
    qtls = process_qtls(frames['credible_set'].lazy(), variant_ids).collect()
    assert 'variantId' in qtls.columns
    assert sorted(qtls['variantId'].to_list()) == ['V1', 'V7', 'V7']
    out = by_variant(view(frames))
    assert [d['datasourceId'] for d in out['V7'][0]['datasources']] == ['mol_qtl']
    assert 'mol_qtl' in [d['datasourceId'] for d in out['V1'][0]['datasources']]


def test_uniprot_accessions_sorted_by_length_then_value() -> None:
    frames = world()
    frames['variant'] = pl.DataFrame(
        {
            'variantId': ['V1', 'V1'],
            'variantEffect': [None, None],
            'transcriptConsequences': [
                [tc('T2', 'A1V', ['BBB', 'C', 'AAAA', 'B', 'AA', 'A'])],
                [tc('T1', 'A1V', ['P12345', 'Q1', 'P1', 'A12345'])],
            ],
        },
        schema={'variantId': pl.String, 'variantEffect': pl.List(VE), 'transcriptConsequences': pl.List(TC)},
    )
    frames['target'] = pl.DataFrame(
        {
            'id': ['T1'],
            'proteinIds': [[{'id': x, 'source': 'uniprot_swissprot'} for x in ['P12345', 'A12345', 'P1', 'Z']]],
        },
        schema={'id': pl.String, 'proteinIds': pl.List(PROT)},
    )
    out = view(frames).sort('targetId')
    assert out['uniprotAccessions'].to_list() == [['P1', 'A12345', 'P12345'], ['A', 'B', 'C', 'AA', 'BBB', 'AAAA']]


def test_diseases_sorted_by_harmonic_sum_desc_tie_by_id() -> None:
    frames = world()
    frames['evidence_eva'] = evidence(
        [('eva', 'V1', 0.5, f'e{d}', d) for d in ['D4', 'D2', 'D1', 'D3']] + [('eva', 'V1', 0.9, 'e9', 'D2')]
    )
    frames['evidence_eva_somatic'] = frames['evidence_eva_somatic'].clear()
    frames['evidence_gwas_credible_set'] = frames['evidence_gwas_credible_set'].clear()
    out = by_variant(view(frames))
    assert [row['diseases'] for row in out['V1']] == [['D2', 'D1', 'D3', 'D4']] * 2


def test_variants_without_evidence_follow_spark_right_join() -> None:
    out = view(world())
    # V3 is coding but without evidence; V9 has evidence but is not coding; V4/V5/V8 have no amino acid change
    assert sorted(set(out['variantId'])) == ['V1', 'V2', 'V6', 'V7']
    # one row per amino acid consequence, not per variant
    assert out.filter(variantId='V1').height == 2


def test_gwas_keeps_tied_maxima_and_coding_fan_out() -> None:
    frames = world()
    variant_ids = process_variants(frames['variant'].lazy()).select('variantId')
    gwas = process_gwas_associations(
        variant_ids,
        frames['disease'].lazy().select(pl.col('id').alias('diseaseId'), 'therapeuticAreas'),
        frames['evidence_gwas_credible_set'].lazy(),
        frames['credible_set'].lazy(),
    ).collect()
    got = sorted(gwas.select('variantId', 'diseaseId', 'score', 'evidenceId').rows())
    # V1 has two coding rows, so every row is doubled; SL1/D1 keeps only its max, SL2/D4 keeps both tied rows
    assert got == sorted(
        [('V1', 'D1', 0.75, 'SL1')] * 2 + [('V1', 'D4', 0.5, 'SL1')] * 2 + [('V1', 'D4', 0.5, 'SL2')] * 4
    )


def test_empty_locus_dropped() -> None:
    frames = world()
    frames['credible_set'] = pl.DataFrame(
        {
            'studyLocusId': ['SL3', 'SL4', 'SL5'],
            'studyType': ['eqtl', 'pqtl', 'sqtl'],
            'locus': [[{'variantId': 'V7', 'posteriorProbability': 0.5}], [], None],
        },
        schema={'studyLocusId': pl.String, 'studyType': pl.String, 'locus': pl.List(LOCUS)},
    )
    variant_ids = process_variants(frames['variant'].lazy()).select('variantId')
    assert process_qtls(frames['credible_set'].lazy(), variant_ids).collect()['variantId'].to_list() == ['V7']


def test_coding_variant_null_and_empty_lists() -> None:
    out = by_variant(view(world()))
    # V4 has no consequences, V5 null consequences: no rows
    assert 'V4' not in out
    assert 'V5' not in out
    # zero-score evidence only: no diseases, no areas, no datasources
    v6 = out['V6'][0]
    assert (v6['diseases'], v6['therapeuticAreas'], v6['datasources']) == ([], [], [])
    # target with null proteinIds keeps the variant's null accessions; no swissprot overlap gives []
    assert out['V2'][0]['uniprotAccessions'] is None
    assert out['V7'][0]['uniprotAccessions'] == []
    # disease with null therapeuticAreas contributes none
    assert out['V2'][0]['therapeuticAreas'] == ['TA3']


def test_build_view_matches_spark() -> None:
    out = view(world())
    assert out.schema == pl.Schema(OUTPUT_SCHEMA)
    assert out.to_dicts() == EXPECTED


def write_parts(frame: pl.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True)
    half = frame.height // 2
    frame.slice(0, half).write_parquet(directory / 'part-0.parquet')
    frame.slice(half).write_parquet(directory / 'part-1.parquet')


def test_coding_variant_end_to_end_is_deterministic(tmp_path: Path) -> None:
    source = {}
    for name, frame in world().items():
        write_parts(frame, tmp_path / 'in' / name)
        source[name] = str(tmp_path / 'in' / name)
    outputs = []
    for run in ('a', 'b'):
        destination = tmp_path / run
        coding_variant(source, str(destination), None, None)
        files = sorted(destination.glob('*.parquet'))
        outputs.append([f.read_bytes() for f in files])
        out = pl.read_parquet(files)
        assert out.schema == pl.Schema(OUTPUT_SCHEMA)
        assert out.to_dicts() == EXPECTED
    assert outputs[0] == outputs[1]
