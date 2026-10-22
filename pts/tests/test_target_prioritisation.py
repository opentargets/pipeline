"""Tests for the target_prioritisation transformer."""

import math
import random

import numpy as np
import polars as pl
import pytest

from pts.transformers.target_prioritisation import (
    baseline_query,
    biotype_query,
    cast_double,
    chemical_probes_query,
    clin_trials_query,
    compute_target_prioritisation,
    constraint_query,
    driver_gene_query,
    find_parent_child_cousins,
    harmonic_sum,
    ligand_pocket_query,
    max_harmonic_sum,
    mouse_model_query,
    orthologs_mouse_query,
    paralogs_query,
    safety_query,
    target_membrane_query,
)

TRACTABILITY_SCHEMA = {'targetId': pl.String, 'modality': pl.String, 'category': pl.String, 'value': pl.Boolean}
TARGET_SAFETY_EVENT_SCHEMA = {'targetId': pl.String, 'event': pl.String}
CHEMICAL_PROBES_SCHEMA = {'targetId': pl.String, 'isHighQuality': pl.Boolean}
HOMOLOGY_SCHEMA = {
    'targetId': pl.String,
    'speciesId': pl.String,
    'speciesName': pl.String,
    'homologyType': pl.String,
    'targetGeneId': pl.String,
    'isHighConfidence': pl.String,
    'targetGeneSymbol': pl.String,
    'queryPercentageIdentity': pl.Float64,
    'targetPercentageIdentity': pl.Float64,
    'priority': pl.Int32,
}
UNIPROT_SCHEMA = {'Subcellular location ID': pl.String, 'Name': pl.String, 'Is part of': pl.String, 'Is a': pl.String}
LOCATION = pl.Struct({
    'location': pl.String,
    'source': pl.String,
    'termSL': pl.String,
    'labelSL': pl.String,
    'targetModifier': pl.String,
})
CONSTRAINT = pl.Struct({'constraintType': pl.String, 'upperRank': pl.Int32})
HALLMARKS = pl.Struct({
    'attributes': pl.List(pl.Struct({'pmid': pl.Int64, 'description': pl.String, 'name': pl.String})),
})
TARGET_SCHEMA = {
    'id': pl.String,
    'biotype': pl.String,
    'subcellularLocations': pl.List(LOCATION),
    'constraint': pl.List(CONSTRAINT),
    'hallmarks': HALLMARKS,
}
MOUSE_SCHEMA = {
    'targetFromSourceId': pl.String,
    'modelPhenotypeClasses': pl.List(pl.Struct({'id': pl.String, 'label': pl.String})),
}
PHENO_SCHEMA = {'id': pl.String, 'label': pl.String, 'score': pl.Float64}
MOA_SCHEMA = {'targets': pl.List(pl.String), 'chemblIds': pl.List(pl.String)}
MOLECULE_SCHEMA = {'id': pl.String, 'maximumClinicalStage': pl.String}
BASELINE_SCHEMA = {
    'targetId': pl.String,
    'datasourceId': pl.String,
    'tissueBiosampleId': pl.String,
    'celltypeBiosampleId': pl.String,
    'specificity_score': pl.Float64,
    'distribution_score': pl.Float64,
}
OUTPUT_SCHEMA = {
    'targetId': pl.String,
    'isInMembrane': pl.Int32,
    'isSecreted': pl.Int32,
    'hasSafetyEvent': pl.Int32,
    'hasPocket': pl.Int32,
    'hasLigand': pl.Int32,
    'hasSmallMoleculeBinder': pl.Int32,
    'geneticConstraint': pl.Float64,
    'paralogMaxIdentityPercentage': pl.Float64,
    'mouseOrthologMaxIdentityPercentage': pl.Float64,
    'isCancerDriverGene': pl.Int32,
    'mouseKOScore': pl.Float64,
    'hasHighQualityChemicalProbes': pl.Int32,
    'maxClinicalStage': pl.Float64,
    'tissueSpecificity': pl.Float64,
    'tissueDistribution': pl.Float64,
    'celltypeSpecificity': pl.Float64,
    'celltypeDistribution': pl.Float64,
}


def frame(rows, schema) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=schema, orient='row')


def queryset(target_ids) -> pl.DataFrame:
    return pl.DataFrame({'targetid': target_ids}, schema={'targetid': pl.String})


def first_row(df: pl.DataFrame, target_id: str) -> dict:
    rows = df.filter(pl.col('targetid') == target_id).rows(named=True)
    assert rows
    return rows[0]


# ---------------------------------------------------------------------------
# ligand_pocket_query
# ---------------------------------------------------------------------------


def test_ligand_pocket_query_flags_high_quality_ligand():
    tractability = frame([('T1', 'SM', 'High-Quality Ligand', True)], TRACTABILITY_SCHEMA)
    row = first_row(ligand_pocket_query(queryset(['T1']), tractability), 'T1')
    assert row['Nr_Ligand'] == 1
    assert row['Nr_Pocket'] == 0
    assert row['Nr_sMBinder'] == 0


def test_ligand_pocket_query_ignores_false_values():
    tractability = frame([('T1', 'SM', 'High-Quality Pocket', False)], TRACTABILITY_SCHEMA)
    row = first_row(ligand_pocket_query(queryset(['T1']), tractability), 'T1')
    assert row['Nr_Pocket'] == 0


def test_ligand_pocket_query_target_with_no_rows_is_kept():
    result = ligand_pocket_query(queryset(['T1']), frame([], TRACTABILITY_SCHEMA))
    assert result.height == 1


def test_ligand_pocket_query_ignores_unrelated_modality_ids():
    tractability = frame(
        [('T1', 'SM', 'High-Quality Ligand', True), ('T1', 'AB', 'Clinical Precedence', True)], TRACTABILITY_SCHEMA
    )
    result = ligand_pocket_query(queryset(['T1']), tractability)
    assert first_row(result, 'T1')['Nr_Ligand'] == 1
    assert 'Clinical Precedence' not in result.columns


def test_ligand_pocket_query_sum_of_two_is_not_one():
    tractability = frame(
        [('T1', 'SM', 'High-Quality Ligand', True), ('T1', 'AB', 'High-Quality Ligand', True)], TRACTABILITY_SCHEMA
    )
    row = first_row(ligand_pocket_query(queryset(['T1', 'T2']), tractability), 'T1')
    assert row['High-Quality Ligand'] == 2
    assert row['Nr_Ligand'] == 0
    assert row['High-Quality Pocket'] is None


# ---------------------------------------------------------------------------
# safety_query
# ---------------------------------------------------------------------------


def test_safety_query_flags_targets_with_events():
    safety = frame([('T1', 'hepatotoxicity')], TARGET_SAFETY_EVENT_SCHEMA)
    row = first_row(safety_query(queryset(['T1']), safety), 'T1')
    assert row['hasSafetyEvent'] == 'Yes'
    assert row['Nr_Event'] == -1


def test_safety_query_target_with_no_events_left_null():
    row = first_row(safety_query(queryset(['T1']), frame([], TARGET_SAFETY_EVENT_SCHEMA)), 'T1')
    assert row['hasSafetyEvent'] is None


def test_safety_query_counts_distinct_events():
    safety = frame([('T1', 'hepatotoxicity'), ('T1', 'cardiotoxicity')], TARGET_SAFETY_EVENT_SCHEMA)
    row = first_row(safety_query(queryset(['T1']), safety), 'T1')
    assert row['nEvents'] == 2
    assert set(row['events']) == {'hepatotoxicity', 'cardiotoxicity'}


def test_safety_query_null_events_only():
    row = first_row(safety_query(queryset(['T1']), frame([('T1', None)], TARGET_SAFETY_EVENT_SCHEMA)), 'T1')
    assert row['nEvents'] == 0
    assert row['Nr_Event'] is None


# ---------------------------------------------------------------------------
# chemical_probes_query
# ---------------------------------------------------------------------------


def test_chemical_probes_query_flags_high_quality():
    probes = frame([('T1', True)], CHEMICAL_PROBES_SCHEMA)
    assert first_row(chemical_probes_query(queryset(['T1']), probes), 'T1')['Nr_chprob'] == 1


def test_chemical_probes_query_low_quality_only():
    probes = frame([('T1', False)], CHEMICAL_PROBES_SCHEMA)
    assert first_row(chemical_probes_query(queryset(['T1']), probes), 'T1')['Nr_chprob'] == 0


def test_chemical_probes_query_target_with_no_probes_left_null():
    probes = frame([], CHEMICAL_PROBES_SCHEMA)
    assert first_row(chemical_probes_query(queryset(['T1']), probes), 'T1')['Nr_chprob'] is None


def test_chemical_probes_query_drops_unresolved_target_id():
    probes = frame([(None, True)], CHEMICAL_PROBES_SCHEMA)
    assert first_row(chemical_probes_query(queryset(['T1']), probes), 'T1')['Nr_chprob'] is None


# ---------------------------------------------------------------------------
# paralogs_query / orthologs_mouse_query
# ---------------------------------------------------------------------------


def homologue(**kwargs) -> pl.DataFrame:
    row = {
        'targetId': 'T1',
        'speciesId': '9606',
        'speciesName': 'Human',
        'homologyType': 'other_paralog',
        'targetGeneId': 'T2',
        'isHighConfidence': '1',
        'targetGeneSymbol': 'GENE2',
        'queryPercentageIdentity': 70.0,
        'targetPercentageIdentity': 65.0,
        'priority': 0,
    }
    row.update(kwargs)
    return pl.DataFrame([row], schema=HOMOLOGY_SCHEMA)


def test_paralogs_query_scores_high_identity_paralog():
    homology = homologue(homologyType='other_paralog', queryPercentageIdentity=80.0)
    assert first_row(paralogs_query(queryset(['T1']), homology), 'T1')['Nr_paralogs'] == -0.5


def test_paralogs_query_low_identity_paralog_scores_zero():
    homology = homologue(homologyType='within_species_paralog', queryPercentageIdentity=40.0)
    assert first_row(paralogs_query(queryset(['T1']), homology), 'T1')['Nr_paralogs'] == 0


def test_paralogs_query_ignores_orthologs():
    homology = homologue(homologyType='ortholog_one2one', queryPercentageIdentity=95.0)
    assert first_row(paralogs_query(queryset(['T1']), homology), 'T1')['Nr_paralogs'] is None


def test_paralogs_query_target_with_no_rows_is_kept():
    assert paralogs_query(queryset(['T1']), frame([], HOMOLOGY_SCHEMA)).height == 1


@pytest.mark.parametrize(
    ('homology_type', 'expected'),
    [
        ('other_paralog', True),
        ('within_species_paralog', True),
        ('ortholog_one2one', False),
        ('paralog_x', True),
        ('', False),
        (None, False),
    ],
)
def test_paralogs_query_homology_type(homology_type, expected):
    homology = homologue(homologyType=homology_type, queryPercentageIdentity=90.0)
    row = first_row(paralogs_query(queryset(['T1']), homology), 'T1')
    assert (row['Nr_paralogs'] is not None) == expected


def test_orthologs_mouse_query_scores_high_identity_mouse_ortholog():
    homology = homologue(homologyType='ortholog_one2one', speciesName='Mouse', queryPercentageIdentity=90.0)
    assert first_row(orthologs_mouse_query(queryset(['T1']), homology), 'T1')['Nr_ortholog'] == 0.5


def test_orthologs_mouse_query_ignores_non_mouse_species():
    homology = homologue(homologyType='ortholog_one2one', speciesName='Zebrafish', queryPercentageIdentity=95.0)
    assert first_row(orthologs_mouse_query(queryset(['T1']), homology), 'T1')['Nr_ortholog'] is None


def test_orthologs_mouse_query_ignores_mouse_paralogs():
    homology = homologue(homologyType='other_paralog', speciesName='Mouse', queryPercentageIdentity=95.0)
    assert first_row(orthologs_mouse_query(queryset(['T1']), homology), 'T1')['Nr_ortholog'] is None


def test_orthologs_mouse_query_target_with_no_rows_is_kept():
    assert orthologs_mouse_query(queryset(['T1']), frame([], HOMOLOGY_SCHEMA)).height == 1


PARALOG_VALUES = [60.0, 61.7, 63.3, 66.9, 70.1, 77.7, 80.0, 88.8, 91.3, 99.9, 100.0]
ORTHOLOG_VALUES = [80.0, 81.7, 83.3, 86.9, 88.8, 91.3, 97.7, 99.9, 100.0]


def test_division_by_literal_matches_numpy():
    homology = pl.concat([
        homologue(targetId=f'P{i}', queryPercentageIdentity=v) for i, v in enumerate(PARALOG_VALUES)
    ] + [
        homologue(targetId=f'O{i}', homologyType='ortholog_one2one', speciesName='Mouse', queryPercentageIdentity=v)
        for i, v in enumerate(ORTHOLOG_VALUES)
    ])  # fmt: skip
    pids = [f'P{i}' for i in range(len(PARALOG_VALUES))]
    oids = [f'O{i}' for i in range(len(ORTHOLOG_VALUES))]
    par = dict(paralogs_query(queryset(pids), homology).select('targetid', 'Nr_paralogs').rows())
    orth = dict(orthologs_mouse_query(queryset(oids), homology).select('targetid', 'Nr_ortholog').rows())
    assert [par[i] for i in pids] == [-((np.float64(v) - 60) / 40) for v in PARALOG_VALUES]
    assert [orth[i] for i in oids] == [(np.float64(v) - 80) / 20 for v in ORTHOLOG_VALUES]
    assert any((np.float64(v) - 60) / 40 != (np.float64(v) - 60) * (1 / 40) for v in PARALOG_VALUES)
    assert any((np.float64(v) - 80) / 20 != (np.float64(v) - 80) * (1 / 20) for v in ORTHOLOG_VALUES)


# ---------------------------------------------------------------------------
# harmonic_sum / max_harmonic_sum
# ---------------------------------------------------------------------------


def _left_to_right_sum(terms):
    # Python 3.12+ compensates float sums, so sum() no longer reproduces the release
    total = 0.0
    for term in terms:
        total += term
    return total


def _harmonic_sum_reference(scores):
    """The former `harmonic_sum` UDF, summed left to right as the release's Python 3.11 `sum` did."""
    if scores is None or len(scores) == 0:
        return 0.0
    sorted_scores = sorted([s for s in scores if s is not None], reverse=True)
    if len(sorted_scores) == 0:
        return 0.0
    denominators = [(i + 1) ** 2 for i in range(len(sorted_scores))]
    return _left_to_right_sum(s / d for s, d in zip(sorted_scores, denominators, strict=False))


def _max_harmonic_sum_reference(scores):
    """The former `max_harmonic_sum` UDF, summed left to right as the release's Python 3.11 `sum` did."""
    if scores is None or len(scores) == 0:
        return 0.0
    n = len([s for s in scores if s is not None])
    if n == 0:
        return 0.0
    denominators = [(i + 1) ** 2 for i in range(n)]
    return _left_to_right_sum(1.0 / d for d in denominators)


def _same_double(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == b


def harmonics(cases) -> list[tuple]:
    df = pl.DataFrame({'score': cases}, schema={'score': pl.List(pl.Float64)})
    return df.select(
        harmonic_sum(pl.col('score')).alias('harmonic'),
        max_harmonic_sum(pl.col('score')).alias('max_harmonic'),
    ).rows()


def test_harmonic_sums_match_udfs():
    rng = random.Random(0)
    cases = [
        None,
        [],
        [None],
        [None, None],
        [0.5],
        [0.0],
        [0.2, 0.9, 0.5],
        [0.5, 0.5, 0.5],
        [None, 0.3, None, 0.8],
        [1.0, 0.0, -0.0, 0.25],
        [float('nan'), 0.3],
    ]
    cases += [[rng.choice([0.0, 1.0, rng.random()]) for _ in range(rng.randint(1, 300))] for _ in range(500)]
    for case, (harmonic, max_harmonic) in zip(cases, harmonics(cases), strict=True):
        assert _same_double(harmonic, _harmonic_sum_reference(case))
        assert _same_double(max_harmonic, _max_harmonic_sum_reference(case))


def test_harmonic_sums_match_udfs_beyond_int32_squared_ranks():
    # ranks above 46,340 overflow an int32 square
    rng = random.Random(1)
    scores = [rng.random() for _ in range(70_000)]
    ((harmonic, max_harmonic),) = harmonics([scores])
    assert harmonic == _harmonic_sum_reference(scores)
    assert max_harmonic == _max_harmonic_sum_reference(scores)


# ---------------------------------------------------------------------------
# find_parent_child_cousins / target_membrane_query
# ---------------------------------------------------------------------------

UNIPROT = [
    ('SL-1', 'Cell membrane', None, None),
    ('SL-2', 'Apical membrane', 'SL-9; Foo; LOCATION', 'SL-1; Cell membrane; LOCATION'),
    ('SL-3', 'Lateral membrane', 'SL-1; Cell membrane; LOCATION', 'SL-5; X; LOCATION, SL-1; Cell membrane; LOCATION'),
    ('SL-4', 'Secreted', None, None),
    ('SL-5', 'X', 'SL-8; Y; LOCATION, SL-4; Secreted; LOCATION', None),
    ('SL-6', 'Extracellular', 'SL-4; Secreted; LOCATION', None),
]


def test_parent_child_cousins():
    got = find_parent_child_cousins(frame(UNIPROT, UNIPROT_SCHEMA))
    rows = sorted(
        (r['Name'], r['SubcellID'], r['Is_a'], sorted(r['toSearch']), sorted(r['concat'].split(',')))
        for r in got.iter_rows(named=True)
    )
    assert rows == [
        ('Apical membrane', 'SL-2', 'SL-1', ['SL-2'], ['SL-2']),
        ('Cell membrane', 'SL-1', None, ['SL-1', 'SL-2', 'SL-3', 'SL-3'], ['SL-1', 'SL-2', 'SL-3', 'SL-3']),
        ('Extracellular', 'SL-6', None, ['SL-6'], ['SL-6']),
        ('Lateral membrane', 'SL-3', ' SL-1', ['SL-3'], ['SL-3']),
        ('Lateral membrane', 'SL-3', 'SL-5', ['SL-3'], ['SL-3']),
        ('Secreted', 'SL-4', None, ['SL-4', 'SL-6'], ['SL-4', 'SL-6']),
        ('X', 'SL-5', None, ['SL-3', 'SL-5'], ['SL-3', 'SL-5']),
    ]


def loc(source, term, location='x'):
    return {'location': location, 'source': source, 'termSL': term, 'labelSL': None, 'targetModifier': None}


NULL_LOCATION = {'location': None, 'source': None, 'termSL': None, 'labelSL': None, 'targetModifier': None}

MEMBRANE_TARGETS = [
    ('T1', [loc('HPA_main', 'SL-2')]),
    ('T2', [loc('HPA_extracellular_location', 'SL-0'), loc('HPA_main', 'SL-9')]),
    ('T3', [loc('HPA_main', 'SL-1'), loc('HPA_extracellular_location', 'SL-0')]),
    ('T4', [loc('uniprot', 'SL-3')]),
    ('T5', [loc('uniprot', 'SL-6')]),
    ('T6', [loc('uniprot', 'SL-1'), loc('uniprot', 'SL-4')]),
    ('T7', [loc('uniprot', 'SL-9')]),
    ('T8', None),
    ('T9', []),
    ('T10', [loc('HPA_main', 'SL-9')]),
    ('T11', [loc('HPA_additional', 'SL-2')]),
    ('T12', [loc('uniprot', 'SL-4', location=None)]),
    ('T13', [loc('HPA_main', None), NULL_LOCATION]),
    ('T14', [loc('HPA_main', 'SL-9'), loc('uniprot', 'SL-1')]),
    ('T15', [loc('uniprot', None)]),
    ('T16', [loc('HPA_additional', 'SL-9'), loc('uniprot', 'SL-5')]),
]

MEMBRANE_EXPECTED = [
    ('T1', 1, 0),
    ('T2', 0, 1),
    ('T3', 1, 1),
    ('T4', 1, 0),
    ('T5', 0, 1),
    ('T6', 1, 1),
    ('T7', 0, 0),
    ('T8', None, None),
    ('T9', None, None),
    ('T10', 0, 0),
    ('T11', 1, 0),
    ('T12', 0, 1),
    ('T13', 0, 0),
    ('T13', None, None),
    ('T14', 0, 0),
    ('T15', 0, 0),
    ('T16', 0, 0),
]


def targets(rows: list[dict]) -> pl.DataFrame:
    base = {'biotype': 'protein_coding', 'subcellularLocations': None, 'constraint': None, 'hallmarks': None}
    return pl.DataFrame([base | row for row in rows], schema=TARGET_SCHEMA)


def membrane_targets() -> pl.DataFrame:
    return targets([{'id': t, 'subcellularLocations': s} for t, s in MEMBRANE_TARGETS])


def test_membrane_secreted():
    pcc = find_parent_child_cousins(frame(UNIPROT, UNIPROT_SCHEMA))
    qs = queryset([t for t, _ in MEMBRANE_TARGETS])
    got = target_membrane_query(qs, membrane_targets(), pcc).select('targetid', 'Nr_mb', 'Nr_secreted')
    assert got.schema == {'targetid': pl.String, 'Nr_mb': pl.Int32, 'Nr_secreted': pl.Int32}
    assert sorted(got.rows(), key=str) == sorted(MEMBRANE_EXPECTED, key=str)


# ---------------------------------------------------------------------------
# biotype_query / constraint_query / driver_gene_query
# ---------------------------------------------------------------------------


def test_biotype():
    tgt = targets([
        {'id': 'A', 'biotype': 'protein_coding'},
        {'id': 'B', 'biotype': ''},
        {'id': 'C', 'biotype': 'lncRNA'},
        {'id': 'D', 'biotype': None},
    ])
    got = biotype_query(queryset(list('ABCD')), tgt)
    assert sorted(got.rows()) == [('A', 'protein_coding', 1), ('B', '', None), ('C', 'lncRNA', 0), ('D', None, 0)]


def lof(rank, kind='lof'):
    return {'constraintType': kind, 'upperRank': rank}


def test_constraint_minmax():
    tgt = targets([
        {'id': 'T1', 'constraint': [lof(10), lof(5, 'mis')]},
        {'id': 'T2', 'constraint': [lof(30)]},
        {'id': 'T3', 'constraint': [lof(20), lof(17)]},
        {'id': 'T4', 'constraint': None},
        {'id': 'T5', 'constraint': []},
        {'id': 'T6', 'constraint': [lof(None)]},
    ])
    got = constraint_query(queryset(['T1', 'T2', 'T3', 'T4', 'T5', 'T6']), tgt).select('targetid', 'cal_score')
    expected = float(2 * (np.float64(7) / np.float64(20)) - 1)
    assert expected != 2 * (7 * (1 / 20)) - 1
    assert sorted(got.rows(), key=str) == sorted(
        [('T1', -1.0), ('T2', 1.0), ('T3', 0.0), ('T3', expected), ('T4', None), ('T5', None), ('T6', None)],
        key=str,
    )


def test_constraint_without_lof_is_null():
    tgt = targets([{'id': 'T1', 'constraint': [lof(5, 'mis')]}])
    got = constraint_query(queryset(['T1']), tgt)
    assert got.select('targetid', 'cal_score').rows() == [('T1', None)]
    assert got.schema['cal_score'] == pl.Float64


def attribute(description):
    return {'pmid': 1, 'description': description, 'name': 'role in cancer'}


def test_driver_gene():
    tgt = targets([
        {'id': 'T1', 'hallmarks': {'attributes': [attribute('oncogene')]}},
        {'id': 'T2', 'hallmarks': {'attributes': [attribute('fusion')]}},
        {'id': 'T3', 'hallmarks': None},
        {'id': 'T4', 'hallmarks': {'attributes': []}},
        {'id': 'T5', 'hallmarks': {'attributes': None}},
        {'id': 'T6', 'hallmarks': {'attributes': [attribute(None), attribute('TSG')]}},
        {'id': 'T7', 'hallmarks': {'attributes': [attribute('oncogene, TSG')]}},
    ])
    got = driver_gene_query(queryset([f'T{i}' for i in range(1, 8)]), tgt).select('targetid', 'counts', 'Nr_CDG')
    assert sorted(got.rows()) == [
        ('T1', 1, -1),
        ('T2', 0, None),
        ('T3', 0, None),
        ('T4', 0, None),
        ('T5', 0, None),
        ('T6', 1, -1),
        ('T7', 0, None),
    ]


# ---------------------------------------------------------------------------
# mouse_model_query / clin_trials_query / baseline_query
# ---------------------------------------------------------------------------


def harmonic(scores):
    total = 0.0
    for i, s in enumerate(sorted(scores, reverse=True)):
        total += s / float((i + 1) * (i + 1))
    return total


def neg_scaled(h, m):
    s = h / m
    return 0.0 if s < 0.6 else (s - 0.6) * -1 / (1 - 0.6)


def mouse_rows():
    return [
        ('G1', [('MP1', 'a'), ('MP2', 'b')]),
        ('G1', [('MP2', 'b')]),
        ('G2', [('MP3', 'c')]),
        ('G3', [('MPX', 'x')]),
        ('G4', None),
        ('G5', []),
        (None, [('MP1', 'a'), ('MP1', 'a'), ('MP1', 'a'), ('MP1', 'a')]),
        ('G7', [('MP4', 'd'), ('MPX', 'x')]),
    ]


def mouse_frame() -> pl.DataFrame:
    rows = [(t, None if c is None else [{'id': i, 'label': lbl} for i, lbl in c]) for t, c in mouse_rows()]
    return frame(rows, MOUSE_SCHEMA)


PHENO = [('MP1', 'a', 1.0), ('MP2', 'b', 0.5), ('MP3', 'c', 0.0), ('MP4', 'd', 0.25)]


def test_mouse_model_full():
    got = mouse_model_query(
        queryset(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7']), mouse_frame(), frame(PHENO, PHENO_SCHEMA)
    ).select('targetid', 'harmonicSum', 'maximum', 'negScaledHarmonicSum')
    m = 1 + 1 / 4 + 1 / 9 + 1 / 16
    h1 = harmonic([1.0, 0.5, 0.5])
    assert neg_scaled(h1, m) < 0
    assert sorted(got.rows(), key=str) == sorted(
        [
            ('G1', h1, m, neg_scaled(h1, m)),
            ('G2', 0.0, m, 0.0),
            ('G3', 0.0, m, 0.0),
            ('G4', None, None, None),
            ('G5', None, None, None),
            ('G6', None, None, None),
            ('G7', 0.25, m, 0.0),
        ],
        key=str,
    )


def test_mouse_model_division_matches_numpy():
    scaled = [0.6, 0.61, 0.7, 0.75, 0.83, 0.9, 0.97, 1.0]
    expected = [(np.float64(s) - 0.6) * -1 / (1 - 0.6) for s in scaled]
    assert any(e != (s - 0.6) * -1 * (1 / (1 - 0.6)) for s, e in zip(scaled, expected, strict=True))
    # one target per scaled value against a maximum of exactly 1.0
    pheno = frame([(f'MP{i}', 'x', s) for i, s in enumerate(scaled)], PHENO_SCHEMA)
    mouse = frame([(f'G{i}', [{'id': f'MP{i}', 'label': 'x'}]) for i in range(len(scaled))], MOUSE_SCHEMA)
    got = dict(
        mouse_model_query(queryset([f'G{i}' for i in range(len(scaled))]), mouse, pheno)
        .select('targetid', 'negScaledHarmonicSum')
        .rows()
    )
    assert [got[f'G{i}'] for i in range(len(scaled))] == expected


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        (' 0.5', 0.5), ('0.5\r', 0.5), ('\t0.5 ', 0.5), ('\x000.5', 0.5), ('0.5d', 0.5), ('0.5F', 0.5),
        ('1e1', 10.0), ('1.', 1.0), ('.5', 0.5), (' 0 ', 0.0), ('Infinity', math.inf), ('-inf', -math.inf),
        ('INF', math.inf), ('abc', None), ('', None), (' ', None), ('+nan', None), ('\x7f0.5', None),
        ('0.5\x7f', None), ('1,5', None), ('0x1p3', None), (None, None),
    ],
)  # fmt: skip
def test_cast_double_matches_spark(text, expected):
    got = pl.DataFrame({'score': [text]}, schema={'score': pl.String}).select(cast_double('score')).item()
    assert got == expected


@pytest.mark.parametrize('text', ['NaN', '+NaN', '-NaN', 'nan'])
def test_cast_double_nan_matches_spark(text):
    got = pl.DataFrame({'score': [text]}).select(cast_double('score')).item()
    assert math.isnan(got)


def test_mouse_model_padded_score_counts():
    pheno = pl.DataFrame({'id': ['MP1'], 'label': ['a'], 'score': [' 0.5\r']})
    pheno = pheno.select('id', 'label', cast_double('score'))
    mouse = frame([('G1', [{'id': 'MP1', 'label': 'a'}])], MOUSE_SCHEMA)
    got = mouse_model_query(queryset(['G1']), mouse, pheno).select('harmonicSum').item()
    assert got == 0.5


def test_clin_trials_cartesian():
    moa = frame(
        [
            (['T1', 'T2'], ['D1']),
            (['T1'], ['D2', 'D3']),
            (['T3'], []),
            (['T4'], ['D3']),
            (['T5'], ['D9']),
            (None, ['D2']),
        ],
        MOA_SCHEMA,
    )
    molecule = frame([('D1', 'PHASE_2'), ('D2', 'APPROVAL'), ('D3', 'WEIRD')], MOLECULE_SCHEMA)
    got = clin_trials_query(queryset(['T1', 'T2', 'T3', 'T4', 'T5']), moa, molecule)
    assert sorted(got.select('targetid', 'inClinicalTrials').rows()) == [
        ('T1', 1.0),
        ('T2', 0.2),
        ('T3', None),
        ('T4', None),
        ('T5', None),
    ]


BASELINE = [
    ('T1', 'gtex', 'UBERON_1', None, 0.9, 0.2),
    ('T1', 'gtex', 'UBERON_2', None, 0.5, 0.1),
    ('T1', 'Tabula_Sapiens', None, 'CL_1', 0.3, 1.0),
    ('T2', 'gtex', 'UBERON_1', 'CL_1', 0.9, 0.2),
    ('T2', 'hpa', 'UBERON_1', None, 0.9, 0.2),
    ('T2', 'gtex', 'UBERON_1', None, None, 0.2),
    ('T3', 'gtex', 'UBERON_2', None, 0.7, 0.6),
    ('T3', 'gtex', 'UBERON_1', None, 0.7, 0.3),
    ('T4', 'gtex', None, None, 0.7, 0.6),
    ('T5', 'gtex', 'UBERON_1', None, 0.7, None),
]


def test_baseline_tie_break():
    got = baseline_query(queryset(['T1', 'T2', 'T3', 'T4', 'T5']), frame(BASELINE, BASELINE_SCHEMA))
    assert sorted(got.rows()) == [
        ('T1', 2.0 * 0.9 - 1.0, 1 - 0.2, 2.0 * 0.3 - 1.0, -1.0),
        ('T2', None, None, None, None),
        # tied specificity: the lowest tissueBiosampleId wins
        ('T3', 2.0 * 0.7 - 1.0, 1 - 0.3, None, None),
        ('T4', None, None, None, None),
        ('T5', 2.0 * 0.7 - 1.0, None, None, None),
    ]


def test_baseline_tie_break_is_order_independent():
    rows = frame(BASELINE, BASELINE_SCHEMA)
    ids = queryset(['T1', 'T2', 'T3', 'T4', 'T5'])
    assert baseline_query(ids, rows).sort('targetid').equals(baseline_query(ids, rows.reverse()).sort('targetid'))


# ---------------------------------------------------------------------------
# compute_target_prioritisation / target_prioritisation
# ---------------------------------------------------------------------------


def inputs() -> dict[str, pl.DataFrame]:
    tgt = targets(
        [{'id': t, 'subcellularLocations': s} for t, s in MEMBRANE_TARGETS]
        + [
            {'id': 'G1', 'constraint': [lof(10)], 'hallmarks': {'attributes': [attribute('TSG')]}},
            {'id': 'G2', 'constraint': [lof(17)], 'hallmarks': {'attributes': None}},
            {'id': 'G3', 'constraint': [lof(30)], 'hallmarks': {'attributes': []}},
        ]
    )
    homology = pl.concat([
        homologue(targetId='T1', queryPercentageIdentity=88.8),
        homologue(targetId='G1', homologyType='ortholog_one2one', speciesName='Mouse', queryPercentageIdentity=88.8),
    ])
    return {
        'targets': tgt,
        'tractability': frame([('T1', 'SM', 'High-Quality Ligand', True)], TRACTABILITY_SCHEMA),
        'target_safety_event': frame([('T2', 'hepatotoxicity')], TARGET_SAFETY_EVENT_SCHEMA),
        'chemical_probes': frame([('T3', True)], CHEMICAL_PROBES_SCHEMA),
        'homology': homology,
        'mouse_phenotypes': mouse_frame(),
        'molecule': frame([('D1', 'PHASE_2'), ('D2', 'APPROVAL')], MOLECULE_SCHEMA),
        'mechanism_of_action': frame([(['T1', 'G2'], ['D1', 'D2'])], MOA_SCHEMA),
        'baseline_expression': frame(
            [(t, *rest) for (_, *rest), t in zip(BASELINE, ['G1'] * 3 + ['T1'] * 7, strict=True)], BASELINE_SCHEMA
        ),
        'uniprot_slterms': frame(UNIPROT, UNIPROT_SCHEMA),
        'mouse_pheno_scores': frame(PHENO, PHENO_SCHEMA),
    }


def two_chunks(df: pl.DataFrame) -> pl.DataFrame:
    half = df.height // 2
    out = pl.concat([df.slice(0, half), df.slice(half)], rechunk=False)
    assert out.n_chunks() == 2 or df.height < 2
    return out


def test_compute_target_prioritisation_schema_and_values():
    got = compute_target_prioritisation(**inputs())
    assert got.schema == OUTPUT_SCHEMA
    assert got.height == 20
    rows = {r['targetId']: r for r in got.iter_rows(named=True) if r['targetId'] != 'T13'}
    assert rows['T1']['hasLigand'] == 1
    assert rows['T1']['paralogMaxIdentityPercentage'] == -((np.float64(88.8) - 60) / 40)
    assert rows['T1']['maxClinicalStage'] == 1.0
    assert rows['T2']['hasSafetyEvent'] == -1
    assert rows['T3']['hasHighQualityChemicalProbes'] == 1
    assert rows['G1']['isCancerDriverGene'] == -1
    assert rows['G1']['mouseOrthologMaxIdentityPercentage'] == (np.float64(88.8) - 80) / 20
    assert rows['G1']['geneticConstraint'] == -1.0
    assert rows['G2']['geneticConstraint'] == 2 * (np.float64(7) / np.float64(20)) - 1
    assert rows['G1']['mouseKOScore'] == neg_scaled(harmonic([1.0, 0.5, 0.5]), 1 + 1 / 4 + 1 / 9 + 1 / 16)
    assert rows['G1']['tissueSpecificity'] == 2.0 * 0.9 - 1.0
    assert rows['G1']['celltypeDistribution'] == -1.0
    assert rows['T1']['tissueSpecificity'] == 2.0 * 0.7 - 1.0
    assert rows['T1']['hasPocket'] == 0
    assert rows['T2']['hasPocket'] is None
    assert [(r['isInMembrane'], r['isSecreted']) for r in got.filter(targetId='T13').iter_rows(named=True)] == [
        (0, 0),
        (None, None),
    ]


def test_compute_target_prioritisation_multi_chunk_input():
    single = compute_target_prioritisation(**inputs())
    chunked = compute_target_prioritisation(**{k: two_chunks(v) for k, v in inputs().items()})
    assert chunked.equals(single)


def test_target_prioritisation_null_and_empty_lists():
    tgt = targets([
        {'id': 'T1', 'subcellularLocations': None, 'constraint': None, 'hallmarks': None},
        {'id': 'T2', 'subcellularLocations': [], 'constraint': [], 'hallmarks': {'attributes': []}},
        {'id': 'T3', 'subcellularLocations': [NULL_LOCATION], 'constraint': [lof(1)], 'hallmarks': None},
    ])
    data = inputs() | {
        'targets': tgt,
        'mouse_phenotypes': frame([('T1', None), ('T2', []), ('T3', [{'id': 'MP1', 'label': 'a'}])], MOUSE_SCHEMA),
        'mechanism_of_action': frame([(None, ['D1']), (['T1'], None), ([], ['D1']), (['T2'], [])], MOA_SCHEMA),
    }
    got = compute_target_prioritisation(**data).sort('targetId')
    assert got['targetId'].to_list() == ['T1', 'T2', 'T3']
    assert got['isInMembrane'].to_list() == [None, None, None]
    assert got['isSecreted'].to_list() == [None, None, None]
    # a single lof rank divides by zero
    assert got['geneticConstraint'].to_list() == [None, None, None]
    assert got['isCancerDriverGene'].to_list() == [None, None, None]
    assert got['mouseKOScore'].to_list() == [None, None, -1.0]
    assert got['maxClinicalStage'].to_list() == [None, None, None]






