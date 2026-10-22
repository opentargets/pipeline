"""Tests for the target_safety_event transformer."""

from __future__ import annotations

import polars as pl

from pts.schemas.target_safety_event import biosample, effect, study
from pts.transformers.target_safety_event import (
    build_target_safety_events,
    clean_phenotype_to_describe_safety_event,
    harmonize_safety_evidence,
    process_adverse_events,
    process_aop,
    process_brennan,
    process_pharmacogenetics,
    process_safety_risk,
    process_toxcast,
)
from pts.transformers.utils.target import ensg_lookup

TOXCAST_URL = 'https://www.epa.gov/chemical-research/exploring-toxcast-data-downloadable-data'
PREFIX = 'Genetic variation linked to this safety liability: '

SAFETY_SCHEMA = {
    'id': pl.String,
    'targetFromSourceId': pl.String,
    'event': pl.String,
    'eventId': pl.String,
    'datasource': pl.String,
    'effects': pl.List(effect),
    'literature': pl.String,
    'url': pl.String,
    'biosamples': pl.List(biosample),
    'studies': pl.List(study),
}
TARGET_SCHEMA = {
    'id': pl.String,
    'approvedSymbol': pl.String,
    'proteinIds': pl.List(pl.Struct({'id': pl.String, 'source': pl.String})),
}
DISEASE_SCHEMA = {'id': pl.String, 'obsoleteTerms': pl.List(pl.String)}
AE_COLUMNS = [
    'symptom', 'biologicalSystem', 'effect', 'efoId', 'ensemblId', 'pmid', 'ref', 'target', 'uberonCode', 'url',
]
SR_COLUMNS = [
    'biologicalSystem', 'ensemblId', 'event', 'eventId', 'liability', 'pmid', 'ref', 'target', 'uberonId', 'url',
]
TC_COLUMNS = [
    'assay_component_endpoint_name', 'assay_component_desc', 'biological_process_target', 'tissue', 'cell_format',
    'cell_short_name', 'assay_format_type', 'official_symbol', 'eventId',
]
AOP_SCHEMA = {
    'id': pl.String,
    'event': pl.String,
    'eventId': pl.String,
    'datasource': pl.String,
    'url': pl.String,
    'effects': pl.List(effect),
    'biosamples': pl.List(biosample),
}
BRENNAN_SCHEMA = {
    'id': pl.String,
    'event': pl.String,
    'eventId': pl.String,
    'datasource': pl.String,
    'literature': pl.String,
    'effects': effect,
    'studies': study,
    'biosamples': biosample,
}
PGX_SCHEMA = {
    'pgxCategory': pl.String,
    'targetFromSourceId': pl.String,
    'phenotypeText': pl.String,
    'genotypeId': pl.String,
    'haplotypeId': pl.String,
    'drugs': pl.List(pl.Struct({'drugFromSource': pl.String, 'drugId': pl.String})),
}


def _strings(columns: list[str], rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=dict.fromkeys(columns, pl.String), orient='row')


def _ae(*rows: tuple) -> pl.DataFrame:
    return _strings(AE_COLUMNS, list(rows))


def _brennan(event_id: str | None, effects: dict | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        [{
            'id': 'ENSG00000082701',
            'event': 'Mitochondrial toxicity',
            'eventId': event_id,
            'datasource': 'Brennan et al. (2024)',
            'literature': '38773351',
            'effects': effects if effects is not None else {'direction': 'Activation', 'dosing': None},
            'studies': {'description': None, 'name': None, 'type': 'clinical'},
            'biosamples': None,
        }],
        schema=BRENNAN_SCHEMA,
    )


def _pgx(*rows: tuple) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                'pgxCategory': cat,
                'targetFromSourceId': target,
                'phenotypeText': text,
                'genotypeId': genotype,
                'haplotypeId': haplotype,
                'drugs': None if drugs is None else [{'drugFromSource': d, 'drugId': None} for d in drugs],
            }
            for cat, target, text, genotype, haplotype, drugs in rows
        ],
        schema=PGX_SCHEMA,
    )


def _safety(*rows: dict) -> pl.DataFrame:
    base = dict.fromkeys(SAFETY_SCHEMA)
    return pl.DataFrame([{**base, 'event': 'hepatotoxicity', 'eventId': 'EFO_0001234', **r} for r in rows],
                        schema=SAFETY_SCHEMA)


def _target(*rows: tuple) -> pl.DataFrame:
    return pl.DataFrame(
        [{'id': i, 'approvedSymbol': s, 'proteinIds': None if p is None else [{'id': x, 'source': 'u'} for x in p]}
         for i, s, p in rows],
        schema=TARGET_SCHEMA,
    )


def _diseases(*rows: tuple) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=DISEASE_SCHEMA, orient='row')


NO_DISEASES = _diseases(('EFO_9999', []))


def _clean(*texts: str | None) -> list[str | None]:
    df = pl.DataFrame({'t': list(texts)}, schema={'t': pl.String})
    return df.select(clean_phenotype_to_describe_safety_event(pl.col('t'))).to_series().to_list()


# ---------------------------------------------------------------------------
# process_* source harmonizers
# ---------------------------------------------------------------------------


def test_process_adverse_events_maps_columns() -> None:
    row = ('bronchoconstriction', 'gastrointestinal', 'activation_general', 'EFO_0009836', 'ENSG00000133019',
           '23197038', 'Bowes et al. (2012)', 'CHRM3', 'UBERON_0005409', None)
    out = process_adverse_events(_ae(row)).row(0, named=True)
    assert out['id'] == 'ENSG00000133019'
    assert out['event'] == 'bronchoconstriction'
    assert out['eventId'] == 'EFO_0009836'
    assert out['datasource'] == 'Bowes et al. (2012)'
    assert out['literature'] == '23197038'
    assert out['effects'] == [{'direction': 'Activation/Increase/Upregulation', 'dosing': 'general'}]
    assert out['biosample'] == {
        'tissueLabel': 'gastrointestinal', 'tissueId': 'UBERON_0005409', 'cellLabel': None, 'cellFormat': None,
        'cellId': None,
    }
    assert out['study'] == {'description': 'bronchoconstriction', 'name': 'EFO_0009836', 'type': 'preclinical'}


def test_process_adverse_events_groups_effects_and_reads_text_as_spark() -> None:
    base = ('e', 's', None, 'EFO_1', 'ENSG1', None, 'Urban et al. (2012)', 'T', 'U', None)
    df = _ae(
        (*base[:2], 'activation_general', *base[3:5], '0042', *base[6:]),
        (*base[:2], 'inhibition_acute', *base[3:5], '7', *base[6:]),
        (*base[:2], 'activation', *base[3:]),
        (None, 's', 'activation_acute', 'EFO_1', 'ENSG1', None, 'Unknown (2000)', 'T', 'U', None),
        ('f', 's', None, 'EFO_1', 'ENSG1', None, 'Unknown (2000)', 'T', 'U', None),
    )
    out = process_adverse_events(df).sort('literature', 'event', nulls_last=True)
    rows = out.select('event', 'literature', 'effects', pl.col('study').struct.field('type')).rows()
    grouped = [
        {'direction': 'Activation/Increase/Upregulation', 'dosing': None},
        {'direction': 'Activation/Increase/Upregulation', 'dosing': 'general'},
        {'direction': 'Inhibition/Decrease/Downregulation', 'dosing': 'acute'},
    ]
    assert rows == [
        ('e', '42', grouped, 'clinical'),
        ('e', '7', grouped, 'clinical'),
        ('e', None, grouped, 'clinical'),
        ('f', None, [{'direction': None, 'dosing': None}], 'Unknown (2000)'),
        (None, None, None, 'Unknown (2000)'),
    ]


def test_process_toxcast_maps_columns() -> None:
    df = _strings(TC_COLUMNS, [
        ('ACEA_ER_80hr', 'some assay', 'cell proliferation', None, 'cell line', 'T47D', 'cell-based', ' ESR1 ', None),
        ('A', 'd', 'e', None, None, None, None, '\tESR1 ', None),
    ])
    out = process_toxcast(df)
    assert out['targetFromSourceId'].to_list() == ['ESR1', '\tESR1']
    first = out.row(0, named=True)
    assert first['event'] == 'cell proliferation'
    assert first['datasource'] == 'ToxCast'
    assert first['url'] == TOXCAST_URL
    assert first['biosample'] == {
        'tissueLabel': None, 'tissueId': None, 'cellLabel': 'T47D', 'cellFormat': 'cell line', 'cellId': None,
    }
    assert first['study'] == {'description': 'some assay', 'name': 'ACEA_ER_80hr', 'type': 'cell-based'}
    assert 'id' not in out.columns


def test_process_safety_risk_maps_columns() -> None:
    df = _strings(SR_COLUMNS, [
        ('cardiovascular', 'ENSG1', 'x', 'y', 'Important', '21283106', 'Force et al. (2011)', 'T', 'UBERON_1', None),
        ('heart', 'ENSG5', 'x', 'y', 'Arrhythmia', '22', 'Lamore et al. (2017)', 'T', 'UBERON_2', None),
        ('other', 'ENSG1', 'some event', 'EFO_A', 'lia', '1', 'Someone (2020)', 'T', 'UBERON_3', 'u'),
        ('other', 'ENSG1', 'some event', 'EFO_A', 'lia', '1', None, 'T', 'UBERON_3', 'u'),
    ])
    out = process_safety_risk(df)
    assert out.select('id', 'event', 'eventId', 'literature', 'datasource', 'study').rows() == [
        ('ENSG1', 'heart disease', 'EFO_0003777', '21283106', 'Force et al. (2011)',
         {'description': 'Important', 'name': None, 'type': 'preclinical'}),
        ('ENSG5', 'cardiac arrhythmia', 'EFO_0004269', '22', 'Lamore et al. (2017)',
         {'description': 'Arrhythmia', 'name': None, 'type': 'cell-based'}),
        ('ENSG1', None, None, '1', 'Someone (2020)', {'description': 'lia', 'name': None, 'type': None}),
        ('ENSG1', None, None, '1', None, {'description': 'lia', 'name': None, 'type': None}),
    ]
    assert out['biosample'][0] == {
        'tissueLabel': 'cardiovascular', 'tissueId': 'UBERON_1', 'cellLabel': None, 'cellFormat': None, 'cellId': None,
    }
    assert 'url' not in out.columns


def test_process_aop_maps_columns() -> None:
    liver = {'tissueLabel': 'liver', 'tissueId': 'UBERON:1', 'cellLabel': None, 'cellFormat': None, 'cellId': None}
    kidney = {'tissueLabel': 'kidney', 'tissueId': 'UBERON:2', 'cellLabel': 'c', 'cellFormat': 'f', 'cellId': 'i'}
    df = pl.DataFrame(
        [
            {'id': 'ENSG1', 'event': 'NAFLD', 'url': 'u1', 'effects': [{'direction': 'Activation', 'dosing': None}]},
            {'id': 'ENSG5', 'event': 'NA  liver injury ', 'url': 'u2', 'biosamples': [liver, kidney]},
            {'id': 'ENSG1', 'event': 'x', 'url': 'u3', 'effects': [{'direction': 'Other', 'dosing': 'acute'}],
             'biosamples': []},
            {'id': 'ENSG1', 'event': 'y', 'url': 'u4', 'effects': [{'direction': 'Inhibition', 'dosing': None}]},
        ],
        schema=AOP_SCHEMA,
    )
    out = process_aop(df)
    assert out.select('id', 'event', 'url', 'effects', 'biosample').rows() == [
        ('ENSG1', 'FLD', 'u1', [{'direction': 'Activation/Increase/Upregulation', 'dosing': None}], None),
        ('ENSG5', 'liver injury', 'u2', None, liver),
        ('ENSG5', 'liver injury', 'u2', None, kidney),
        ('ENSG1', 'x', 'u3', [{'direction': None, 'dosing': 'acute'}], None),
        ('ENSG1', 'y', 'u4', [{'direction': 'Inhibition/Decrease/Downregulation', 'dosing': None}], None),
    ]
    assert out['study'].unique().to_list() == [{'description': None, 'name': None, 'type': 'cell-based'}]


def test_process_brennan_normalises_empty_string_event_id() -> None:
    assert process_brennan(_brennan('')).item(0, 'eventId') is None


def test_process_brennan_normalises_whitespace_only_event_id() -> None:
    assert process_brennan(_brennan('  ')).item(0, 'eventId') is None


def test_process_brennan_trims_a_real_event_id() -> None:
    assert process_brennan(_brennan(' EFO_0009836 ')).item(0, 'eventId') == 'EFO_0009836'


def test_process_brennan_keeps_real_event_id() -> None:
    assert process_brennan(_brennan('EFO_0009836')).item(0, 'eventId') == 'EFO_0009836'


def test_process_brennan_maps_effects_study_and_biosample() -> None:
    df = pl.concat(
        [
            _brennan('a', {'direction': 'Inhibition', 'dosing': 'acute'}),
            _brennan('b', {'direction': 'Other', 'dosing': None}),
            _brennan('c').with_columns(pl.lit(None, dtype=effect).alias('effects'), pl.lit(None, dtype=study).alias(
                'studies')),
        ],
        rechunk=False,
    )
    out = process_brennan(df)
    assert out['effects'].to_list() == [
        [{'direction': 'Inhibition/Decrease/Downregulation', 'dosing': 'acute'}],
        [{'direction': 'Other', 'dosing': None}],
        [{'direction': None, 'dosing': None}],
    ]
    assert out['study'].to_list()[2] == {'description': None, 'name': None, 'type': None}
    assert out['biosample'].to_list() == [None, None, None]
    assert out.schema['biosample'] == biosample


def test_process_pharmacogenetics_filters_and_explodes() -> None:
    df = _pgx(
        ('toxicity', 'ENSG1', 'increased risk of toxicity', 'g1', 'h1', ['drugA', 'drugB']),
        ('toxicity', 'ENSG5', 'neutropenia when taking X', None, None, [None, 'drugC']),
        ('toxicity', 'ENSG5', 'no significant association with rash', 'g', None, ['d']),
        ('toxicity', 'ENSG5', 'rash not associated with d', 'g', None, ['d']),
        ('toxicity', 'ENSG5', 'response', 'g', None, ['d']),
        ('efficacy', 'ENSG5', 'rash', 'g', None, ['d']),
        ('toxicity', None, 'rash', 'g', None, ['d']),
        ('toxicity', 'ENSG4', None, 'g', None, ['d']),
        ('toxicity', 'ENSG1', 'rash', 'g3', 'g3', ['drugD']),
        ('toxicity', 'ENSG1', 'hepatotoxicity', 'g4', None, None),
        ('toxicity', 'ENSG1', 'hepatotoxicity', 'g4', None, []),
    )
    out = process_pharmacogenetics(df)
    rows = sorted(
        out.select('id', 'event', 'eventId', 'datasource', 'url', 'supporting_variation',
                   pl.col('study').struct.field('name')).rows(),
        key=repr,
    )
    url = 'https://www.clinpgx.org/search?query='
    assert rows == sorted(
        [
            ('ENSG1', 'drug toxicity', None, 'ClinPGx', url + 'ENSG1', 'g1', 'drugA induced effect'),
            ('ENSG1', 'drug toxicity', None, 'ClinPGx', url + 'ENSG1', 'h1', 'drugA induced effect'),
            ('ENSG1', 'drug toxicity', None, 'ClinPGx', url + 'ENSG1', 'g1', 'drugB induced effect'),
            ('ENSG1', 'drug toxicity', None, 'ClinPGx', url + 'ENSG1', 'h1', 'drugB induced effect'),
            ('ENSG5', 'neutropenia', None, 'ClinPGx', url + 'ENSG5', None, None),
            ('ENSG5', 'neutropenia', None, 'ClinPGx', url + 'ENSG5', None, 'drugC induced effect'),
            ('ENSG1', 'rash', None, 'ClinPGx', url + 'ENSG1', 'g3', 'drugD induced effect'),
        ],
        key=repr,
    )
    assert out['study'].struct.field('description').null_count() == out.height
    assert set(out['study'].struct.field('type')) == {'clinical'}


# ---------------------------------------------------------------------------
# clean_phenotype_to_describe_safety_event
# ---------------------------------------------------------------------------


def test_clean_phenotype_maps_toxicity_to_drug_toxicity() -> None:
    assert _clean('toxicity') == ['drug toxicity']


def test_clean_phenotype_matches_spark() -> None:
    cases = {
        'toxicity': 'drug toxicity',
        'increased risk of toxicity': 'drug toxicity',
        'neutropenia when taking X': 'neutropenia',
        '  rash  ': 'rash',
        'NA': 'NA',
        'risk': 'drug toxicity',
        'response': 'drug response',
        'altered toxicity': 'altered toxicity',
        'risk and toxicity': 'risk and toxicity',
        'increased risk and increased severity of nausea': 'nausea',
        'more likely to develop rash based on dose': 'rash',
        'rash\nas a result of taking y': '',
        'rash\u2028x': '',
        '': '',
    }
    assert _clean(*cases, None) == [*cases.values(), None]


def test_phenotype_regex_ascii_whitespace() -> None:
    assert _clean('severe\u00a0\u00a0rash\t\tx', 'a\x0bb\x0cc', 'r\u3000s') == [
        'severe\u00a0\u00a0rash x',
        'a b c',
        'r\u3000s',
    ]


# ---------------------------------------------------------------------------
# harmonize_safety_evidence
# ---------------------------------------------------------------------------


def test_harmonize_safety_evidence_dedupes_identical_source_rows() -> None:
    row = ('bronchoconstriction', 'gastrointestinal', 'activation_general', 'EFO_0009836', 'ENSG00000133019',
           '23197038', 'Bowes et al. (2012)', 'CHRM3', 'UBERON_0005409', None)
    empty_toxcast = process_toxcast(_strings(TC_COLUMNS, []))
    result = harmonize_safety_evidence([process_adverse_events(_ae(row, row)), empty_toxcast])
    assert result.height == 1


def test_union_struct_fields_by_name() -> None:
    ae = ('e', 'gastrointestinal', 'activation_general', 'EFO_1', 'ENSG1', '1', 'Bowes et al. (2012)', 'T', 'U1',
          None)
    brennan_biosample = pl.Struct(dict.fromkeys(['cellFormat', 'cellLabel', 'tissueId', 'tissueLabel'], pl.String))
    brennan = _brennan('EFO_1').with_columns(
        pl.lit({'cellFormat': None, 'cellLabel': None, 'tissueId': 'U2', 'tissueLabel': 'GI'},
               dtype=brennan_biosample).alias('biosamples')
    )
    aop_fields = ['cellFormat', 'cellId', 'cellLabel', 'tissueId', 'tissueLabel']
    aop_biosample = pl.Struct(dict.fromkeys(aop_fields, pl.String))
    aop = pl.DataFrame(
        [{'id': 'ENSG1', 'event': 'x', 'datasource': 'AOP-Wiki', 'biosamples': [
            {'cellFormat': 'f', 'cellId': 'i', 'cellLabel': 'c', 'tissueId': 'U3', 'tissueLabel': 'liver'}]}],
        schema={**AOP_SCHEMA, 'biosamples': pl.List(aop_biosample)},
    )
    result = harmonize_safety_evidence([process_adverse_events(_ae(ae)), process_brennan(brennan), process_aop(aop)])
    assert result.schema['biosamples'] == pl.List(biosample)
    assert result.schema['studies'] == pl.List(study)
    assert result.schema['effects'] == pl.List(effect)
    by_source = dict(result.select('datasource', 'biosamples').rows())
    assert by_source == {
        'Bowes et al. (2012)': [
            {'tissueLabel': 'gastrointestinal', 'tissueId': 'U1', 'cellLabel': None, 'cellFormat': None,
             'cellId': None}],
        'Brennan et al. (2024)': [
            {'tissueLabel': 'GI', 'tissueId': 'U2', 'cellLabel': None, 'cellFormat': None, 'cellId': None}],
        'AOP-Wiki': [{'tissueLabel': 'liver', 'tissueId': 'U3', 'cellLabel': 'c', 'cellFormat': 'f', 'cellId': 'i'}],
    }


def test_clinpgx_description_sorted_join() -> None:
    df = _pgx(
        ('toxicity', 'ENSG1', 'rash', 'h1', 'g2', ['drugB', 'drugA']),
        ('toxicity', 'ENSG1', 'rash', 'g1', 'g1', ['drugA']),
        ('toxicity', 'ENSG5', 'rash', None, None, ['drugC']),
    )
    result = harmonize_safety_evidence([process_pharmacogenetics(df)]).sort('id')
    assert result['studies'].to_list() == [
        [
            {'description': PREFIX + 'g1, g2, h1', 'name': 'drugA induced effect', 'type': 'clinical'},
            {'description': PREFIX + 'g1, g2, h1', 'name': 'drugB induced effect', 'type': 'clinical'},
        ],
        [{'description': PREFIX, 'name': 'drugC induced effect', 'type': 'clinical'}],
    ]


def test_target_safety_event_null_and_empty_lists() -> None:
    aop = pl.DataFrame(
        [
            {'id': 'ENSG1', 'event': 'a', 'effects': None, 'biosamples': None},
            {'id': 'ENSG1', 'event': 'b', 'effects': [], 'biosamples': []},
        ],
        schema=AOP_SCHEMA,
    )
    pgx = _pgx(
        ('toxicity', 'ENSG1', 'rash', 'g', None, None),
        ('toxicity', 'ENSG1', 'rash', 'g', None, []),
    )
    diseases = _diseases(('EFO_A', None), ('EFO_B', []))
    safety = harmonize_safety_evidence([process_aop(aop), process_pharmacogenetics(pgx)])
    result = build_target_safety_events(safety, _target(('ENSG1', None, None)), diseases).sort('event')
    assert result.select('event', 'effects', 'biosamples').rows() == [('a', None, None), ('b', [], None)]


# ---------------------------------------------------------------------------
# ensg_lookup
# ---------------------------------------------------------------------------


def test_ensg_lookup_fans_out_and_tolerates_null_proteins():
    target = pl.LazyFrame(
        {
            'id': ['ENSG1', 'ENSG2', 'ENSG3'],
            'approvedSymbol': ['A', 'B', None],
            'proteinIds': [[{'id': 'P1'}, {'id': 'P1'}], None, [{'id': 'A'}]],
        }
    )
    got = ensg_lookup(target).collect().sort('ensgId', 'name').rows()
    assert got == [('ENSG1', 'A'), ('ENSG1', 'P1'), ('ENSG2', 'B'), ('ENSG3', 'A')]


def test_ensg_lookup_includes_approved_symbol_and_protein_id() -> None:
    got = ensg_lookup(_target(('ENSG00000001', 'GENE1', ['P12345'])).lazy()).collect()
    assert sorted(got.rows()) == [('ENSG00000001', 'GENE1'), ('ENSG00000001', 'P12345')]


def test_ensg_lookup_handles_empty_protein_ids() -> None:
    got = ensg_lookup(_target(('ENSG00000002', 'GENE2', [])).lazy()).collect()
    assert got.rows() == [('ENSG00000002', 'GENE2')]


def test_ensg_lookup_handles_null_protein_ids() -> None:
    got = ensg_lookup(_target(('ENSG00000003', 'MIR122', None)).lazy()).collect()
    assert got.rows() == [('ENSG00000003', 'MIR122')]


def test_ensg_lookup_output_columns() -> None:
    got = ensg_lookup(_target(('ENSG00000003', 'G3', [])).lazy()).collect()
    assert got.columns == ['ensgId', 'name']


# ---------------------------------------------------------------------------
# build_target_safety_events
# ---------------------------------------------------------------------------


def test_build_target_safety_events_output_columns() -> None:
    result = build_target_safety_events(
        _safety({'id': 'ENSG00000001'}), _target(('ENSG00000001', 'G1', [])), NO_DISEASES
    )
    assert result.columns == [
        'targetId', 'event', 'eventId', 'effects', 'biosamples', 'datasource', 'literature', 'url', 'studies',
    ]


def test_build_target_safety_events_one_row_per_liability() -> None:
    safety = _safety(
        {'id': 'ENSG00000001', 'event': 'hepatotoxicity', 'eventId': 'EFO_0001'},
        {'id': 'ENSG00000001', 'event': 'cardiotoxicity', 'eventId': 'EFO_0002'},
    )
    result = build_target_safety_events(safety, _target(('ENSG00000001', 'G1', [])), NO_DISEASES)
    assert result.height == 2
    assert result.filter(pl.col('targetId') == 'ENSG00000001').height == 2


def test_build_target_safety_events_resolves_symbol_to_ensg() -> None:
    safety = _safety({'targetFromSourceId': 'GENE1', 'datasource': 'ToxCast'})
    result = build_target_safety_events(safety, _target(('ENSG00000001', 'GENE1', [])), NO_DISEASES)
    assert result['targetId'].to_list() == ['ENSG00000001']


def test_build_target_safety_events_resolves_protein_id_to_ensg() -> None:
    safety = _safety({'targetFromSourceId': 'P12345', 'datasource': 'AstraZeneca'})
    result = build_target_safety_events(safety, _target(('ENSG00000002', 'GENE2', ['P12345'])), NO_DISEASES)
    assert result['targetId'].to_list() == ['ENSG00000002']


def test_build_target_safety_events_resolves_symbol_for_non_coding_gene() -> None:
    safety = _safety({'targetFromSourceId': 'MIR122', 'datasource': 'ToxCast'})
    result = build_target_safety_events(safety, _target(('ENSG_MIR122', 'MIR122', None)), NO_DISEASES)
    assert result['targetId'].to_list() == ['ENSG_MIR122']


def test_build_target_safety_events_drops_unresolvable_rows() -> None:
    safety = _safety({'targetFromSourceId': 'UNKNOWN_SYMBOL', 'datasource': 'ToxCast'})
    result = build_target_safety_events(safety, _target(('ENSG00000001', 'GENE1', [])), NO_DISEASES)
    assert result.height == 0


def test_unknown_target_dropped_after_symbol_resolution() -> None:
    safety = _safety(
        {'targetFromSourceId': 'SHARED', 'datasource': 'ToxCast'},
        {'targetFromSourceId': 'UNKNOWN', 'datasource': 'ToxCast'},
        {'targetFromSourceId': ' SHARED', 'datasource': 'ToxCast'},
        {'id': 'ENSG_STALE', 'datasource': 'AstraZeneca'},
        {'id': 'ENSG4', 'datasource': 'AstraZeneca'},
    )
    target = _target(('ENSG2', 'SHARED', None), ('ENSG3', 'SHARED', ['SHARED']), ('ENSG4', None, None))
    result = build_target_safety_events(safety, target, NO_DISEASES)
    assert sorted(result.select('targetId', 'datasource').rows()) == [
        ('ENSG2', 'ToxCast'),
        ('ENSG3', 'ToxCast'),
        ('ENSG4', 'AstraZeneca'),
    ]


def test_build_target_safety_events_drops_ids_not_in_target() -> None:
    safety = _safety(
        {'id': 'ENSG00000001', 'event': 'hepatotoxicity'},
        {'id': 'ENSG_STALE_ID', 'event': 'cardiotoxicity'},
    )
    result = build_target_safety_events(safety, _target(('ENSG00000001', 'G1', [])), NO_DISEASES)
    assert result['targetId'].to_list() == ['ENSG00000001']


def test_build_target_safety_events_remaps_obsolete_efo() -> None:
    safety = _safety({'id': 'ENSG00000001', 'eventId': 'EFO_OBSOLETE'})
    diseases = _diseases(('EFO_CURRENT', ['EFO_OBSOLETE']))
    result = build_target_safety_events(safety, _target(('ENSG00000001', 'G1', [])), diseases)
    assert result['eventId'].to_list() == ['EFO_CURRENT']


def test_obsolete_efo_remapped() -> None:
    safety = _safety(
        {'id': 'ENSG1', 'event': 'a', 'eventId': 'EFO_OLD'},
        {'id': 'ENSG1', 'event': 'b', 'eventId': 'EFO_OLD2'},
        {'id': 'ENSG1', 'event': 'c', 'eventId': 'EFO_CURRENT'},
        {'id': 'ENSG1', 'event': 'd', 'eventId': None},
    )
    diseases = _diseases(
        ('EFO_NEW', ['EFO_OLD']), ('EFO_NEW2', ['EFO_OLD', 'EFO_OLD2', None]), ('EFO_A', None), ('EFO_B', []),
    )
    result = build_target_safety_events(safety, _target(('ENSG1', 'G1', [])), diseases)
    assert sorted(result.select('event', 'eventId').rows(), key=repr) == [
        ('a', 'EFO_NEW'),
        ('a', 'EFO_NEW2'),
        ('b', 'EFO_NEW2'),
        ('c', 'EFO_CURRENT'),
        ('d', None),
    ]


def test_build_target_safety_events_keeps_current_efo_unchanged() -> None:
    safety = _safety({'id': 'ENSG00000001', 'eventId': 'EFO_CURRENT'})
    diseases = _diseases(('EFO_OTHER', ['EFO_SOMETHINGELSE']))
    result = build_target_safety_events(safety, _target(('ENSG00000001', 'G1', [])), diseases)
    assert result['eventId'].to_list() == ['EFO_CURRENT']


def test_build_target_safety_events_multiple_targets() -> None:
    safety = _safety(
        {'id': 'ENSG00000001', 'event': 'hepatotoxicity'},
        {'id': 'ENSG00000002', 'event': 'cardiotoxicity'},
    )
    target = _target(('ENSG00000001', 'G1', []), ('ENSG00000002', 'G2', []))
    result = build_target_safety_events(safety, target, NO_DISEASES)
    assert result.height == 2


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


AOP = [
    {'id': 'ENSG1', 'event': 'NAFLD', 'eventId': None, 'datasource': 'AOP-Wiki', 'url': 'u1',
     'isHumanApplicable': True, 'effects': [{'direction': 'Activation', 'dosing': None}], 'biosamples': None},
    {'id': 'ENSG5', 'event': 'NA  liver injury ', 'eventId': None, 'datasource': 'AOP-Wiki', 'url': 'u2',
     'isHumanApplicable': False, 'effects': None, 'biosamples': [
         {'tissueLabel': 'liver', 'tissueId': 'UBERON:1', 'cellLabel': None, 'cellFormat': None, 'cellId': None},
         {'tissueLabel': 'kidney', 'tissueId': 'UBERON:2', 'cellLabel': 'c', 'cellFormat': 'f', 'cellId': 'i'},
     ]},
    {'id': 'ENSG1', 'event': 'x', 'eventId': None, 'datasource': 'AOP-Wiki', 'url': 'u3',
     'isHumanApplicable': True, 'effects': [{'direction': 'Other', 'dosing': 'acute'}], 'biosamples': []},
    {'id': 'ENSG1', 'event': 'y', 'eventId': None, 'datasource': 'AOP-Wiki', 'url': 'u4', 'isHumanApplicable': True,
     'effects': [{'direction': 'Inhibition', 'dosing': 'chronic'}, {'direction': 'Activation', 'dosing': 'acute'}],
     'biosamples': None},
]
GI = {'cellFormat': None, 'cellLabel': None, 'tissueId': 'UBERON_1', 'tissueLabel': 'GI'}








def _canonical(value: object) -> object:
    if isinstance(value, dict):
        return tuple(_canonical(v) for v in value.values())
    if isinstance(value, list):
        return sorted((_canonical(v) for v in value), key=repr)
    return value






