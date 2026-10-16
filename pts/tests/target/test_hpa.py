"""The HPA subcellular location builder."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from pts.target.hpa import build_hpa, read_hpa, read_hpa_ssl
from tests.target.rows import rows

HPA_HEADER = 'Gene\tGene name\tReliability\tMain location\tAdditional location\tExtracellular location'
HPA_LINES = [
    'ENSG00000000001\tA\tApproved\tCytosol;Nucleoplasm\tCytosol\t',
    'ENSG00000000002\tB\tApproved\tVesicles\t\tPredicted to be secreted',
    'ENSG00000000003\tC\tUncertain\t\t\t',
]
SSL_LINES = ['HPA_location\ttermSL\tlabelSL', 'Cytosol\tSL-0091\tCytosol', 'Nucleoplasm\tSL-0190\tNucleus']


def _write(tmp_path: Path) -> tuple[str, str]:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('subcellular_location.tsv', '\n'.join([HPA_HEADER, *HPA_LINES]) + '\n')
    hpa = tmp_path / 'subcellular_location.tsv.zip'
    hpa.write_bytes(buffer.getvalue())
    ssl = tmp_path / 'subcellular_locations_ssl.tsv'
    ssl.write_text('\n'.join(SSL_LINES) + '\n')
    return str(hpa), str(ssl)


def _by_gene(tmp_path: Path) -> dict:
    hpa, ssl = _write(tmp_path)
    return {r['id']: r['locations'] for r in build_hpa(read_hpa(hpa), read_hpa_ssl(ssl)).to_dicts()}


def test_main_additional_and_extracellular_in_order(tmp_path: Path) -> None:
    assert _by_gene(tmp_path)['ENSG00000000001'] == [
        {
            'location': 'Cytosol',
            'source': 'HPA_main',
            'termSL': 'SL-0091',
            'labelSL': 'Cytosol',
            'targetModifier': None,
        },
        {
            'location': 'Nucleoplasm',
            'source': 'HPA_main',
            'termSL': 'SL-0190',
            'labelSL': 'Nucleus',
            'targetModifier': None,
        },
        {
            'location': 'Cytosol',
            'source': 'HPA_additional',
            'termSL': 'SL-0091',
            'labelSL': 'Cytosol',
            'targetModifier': None,
        },
    ]


def test_unmapped_location_keeps_null_terms(tmp_path: Path) -> None:
    assert _by_gene(tmp_path)['ENSG00000000002'][1] == {
        'location': 'Predicted to be secreted',
        'source': 'HPA_extracellular_location',
        'termSL': None,
        'labelSL': None,
        'targetModifier': None,
    }


def test_a_gene_without_locations_is_absent(tmp_path: Path) -> None:
    assert 'ENSG00000000003' not in _by_gene(tmp_path)


def test_matches_the_spark_output(tmp_path: Path) -> None:
    hpa, ssl = _write(tmp_path)
    assert rows(build_hpa(read_hpa(hpa), read_hpa_ssl(ssl)), 'id', ('locations',)) == SPARK_ROWS


#: The spark output for this fixture, as `rows` normalises it.
SPARK_ROWS = [
    {
        'id': 'ENSG00000000001',
        'locations': [
            {
                'location': 'Cytosol',
                'source': 'HPA_additional',
                'termSL': 'SL-0091',
                'labelSL': 'Cytosol',
                'targetModifier': None,
            },
            {
                'location': 'Cytosol',
                'source': 'HPA_main',
                'termSL': 'SL-0091',
                'labelSL': 'Cytosol',
                'targetModifier': None,
            },
            {
                'location': 'Nucleoplasm',
                'source': 'HPA_main',
                'termSL': 'SL-0190',
                'labelSL': 'Nucleus',
                'targetModifier': None,
            },
        ],
    },
    {
        'id': 'ENSG00000000002',
        'locations': [
            {
                'location': 'Predicted to be secreted',
                'source': 'HPA_extracellular_location',
                'termSL': None,
                'labelSL': None,
                'targetModifier': None,
            },
            {'location': 'Vesicles', 'source': 'HPA_main', 'termSL': None, 'labelSL': None, 'targetModifier': None},
        ],
    },
]
