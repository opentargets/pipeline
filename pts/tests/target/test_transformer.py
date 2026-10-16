"""The target transformer end to end, on a tiny core dump, ChEMBL dump and inputs."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from otter.config.model import Config

from pts.schemas.target import TARGET
from pts.transformers.target import target
from pts.transformers.utils.dataset import scan_dataset
from tests.target.fixtures import write_core_dump, write_inputs

pytestmark = pytest.mark.pgserver


@pytest.fixture(scope='module')
def chembl_dump(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """A ChEMBL pg_dump holding the five classification tables."""
    from pixeltable_pgserver.postgres_server import get_server

    path = tmp_path_factory.mktemp('chembl')
    server = get_server(path / 'pgdata', cleanup_mode='delete')
    try:
        server.psql(
            'CREATE TABLE public.target_dictionary (tid int, chembl_id text, pref_name text, target_type text);'
            'CREATE TABLE public.target_components (targcomp_id int, tid int, component_id int);'
            'CREATE TABLE public.component_sequences (component_id int, accession text);'
            'CREATE TABLE public.component_class (comp_class_id int, component_id int, protein_class_id int);'
            'CREATE TABLE public.protein_classification '
            '(protein_class_id int, parent_id int, pref_name text, class_level int);'
            "INSERT INTO public.target_dictionary VALUES (20, 'CHEMBL3708', 'p53', 'SINGLE PROTEIN');"
            'INSERT INTO public.target_components VALUES (2001, 20, 300);'
            "INSERT INTO public.component_sequences VALUES (300, 'P04637');"
            'INSERT INTO public.component_class VALUES (5001, 300, 901);'
            "INSERT INTO public.protein_classification VALUES (900, NULL, 'Protein class', 0), "
            "(901, 900, 'Transcription factor', 1);"
        )
        dump = path / 'chembl.dmp'
        subprocess.run([str(server.bin_path / 'pg_dump'), '-Fc', '-f', str(dump), server.get_uri()], check=True)
        yield dump
    finally:
        server.cleanup()


@pytest.fixture(scope='module')
def output(chembl_dump: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp('target')
    source = {
        'ensembl': write_core_dump(root / 'ensembl'),
        'chembl': str(chembl_dump),
        **write_inputs(root / 'inputs'),
    }
    destination = root / 'output' / 'target'
    target(source, str(destination), {}, Config(step='target', steps=['target'], work_path=root))
    result = scan_dataset(str(destination)).collect()
    return {'frame': result, 'rows': {r['id']: r for r in result.to_dicts()}}


class TestTarget:
    def test_schema_is_the_published_one(self, output: dict) -> None:
        assert output['frame'].schema == TARGET

    def test_genes_on_the_primary_assembly_only(self, output: dict) -> None:
        assert output['frame']['id'].to_list() == ['ENSG00000000002', 'ENSG00000141510']

    def test_tp53(self, output: dict) -> None:
        tp53 = output['rows']['ENSG00000141510']
        assert tp53['approvedSymbol'] == 'TP53'
        assert tp53['approvedName'] == 'tumor protein p53'
        assert tp53['genomicLocation'] == {'chromosome': '17', 'start': 7661779, 'end': 7687546, 'strand': -1}
        assert tp53['tss'] == 7687538
        assert tp53['proteinIds'] == [
            {'id': 'P04637', 'source': 'uniprot_swissprot'},
            {'id': 'K7PPA8', 'source': 'uniprot_trembl'},
            {'id': 'ENSP00000269305', 'source': 'ensembl_PRO'},
            {'id': 'Q15086', 'source': 'uniprot_obsolete'},
        ]
        assert sorted((x['id'], x['source']) for x in tp53['dbXrefs']) == [
            ('11998', 'HGNC'),
            ('1A1U', 'PDB'),
            ('SignalP-noTM', 'signalP'),
        ]
        assert tp53['targetClass'] == [{'id': 901, 'label': 'Transcription factor', 'level': 'l1'}]
        assert [g['id'] for g in tp53['go']] == ['GO:0003677']
        assert [p['pathwayId'] for p in tp53['pathways']] == ['R-HSA-69278']
        assert [c['constraintType'] for c in tp53['constraint']] == ['syn', 'mis', 'lof']
        assert tp53['hallmarks']['cancerHallmarks'][0]['label'] == 'angiogenesis'
        assert {s['source'] for s in tp53['synonyms']} == {'HGNC', 'uniprot', 'NCBI_entrez'}
        assert {(loc['location'], loc['source']) for loc in tp53['subcellularLocations']} == {
            ('Nucleus', 'uniprot'),
            ('Nucleoplasm', 'HPA_main'),
        }

    def test_tec_gene_without_hgnc(self, output: dict) -> None:
        tec = output['rows']['ENSG00000000002']
        assert (tec['approvedSymbol'], tec['approvedName'], tec['biotype']) == ('ENSG00000000002', '', '')
        assert tec['proteinIds'] is None
        assert tec['tss'] == 1000
