"""End-to-end test for the search transformer against a miniature release on disk."""

from pathlib import Path

import polars as pl

from pts.transformers.search import search
from pts.transformers.search.helpers import LIST_STR

LABELLED = pl.List(pl.Struct({'label': pl.String, 'source': pl.String}))
IDENTIFIED = pl.List(pl.Struct({'id': pl.String, 'source': pl.String}))
XREFS = pl.List(pl.Struct({'source': pl.String, 'ids': LIST_STR}))
CONSEQUENCES = pl.List(
    pl.Struct({'targetId': pl.String, 'consequenceScore': pl.Float32, 'distanceFromFootprint': pl.Int64})
)
SYNONYM_STRUCT = pl.Struct(
    {
        'hasExactSynonym': LIST_STR,
        'hasRelatedSynonym': LIST_STR,
        'hasNarrowSynonym': LIST_STR,
        'hasBroadSynonym': LIST_STR,
    }
)

RELEASE_COLUMNS = [
    'id', 'name', 'description', 'entity', 'category',
    'keywords', 'prefixes', 'ngrams', 'terms', 'terms25', 'terms5', 'multiplier',
]


def _write(root: Path, name: str, frame: pl.DataFrame) -> str:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / '00000000.parquet')
    return str(directory)


def _build_release(root: Path) -> dict[str, str]:
    paths = {}
    paths['disease'] = _write(root, 'disease', pl.DataFrame(
        [{'id': 'D1', 'name': 'asthma', 'description': 'd', 'synonyms': None, 'therapeuticAreas': []}],
        schema={'id': pl.String, 'name': pl.String, 'description': pl.String, 'synonyms': SYNONYM_STRUCT,
                'therapeuticAreas': LIST_STR},
    ))
    paths['target'] = _write(root, 'target', pl.DataFrame(
        [{'id': 'T1', 'approvedSymbol': 'EGFR', 'approvedName': 'receptor', 'biotype': 'protein_coding',
          'synonyms': [], 'proteinIds': [], 'dbXrefs': []}],
        schema={'id': pl.String, 'approvedSymbol': pl.String, 'approvedName': pl.String, 'biotype': pl.String,
                'synonyms': LABELLED, 'proteinIds': IDENTIFIED, 'dbXrefs': IDENTIFIED},
    ))
    paths['drug'] = _write(root, 'drug_molecule', pl.DataFrame(
        [{'id': 'CH1', 'name': 'aspirin', 'description': 'x', 'drugType': 'Small molecule',
          'synonyms': [], 'tradeNames': [], 'crossReferences': [], 'childChemblIds': []}],
        schema={'id': pl.String, 'name': pl.String, 'description': pl.String, 'drugType': pl.String,
                'synonyms': LABELLED, 'tradeNames': LABELLED, 'crossReferences': XREFS, 'childChemblIds': LIST_STR},
    ))
    paths['mechanism'] = _write(root, 'drug_mechanism_of_action', pl.DataFrame(
        [{'chemblIds': ['CH1'], 'mechanismOfAction': 'COX', 'references': None, 'targetName': 'n',
          'targets': [], 'actionType': 'INHIBITOR', 'targetType': 'protein'}],
        schema={'chemblIds': LIST_STR, 'mechanismOfAction': pl.String, 'references': LIST_STR, 'targetName': pl.String,
                'targets': LIST_STR, 'actionType': pl.String, 'targetType': pl.String},
    ))
    paths['indication'] = _write(root, 'clinical_indication', pl.DataFrame(
        [{'clinicalReportIds': ['nct01'], 'drugId': 'CH1', 'diseaseId': 'D1'}],
        schema={'clinicalReportIds': LIST_STR, 'drugId': pl.String, 'diseaseId': pl.String},
    ))
    paths['association'] = _write(root, 'association_overall_indirect', pl.DataFrame(
        [{'diseaseId': 'D1', 'targetId': 'T1', 'associationScore': 0.8}]))
    paths['evidence'] = str(root / 'evidence_*')
    _write(root, 'evidence_a', pl.DataFrame([{'drugId': 'CH1', 'targetId': 'T1', 'diseaseId': 'D1'}]))
    _write(root, 'evidence_b', pl.DataFrame([{'targetId': 'T1', 'diseaseId': 'D1'}]))
    paths['disease_hpo'] = _write(root, 'disease_phenotype', pl.DataFrame([{'disease': 'D1', 'phenotype': 'HP1'}]))
    paths['hpo'] = _write(root, 'disease_hpo', pl.DataFrame([{'id': 'HP1', 'name': 'wheeze'}]))
    paths['studies'] = _write(root, 'study', pl.DataFrame(
        [{'studyId': 'S1', 'traitFromSource': 't', 'pubmedId': 'PM1', 'publicationFirstAuthor': 'Smith',
          'diseaseIds': ['D1'], 'nSamples': 10, 'geneId': 'T1'}],
        schema={'studyId': pl.String, 'traitFromSource': pl.String, 'pubmedId': pl.String,
                'publicationFirstAuthor': pl.String, 'diseaseIds': LIST_STR, 'nSamples': pl.Int32, 'geneId': pl.String},
    ))
    paths['variants'] = _write(root, 'variant', pl.DataFrame(
        [{'variantId': '1_100_A_G', 'rsIds': ['rs1'], 'hgvsId': 'h', 'dbXrefs': [], 'chromosome': '1', 'position': 100,
          'transcriptConsequences': [{'targetId': 'T1', 'consequenceScore': 1.0, 'distanceFromFootprint': 5}]}],
        schema={'variantId': pl.String, 'rsIds': LIST_STR, 'hgvsId': pl.String, 'dbXrefs': IDENTIFIED,
                'chromosome': pl.String, 'position': pl.Int32, 'transcriptConsequences': CONSEQUENCES},
    ))
    paths['credible_sets'] = _write(root, 'credible_set', pl.DataFrame([{'studyId': 'S1'}]))
    return paths


def test_search_writes_all_five_views_with_the_release_schema(tmp_path: Path) -> None:
    source = _build_release(tmp_path / 'output')
    destination = {name: str(tmp_path / 'view' / f'search_{name}') for name in
                   ('diseases', 'targets', 'drugs', 'variants', 'studies')}

    search(source, destination, None, None)

    for path in destination.values():
        frame = pl.read_parquet(Path(path) / '**' / '*.parquet')
        assert frame.columns == RELEASE_COLUMNS
        assert frame.schema['multiplier'] == pl.Float64
        assert frame.height == 1


def test_search_emits_one_document_per_source_entity(tmp_path: Path) -> None:
    """The invariant that holds on the real release: 45,896 / 78,733 / 19,170 / 8,173,485 /
    4,074,292 documents for the same number of source rows."""
    source = _build_release(tmp_path / 'output')
    destination = {name: str(tmp_path / 'view' / f'search_{name}') for name in
                   ('diseases', 'targets', 'drugs', 'variants', 'studies')}

    search(source, destination, None, None)

    expected = {'diseases': 'D1', 'targets': 'T1', 'drugs': 'CH1', 'variants': '1_100_A_G', 'studies': 'S1'}
    for name, identifier in expected.items():
        frame = pl.read_parquet(Path(destination[name]) / '**' / '*.parquet')
        assert frame['id'].to_list() == [identifier]
