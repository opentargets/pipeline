"""Tests for the homology transformer."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from pts.pyspark.homology import _build_homology, _resolve_homology
from pts.schemas.homology import homology_schema, orthologs_schema
from pts.transformers.homology import COMPARA_COLUMNS, build_homology, resolve_homology

SPECIES_SCHEMA = {'#name': pl.String, 'species': pl.String, 'taxonomy_id': pl.String}
COMPARA_SCHEMA = dict.fromkeys(COMPARA_COLUMNS, pl.String)
NAMES_SCHEMA = {'id': pl.String, 'name': pl.String}
TARGET_SCHEMA = {'id': pl.String, 'approvedSymbol': pl.String}

#: Compara columns the step does not read.
OTHER_COMPARA_COLUMNS = [
    'protein_stable_id',
    'homology_protein_stable_id',
    'dn',
    'ds',
    'goc_score',
    'wga_coverage',
    'homology_id',
]

MOUSE = ('Mouse', 'mus_musculus', '10090')


def _species(*rows: tuple[str | None, ...]) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=SPECIES_SCHEMA, orient='row')


def _pair(**kwargs: str | None) -> dict[str, str | None]:
    """One homo_sapiens/mouse Compara row, whitelisted by default."""
    row: dict[str, str | None] = {
        'gene_stable_id': 'ENSG0001',
        'species': 'homo_sapiens',
        'identity': '100.0',
        'homology_type': 'ortholog_one2one',
        'homology_gene_stable_id': 'ENSMUSG0001',
        'homology_species': 'mus_musculus',
        'homology_identity': '88.0',
        'is_high_confidence': '1',
    }
    row.update(kwargs)
    return row


def _homologies(*rows: dict[str, str | None]) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=COMPARA_SCHEMA)


def _names(*rows: tuple[str, str | None]) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=NAMES_SCHEMA, orient='row')


def _target(*rows: tuple[str, str | None]) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=TARGET_SCHEMA, orient='row')


def _orthologs(**kwargs: object) -> pl.DataFrame:
    row: dict[str, object] = {
        'id': 'ENSG1',
        'speciesId': '10090',
        'speciesName': 'mouse',
        'homologyType': 'ortholog_one2one',
        'targetGeneId': 'ENSMUSG1',
        'isHighConfidence': '1',
        'targetGeneSymbol': 'Gene1',
        'queryPercentageIdentity': 90.0,
        'targetPercentageIdentity': 88.0,
        'priority': 0,
    }
    row.update(kwargs)
    return pl.DataFrame([row], schema=orthologs_schema)


def _single_pair(pair: dict[str, str | None], gene_name: str | None) -> pl.DataFrame:
    """A whitelisted mouse pair plus a one-row gene name lookup, for isolated checks."""
    return build_homology(
        _species(MOUSE),
        _homologies(pair),
        _names(('ENSMUSG0001', gene_name)),
        ['10090-mus_musculus'],
    )


# ---------------------------------------------------------------------------
# build_homology
# ---------------------------------------------------------------------------


def test_homology_whitelist_filtering() -> None:
    """Only species in the whitelist are included in homologues."""
    species = _species(
        ('mus_musculus', 'mus_musculus', '10090'),
        ('rattus_norvegicus', 'rattus_norvegicus', '10116'),
        ('danio_rerio', 'danio_rerio', '7955'),
    )
    homologies = _homologies(
        _pair(),
        _pair(
            homology_gene_stable_id='ENSRNOG0001',
            homology_species='rattus_norvegicus',
            homology_identity='85.0',
        ),
    )
    result = build_homology(species, homologies, _names(('ENSMUSG0001', 'Trp53')), ['10090-mus_musculus'])
    species_ids = set(result['speciesId'].to_list())
    assert '10090' in species_ids
    assert '10116' not in species_ids


def test_homology_normalises_stringified_null_confidence() -> None:
    """The raw TSV's literal "NULL" text for is_high_confidence becomes a real null."""
    result = _single_pair(_pair(is_high_confidence='NULL'), gene_name='Trp53')
    assert result['isHighConfidence'].to_list() == [None]


def test_homology_keeps_real_confidence_flag() -> None:
    """A genuine "1"/"0" confidence flag passes through unchanged."""
    result = _single_pair(_pair(is_high_confidence='1'), gene_name='Trp53')
    assert result['isHighConfidence'].to_list() == ['1']


@pytest.mark.parametrize('real_symbol', ['na', 'nan', 'Nil'])
def test_homology_keeps_genuine_symbols_that_look_like_null_tokens(real_symbol: str) -> None:
    """The symbols "na", "nan" and "Nil" are real FlyBase genes, not placeholder tokens."""
    result = _single_pair(_pair(), gene_name=real_symbol)
    assert result['targetGeneSymbol'].to_list() == [real_symbol]


def test_homology_keeps_real_gene_symbol() -> None:
    """A genuine gene symbol is kept."""
    result = _single_pair(_pair(), gene_name='Trp53')
    assert result['targetGeneSymbol'].to_list() == ['Trp53']


def test_homology_trims_a_real_gene_symbol() -> None:
    """A real gene symbol with stray leading/trailing spaces is trimmed."""
    result = _single_pair(_pair(), gene_name=' Trp53 ')
    assert result['targetGeneSymbol'].to_list() == ['Trp53']


def test_homology_normalises_whitespace_only_gene_symbol() -> None:
    """A whitespace-only name is treated the same as an empty one."""
    result = _single_pair(_pair(), gene_name='   ')
    assert result['targetGeneSymbol'].to_list() == ['ENSMUSG0001']


def test_homology_falls_back_to_the_id_for_a_null_name() -> None:
    """A gene with no name takes its stable id."""
    result = _single_pair(_pair(), gene_name=None)
    assert result['targetGeneSymbol'].to_list() == ['ENSMUSG0001']


def test_homology_trims_spaces_only() -> None:
    """Spark's trim strips spaces, not tabs."""
    result = _single_pair(_pair(), gene_name='\tTrp53 ')
    assert result['targetGeneSymbol'].to_list() == ['\tTrp53']


def test_build_homology_schema() -> None:
    """The pairs have the orthologs schema, with priority as Int32."""
    result = _single_pair(_pair(), gene_name='Trp53')
    assert result.schema == orthologs_schema


@pytest.mark.parametrize(
    ('text', 'expected'),
    [('41.5335', 41.5335), (' 88.0 ', 88.0), ('NULL', None), ('', None), ('5d', 5.0), ('1e2', 100.0)],
)
def test_build_homology_casts_identity_as_spark(text: str, expected: float | None) -> None:
    """Percentage identities are cast from text as spark's non-ANSI cast does."""
    result = _single_pair(_pair(identity=text, homology_identity=text), gene_name='Trp53')
    assert result['queryPercentageIdentity'].to_list() == [expected]
    assert result['targetPercentageIdentity'].to_list() == [expected]


def test_build_homology_swaps_rows_whose_homologue_is_human() -> None:
    """A row from another species' file pointing at a human gene becomes a human query."""
    row = _pair(
        gene_stable_id='ENSMUSG0001',
        species='mus_musculus',
        identity='70',
        homology_gene_stable_id='ENSG0001',
        homology_species='homo_sapiens',
        homology_identity='75',
    )
    species = _species(('Human', 'homo_sapiens', '9606'), MOUSE)
    names = _names(('ENSMUSG0001', 'Trp53'), ('ENSG0001', 'TP53'))
    result = build_homology(species, _homologies(row), names, ['9606-human', '10090-mouse'])
    assert result.select('id', 'targetGeneId', 'queryPercentageIdentity', 'targetPercentageIdentity').rows() == [
        ('ENSG0001', 'ENSMUSG0001', 75.0, 70.0)
    ]


def test_build_homology_keeps_human_paralogs_both_ways() -> None:
    """A human paralog row yields the pair from both genes."""
    row = _pair(
        homology_type='other_paralog',
        homology_gene_stable_id='ENSG0002',
        homology_species='homo_sapiens',
    )
    species = _species(('Human', 'homo_sapiens', '9606'))
    names = _names(('ENSG0001', 'A'), ('ENSG0002', 'B'))
    result = build_homology(species, _homologies(row), names, ['9606-human'])
    assert sorted(result.select('id', 'targetGeneId', 'targetGeneSymbol').rows()) == [
        ('ENSG0001', 'ENSG0002', 'B'),
        ('ENSG0002', 'ENSG0001', 'A'),
    ]


def test_build_homology_joins_species_by_name_not_taxonomy() -> None:
    """A strain sharing the taxonomy id does not duplicate the pair."""
    species = _species(MOUSE, ('Mouse A/J', 'mus_musculus_aj', '10090'))
    result = build_homology(species, _homologies(_pair()), _names(('ENSMUSG0001', 'Trp53')), ['10090-mouse'])
    assert result['speciesName'].to_list() == ['Mouse']


def test_build_homology_priority_follows_the_whitelist_order() -> None:
    """Priority is the species' position in the whitelist."""
    species = _species(('Rat', 'rattus_norvegicus', '10116'), MOUSE)
    homologies = _homologies(
        _pair(),
        _pair(homology_gene_stable_id='ENSRNOG0001', homology_species='rattus_norvegicus'),
    )
    result = build_homology(species, homologies, _names(), ['10116-rat', '10090-mouse'])
    assert sorted(result.select('speciesId', 'priority').rows()) == [('10090', 1), ('10116', 0)]


# ---------------------------------------------------------------------------
# resolve_homology
# ---------------------------------------------------------------------------


def test_resolve_homology_output_columns() -> None:
    """Output is flat: targetId plus the existing homologue fields, no wrapper array."""
    result = resolve_homology(_orthologs(), _target(('ENSG1', 'GENE1')))
    assert result.schema == homology_schema


def test_resolve_homology_renames_id_to_target_id() -> None:
    """The query gene id becomes targetId."""
    result = resolve_homology(_orthologs(id='ENSG1'), _target(('ENSG1', 'GENE1')))
    assert result['targetId'].to_list() == ['ENSG1']


def test_resolve_homology_drops_unresolvable_rows() -> None:
    """A query id absent from output/target is dropped (validation)."""
    result = resolve_homology(_orthologs(id='UNKNOWN'), _target(('ENSG1', 'GENE1')))
    assert result.height == 0


def test_resolve_homology_falls_back_to_ortholog_symbol() -> None:
    """Without a within-target paralog match, the ortholog's own symbol is kept."""
    orthologs = _orthologs(id='ENSG1', targetGeneId='ENSMUSG1', targetGeneSymbol='Trp53')
    result = resolve_homology(orthologs, _target(('ENSG1', 'GENE1')))
    assert result['targetGeneSymbol'].to_list() == ['Trp53']


def test_resolve_homology_prefers_paralog_symbol() -> None:
    """When targetGeneId matches another target in the universe, its approvedSymbol wins."""
    orthologs = _orthologs(id='ENSG1', targetGeneId='ENSG2', targetGeneSymbol='stale-symbol')
    result = resolve_homology(orthologs, _target(('ENSG1', 'GENE1'), ('ENSG2', 'GENE2')))
    assert result['targetGeneSymbol'].to_list() == ['GENE2']


def test_resolve_homology_falls_back_to_the_query_symbol() -> None:
    """A homologue absent from the gene names takes the query gene's approvedSymbol."""
    orthologs = _orthologs(id='ENSG1', targetGeneId='ENSMUSG1', targetGeneSymbol=None)
    result = resolve_homology(orthologs, _target(('ENSG1', 'GENE1')))
    assert result['targetGeneSymbol'].to_list() == ['GENE1']


def test_resolve_homology_sorts_rows() -> None:
    """Rows are sorted by targetId, then targetGeneId."""
    orthologs = pl.concat([
        _orthologs(id='ENSG2', targetGeneId='ENSMUSG1'),
        _orthologs(id='ENSG1', targetGeneId='ENSMUSG2'),
        _orthologs(id='ENSG1', targetGeneId='ENSMUSG1'),
    ])
    result = resolve_homology(orthologs, _target(('ENSG1', 'GENE1'), ('ENSG2', 'GENE2')))
    assert result.select('targetId', 'targetGeneId').rows() == [
        ('ENSG1', 'ENSMUSG1'),
        ('ENSG1', 'ENSMUSG2'),
        ('ENSG2', 'ENSMUSG1'),
    ]


# ---------------------------------------------------------------------------
# parity with the spark job
# ---------------------------------------------------------------------------

PARITY_SPECIES = [
    ('Human', 'homo_sapiens', '9606'),
    MOUSE,
    ('Mouse A/J', 'mus_musculus_aj', '10090'),
    ('Rat', 'rattus_norvegicus', '10116'),
    ('Gorilla', 'gorilla_gorilla', '9595'),
    ('Zebrafish', 'danio_rerio', '7955'),
    ('Nameless', 'no_taxonomy', None),
]
PARITY_HOMOLOGIES = [
    _pair(),
    _pair(
        gene_stable_id='ENSG0002',
        identity=' 41.5335 ',
        homology_type='other_paralog',
        homology_gene_stable_id='ENSG0003',
        homology_species='homo_sapiens',
        homology_identity='41.8006',
        is_high_confidence='NULL',
    ),
    _pair(
        gene_stable_id='ENSG0003',
        identity='NULL',
        homology_type='within_species_paralog',
        homology_gene_stable_id='ENSG0001',
        homology_species='homo_sapiens',
        homology_identity='',
        is_high_confidence=None,
    ),
    _pair(
        gene_stable_id='ENSMUSG0002',
        species='mus_musculus',
        identity='70',
        homology_type='ortholog_one2many',
        homology_gene_stable_id='ENSG0002',
        homology_species='homo_sapiens',
        homology_identity='75',
        is_high_confidence='0',
    ),
    _pair(
        gene_stable_id='ENSMUSG0003',
        species='mus_musculus',
        homology_gene_stable_id='ENSRNOG0001',
        homology_species='rattus_norvegicus',
    ),
    _pair(homology_gene_stable_id='ENSGGOG0001', homology_species='gorilla_gorilla'),
    _pair(homology_gene_stable_id='ENSMUSG0004', homology_species='mus_musculus_aj'),
    _pair(gene_stable_id='ENSG0099', homology_gene_stable_id='ENSMUSG0005'),
    _pair(
        homology_gene_stable_id='ENSDARG0001',
        homology_species='danio_rerio',
        identity='5d',
        homology_identity='1e2',
    ),
    _pair(homology_type=None, homology_gene_stable_id='ENSMUSG0006'),
    _pair(
        gene_stable_id='ENSMUSG0007', species=None, homology_gene_stable_id='ENSG0001', homology_species='homo_sapiens'
    ),
    _pair(homology_gene_stable_id='ENSXXXG0001', homology_species='no_taxonomy'),
]
PARITY_NAMES = [
    ('ENSMUSG0001', ' Trp53 '),
    ('ENSMUSG0002', '   '),
    ('ENSMUSG0003', 'na'),
    ('ENSMUSG0004', 'Trp53'),
    ('ENSMUSG0005', None),
    ('ENSMUSG0006', ''),
    ('ENSG0001', 'GENE1-ensembl'),
    ('ENSG0002', ''),
    ('ENSG0003', '\tTAB'),
]
PARITY_TARGET = [('ENSG0001', 'GENE1'), ('ENSG0002', 'GENE2'), ('ENSG0003', None)]
PARITY_WHITELIST = ['9606-human', '10090-mouse', '10116-rat', '7955-zebrafish']


def _strings(spark: SparkSession, rows: list[dict[str, str | None]], columns: list[str]):
    schema = StructType([StructField(c, StringType()) for c in columns])
    return spark.createDataFrame([tuple(row.get(c) for c in columns) for row in rows], schema)


def _from_spark(df, schema: pl.Schema) -> pl.DataFrame:
    return pl.DataFrame([row.asDict() for row in df.collect()], schema=schema)


def _spark_inputs(spark: SparkSession):
    species = _strings(
        spark, [dict(zip(SPECIES_SCHEMA, row, strict=True)) for row in PARITY_SPECIES], list(SPECIES_SCHEMA)
    )
    homologies = _strings(spark, PARITY_HOMOLOGIES, COMPARA_COLUMNS + OTHER_COMPARA_COLUMNS)
    names = _strings(spark, [dict(zip(NAMES_SCHEMA, row, strict=True)) for row in PARITY_NAMES], list(NAMES_SCHEMA))
    target = _strings(spark, [dict(zip(TARGET_SCHEMA, row, strict=True)) for row in PARITY_TARGET], list(TARGET_SCHEMA))
    return species, homologies, names, target


def _sorted(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(df.columns, nulls_last=True)


def test_build_homology_matches_spark(spark: SparkSession) -> None:
    """The polars pairs equal the spark job's on the same inputs."""
    species, homologies, names, _ = _spark_inputs(spark)
    expected = _from_spark(_build_homology(species, homologies, names, PARITY_WHITELIST), orthologs_schema)
    result = build_homology(
        _species(*PARITY_SPECIES),
        _homologies(*PARITY_HOMOLOGIES),
        _names(*PARITY_NAMES),
        PARITY_WHITELIST,
    )
    assert_frame_equal(_sorted(result), _sorted(expected))


def test_resolve_homology_matches_spark(spark: SparkSession) -> None:
    """The polars dataset equals the spark job's on the same inputs, row order aside."""
    species, homologies, names, target = _spark_inputs(spark)
    spark_orthologs = _build_homology(species, homologies, names, PARITY_WHITELIST)
    expected = _from_spark(_resolve_homology(spark_orthologs, target), homology_schema)
    orthologs = build_homology(
        _species(*PARITY_SPECIES),
        _homologies(*PARITY_HOMOLOGIES),
        _names(*PARITY_NAMES),
        PARITY_WHITELIST,
    )
    result = resolve_homology(orthologs, _target(*PARITY_TARGET))
    assert result.height > 0
    assert_frame_equal(_sorted(result), _sorted(expected))
