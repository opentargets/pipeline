"""Tests for the target pyspark module.

Ported from platform-etl-backend target step.
"""

import pytest
from pyspark.sql import Row
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.target import (
    _add_tss,
    _build_gene_code,
    _build_gene_ontology,
    _build_gene_with_location,
    _build_genetic_constraints,
    _build_hallmarks,
    _build_hgnc,
    _build_protein_classification,
    _build_reactome,
    _filter_ensembl,
    _flatten_protein_classification,
    _map_uniprot_locations_to_ssl,
    _merge_hgnc_ensembl,
)

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

INCLUDE_CHROMS = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']


def _spark_df(spark, data, schema):
    return spark.createDataFrame(data, schema)


# ---------------------------------------------------------------------------
# 1. Ensembl gene filtering (canonical chromosomes, protein-coding filter)
# ---------------------------------------------------------------------------

ENSEMBL_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('chromosome', StringType()),
    StructField('biotype', StringType()),
    StructField('start', LongType()),
    StructField('end', LongType()),
    StructField('strand', IntegerType()),
    StructField('description', StringType()),
    StructField('approvedSymbol', StringType()),
    StructField(
        'transcripts',
        ArrayType(
            StructType([
                StructField('id', StringType()),
                StructField('biotype', StringType()),
            ])
        ),
    ),
    StructField('uniprot_swissprot', ArrayType(StringType())),
    StructField('uniprot_trembl', ArrayType(StringType())),
    StructField('translations', ArrayType(StructType([StructField('id', StringType())]))),
    StructField('signalP', ArrayType(StringType())),
])


def _ensembl_row(id, chromosome, biotype='protein_coding', swissprot=None):
    return Row(
        id=id,
        chromosome=chromosome,
        biotype=biotype,
        start=1000,
        end=2000,
        strand=1,
        description='test description [Source:test]',
        approvedSymbol=id,
        transcripts=[],
        uniprot_swissprot=swissprot,
        uniprot_trembl=None,
        translations=[],
        signalP=None,
    )


def test_ensembl_keeps_canonical_chromosomes(spark):
    """Genes on canonical chromosomes 1-22, X, Y, MT are retained."""
    data = [
        _ensembl_row('ENSG00000001', '1'),
        _ensembl_row('ENSG00000002', 'X'),
        _ensembl_row('ENSG00000003', 'MT'),
        _ensembl_row('ENSG00000004', 'CHR_HSCHR6_MHC_APD_CTG1'),  # non-canonical, no swissprot
    ]
    df = spark.createDataFrame(data, ENSEMBL_SCHEMA)
    result = _filter_ensembl(df)
    ids = {row.id for row in result.collect()}
    assert 'ENSG00000001' in ids
    assert 'ENSG00000002' in ids
    assert 'ENSG00000003' in ids
    assert 'ENSG00000004' not in ids


def test_ensembl_keeps_reviewed_non_canonical(spark):
    """Genes with uniprot_swissprot on non-canonical chromosomes are kept."""
    data = [
        _ensembl_row('ENSG00000005', 'CHR_HSCHR6_MHC_APD_CTG1', swissprot=['P00533']),
    ]
    df = spark.createDataFrame(data, ENSEMBL_SCHEMA)
    result = _filter_ensembl(df)
    ids = {row.id for row in result.collect()}
    assert 'ENSG00000005' in ids


def test_ensembl_filters_non_ensg(spark):
    """Non-ENSG IDs are excluded."""
    data = [
        _ensembl_row('ENSG00000006', '1'),
        _ensembl_row('LRG_71', '1'),
    ]
    df = spark.createDataFrame(data, ENSEMBL_SCHEMA)
    result = _filter_ensembl(df)
    ids = {row.id for row in result.collect()}
    assert 'ENSG00000006' in ids
    assert 'LRG_71' not in ids


# ---------------------------------------------------------------------------
# 2. HGNC symbol mapping
# ---------------------------------------------------------------------------

HGNC_SCHEMA = StructType([
    StructField(
        'response',
        StructType([
            StructField(
                'docs',
                ArrayType(
                    StructType([
                        StructField('ensembl_gene_id', StringType()),
                        StructField('hgnc_id', StringType()),
                        StructField('symbol', StringType()),
                        StructField('name', StringType()),
                        StructField('uniprot_ids', ArrayType(StringType())),
                        StructField('alias_symbol', ArrayType(StringType())),
                        StructField('alias_name', ArrayType(StringType())),
                        StructField('prev_symbol', ArrayType(StringType())),
                        StructField('prev_name', ArrayType(StringType())),
                    ])
                ),
            )
        ]),
    )
])


def test_hgnc_symbol_mapping(spark):
    """HGNC maps ensembl_gene_id to approvedSymbol and approvedName."""
    data = [
        {
            'response': {
                'docs': [
                    {
                        'ensembl_gene_id': 'ENSG00000141510',
                        'hgnc_id': 'HGNC:11998',
                        'symbol': 'TP53',
                        'name': 'tumor protein p53',
                        'uniprot_ids': ['P04637'],
                        'alias_symbol': None,
                        'alias_name': None,
                        'prev_symbol': None,
                        'prev_name': None,
                    }
                ]
            }
        }
    ]
    df = spark.createDataFrame(data, HGNC_SCHEMA)
    result = _build_hgnc(df)
    rows = result.collect()
    assert len(rows) == 1
    row = rows[0]
    assert row.ensemblId == 'ENSG00000141510'
    assert row.approvedSymbol == 'TP53'
    assert row.approvedName == 'tumor protein p53'


# ---------------------------------------------------------------------------
# 3. GO annotation grouping by aspect (BP, MF, CC)
# ---------------------------------------------------------------------------

GO_HUMAN_SCHEMA = StructType([
    StructField('_c0', StringType()),  # database
    StructField('_c1', StringType()),  # dbObjectId
    StructField('_c2', StringType()),  # dbObjectSymbol
    StructField('_c3', StringType()),  # qualifier
    StructField('_c4', StringType()),  # goId
    StructField('_c5', StringType()),  # dbReference
    StructField('_c6', StringType()),  # evidenceCode
    StructField('_c7', StringType()),  # withOrFrom
    StructField('_c8', StringType()),  # aspect
    StructField('_c9', StringType()),  # dbObjectName
    StructField('_c10', StringType()),  # dbObjectSynonym
    StructField('_c11', StringType()),  # dbObjectType
    StructField('_c12', StringType()),  # taxon
    StructField('_c13', StringType()),  # date
    StructField('_c14', StringType()),  # assignedBy
    StructField('_c15', StringType()),  # annotationExtension
    StructField('_c16', StringType()),  # geneProductFormId
])


def _go_row(db_obj_id, go_id, evidence, aspect, db_ref='PMID:1234'):
    return Row(
        _c0='UniProtKB',
        _c1=db_obj_id,
        _c2='SYMBOL',
        _c3='enables',
        _c4=go_id,
        _c5=db_ref,
        _c6=evidence,
        _c7='',
        _c8=aspect,
        _c9='Protein name',
        _c10='',
        _c11='protein',
        _c12='taxon:9606',
        _c13='20230101',
        _c14='UniProt',
        _c15='',
        _c16='',
    )


def test_go_grouping_by_aspect(spark):
    """GO annotations are grouped per Ensembl ID with aspect preserved."""
    human_data = [
        _go_row('P04637', 'GO:0003677', 'IDA', 'F'),  # MF
        _go_row('P04637', 'GO:0008150', 'TAS', 'P'),  # BP
        _go_row('P04637', 'GO:0005634', 'IDA', 'C'),  # CC
    ]
    human_df = spark.createDataFrame(human_data, GO_HUMAN_SCHEMA)
    rna_df = spark.createDataFrame([], GO_HUMAN_SCHEMA)

    rna_lookup_schema = StructType([
        StructField('_c0', StringType()),
        StructField('_c1', StringType()),
        StructField('_c2', StringType()),
        StructField('_c3', StringType()),
        StructField('_c4', StringType()),
        StructField('_c5', StringType()),
    ])
    rna_lookup_df = spark.createDataFrame([], rna_lookup_schema)

    eco_schema = StructType([
        StructField('_c1', StringType()),
        StructField('_c3', StringType()),
        StructField('_c5', StringType()),
        StructField('_c11', StringType()),
    ])
    eco_df = spark.createDataFrame([], eco_schema)

    # Build a minimal ensembl-like lookup for GO
    ensembl_go_schema = StructType([
        StructField('id', StringType()),
        StructField('approvedSymbol', StringType()),
        StructField(
            'proteinIds',
            ArrayType(
                StructType([
                    StructField('id', StringType()),
                    StructField('source', StringType()),
                ])
            ),
        ),
    ])
    ensembl_go_data = [
        Row(
            id='ENSG00000141510',
            approvedSymbol='TP53',
            proteinIds=[Row(id='P04637', source='uniprot_swissprot')],
        )
    ]
    ensembl_df = spark.createDataFrame(ensembl_go_data, ensembl_go_schema)

    result = _build_gene_ontology(human_df, rna_df, rna_lookup_df, eco_df, ensembl_df)
    rows = result.collect()
    assert len(rows) == 1
    row = rows[0]
    assert row.ensemblId == 'ENSG00000141510'
    aspects = {g.aspect for g in row.go}
    assert 'F' in aspects
    assert 'P' in aspects
    assert 'C' in aspects


# ---------------------------------------------------------------------------
# 5. Genetic constraints
# ---------------------------------------------------------------------------


def test_genetic_constraints_structure(spark):
    """Genetic constraints are grouped as an array with syn/mis/lof entries."""
    constraint_schema = StructType([
        StructField('gene_id', StringType()),
        StructField('canonical', StringType()),
        StructField('transcript_type', StringType()),
        StructField('syn.z_score', StringType()),
        StructField('syn.exp', StringType()),
        StructField('syn.obs', StringType()),
        StructField('syn.oe', StringType()),
        StructField('syn.oe_ci.lower', StringType()),
        StructField('syn.oe_ci.upper', StringType()),
        StructField('mis.z_score', StringType()),
        StructField('mis.exp', StringType()),
        StructField('mis.obs', StringType()),
        StructField('mis.oe', StringType()),
        StructField('mis.oe_ci.lower', StringType()),
        StructField('mis.oe_ci.upper', StringType()),
        StructField('lof.pLI', StringType()),
        StructField('lof.exp', StringType()),
        StructField('lof.obs', StringType()),
        StructField('lof.oe', StringType()),
        StructField('lof.oe_ci.lower', StringType()),
        StructField('lof.oe_ci.upper', StringType()),
        StructField('lof.oe_ci.upper_rank', StringType()),
        StructField('lof.oe_ci.upper_bin_decile', StringType()),
    ])
    constraint_data = [
        {
            'gene_id': 'ENSG00000141510',
            'canonical': 'true',
            'transcript_type': 'protein_coding',
            'syn.z_score': '1.23',
            'syn.exp': '100.5',
            'syn.obs': '95',
            'syn.oe': '0.95',
            'syn.oe_ci.lower': '0.8',
            'syn.oe_ci.upper': '1.1',
            'mis.z_score': '2.5',
            'mis.exp': '200.0',
            'mis.obs': '180',
            'mis.oe': '0.9',
            'mis.oe_ci.lower': '0.75',
            'mis.oe_ci.upper': '1.05',
            'lof.pLI': '0.99',
            'lof.exp': '50.0',
            'lof.obs': '5',
            'lof.oe': '0.1',
            'lof.oe_ci.lower': '0.05',
            'lof.oe_ci.upper': '0.2',
            'lof.oe_ci.upper_rank': '1000',
            'lof.oe_ci.upper_bin_decile': '1',
        }
    ]
    df = spark.createDataFrame(constraint_data, constraint_schema)
    result = _build_genetic_constraints(df)
    rows = result.collect()
    assert len(rows) == 1
    row = rows[0]
    assert row.id == 'ENSG00000141510'
    constraint_types = {c.constraintType for c in row.constraint}
    assert constraint_types == {'syn', 'mis', 'lof'}


CONSTRAINT_METRICS = [
    f'{kind}.{field}'
    for kind in ('syn', 'mis')
    for field in ('z_score', 'exp', 'obs', 'oe', 'oe_ci.lower', 'oe_ci.upper')
] + [
    'lof.pLI',
    'lof.exp',
    'lof.obs',
    'lof.oe',
    'lof.oe_ci.lower',
    'lof.oe_ci.upper',
    'lof.oe_ci.upper_bin_decile',
]

CONSTRAINT_BIN_SCHEMA = StructType(
    [
        StructField('gene_id', StringType()),
        StructField('canonical', StringType()),
        StructField('transcript_type', StringType()),
        StructField('lof.oe_ci.upper_rank', StringType()),
    ]
    + [StructField(m, StringType()) for m in CONSTRAINT_METRICS]
)


def _constraint_row(gene_id: str, rank: str) -> dict:
    return {
        'gene_id': gene_id,
        'canonical': 'true',
        'transcript_type': 'protein_coding',
        'lof.oe_ci.upper_rank': rank,
        **dict.fromkeys(CONSTRAINT_METRICS, '1'),
    }


def _lof_bins(spark, ranked_count: int, unranked_count: int):
    """Build a frame of ranked + unranked genes and return their lof bins by id."""
    ranked = [_constraint_row(f'ENSG{i:011d}', str(i)) for i in range(1, ranked_count + 1)]
    unranked = [_constraint_row(f'ENSGNA{i:09d}', 'NA') for i in range(unranked_count)]

    result = _build_genetic_constraints(spark.createDataFrame(ranked + unranked, CONSTRAINT_BIN_SCHEMA))

    bins = {}
    for row in result.collect():
        lof = next(c for c in row.constraint if c.constraintType == 'lof')
        bins[row.id] = lof
    return ranked, unranked, bins


def test_genetic_constraints_sextiles_ignore_unranked_genes(spark):
    """Genes without a LOEUF rank do not consume sextile slots.

    `upperBin6` splits the ranked genes into six equal groups. Genes whose
    `lof.oe_ci.upper_rank` is 'NA' have no place in that ordering — if they sit
    inside the window they sort first, take slots from the lowest bins, and leave
    bin 0 short of the genes that belong there.
    """
    ranked, unranked, bins = _lof_bins(spark, ranked_count=12, unranked_count=6)

    ranked_bins = sorted(bins[g['gene_id']].upperBin6 for g in ranked)
    assert ranked_bins == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

    assert all(bins[g['gene_id']].upperBin6 is None for g in unranked)


def test_genetic_constraints_deciles_are_computed_locally(spark):
    """`upperBin` splits the same ranked genes into ten equal groups.

    gnomAD's own `lof.oe_ci.upper_bin_decile` is binned against a wider gene set
    than the one it ships — in 4.1.1 it never reaches 9 — so the decile is
    computed here, over the genes we actually rank, exactly like the sextile.
    """
    ranked, unranked, bins = _lof_bins(spark, ranked_count=20, unranked_count=6)

    ranked_bins = sorted(bins[g['gene_id']].upperBin for g in ranked)
    assert ranked_bins == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]

    assert all(bins[g['gene_id']].upperBin is None for g in unranked)


# ---------------------------------------------------------------------------
# 6. Hallmarks
# ---------------------------------------------------------------------------


def test_hallmarks_split_cancer_vs_non_cancer(spark):
    """Hallmarks splits records into cancerHallmarks and attributes."""
    hallmark_schema = StructType([
        StructField('GENE_SYMBOL', StringType()),
        StructField('PUBMED_PMID', StringType()),
        StructField('HALLMARK', StringType()),
        StructField('IMPACT', StringType()),
        StructField('DESCRIPTION', StringType()),
    ])
    hallmark_data = [
        Row(
            GENE_SYMBOL='TP53',
            PUBMED_PMID='12345',
            HALLMARK='angiogenesis',
            IMPACT='promotes',
            DESCRIPTION='promotes tumour angiogenesis',
        ),
        Row(
            GENE_SYMBOL='TP53',
            PUBMED_PMID='67890',
            HALLMARK='apoptosis',
            IMPACT='suppresses',
            DESCRIPTION='induces apoptosis',
        ),
    ]
    df = spark.createDataFrame(hallmark_data, hallmark_schema)
    result = _build_hallmarks(df)
    rows = result.collect()
    assert len(rows) == 1
    row = rows[0]
    assert row.approvedSymbol == 'TP53'
    # angiogenesis is a cancer hallmark
    cancer_labels = {h.label for h in (row.hallmarks.cancerHallmarks or [])}
    assert 'angiogenesis' in cancer_labels
    # apoptosis is NOT a cancer hallmark → attributes
    attr_names = {a.name for a in (row.hallmarks.attributes or [])}
    assert 'apoptosis' in attr_names


# ---------------------------------------------------------------------------
# 7. Reactome pathways
# ---------------------------------------------------------------------------


def test_reactome_pathways(spark):
    """Reactome groups pathways per Ensembl ID with topLevelTerm."""
    reactome_pathways_schema = StructType([
        StructField('_c0', StringType()),  # ensemblId
        StructField('_c1', StringType()),  # reactomeId
        StructField('_c2', StringType()),  # url
        StructField('_c3', StringType()),  # eventName
        StructField('_c4', StringType()),  # eventCode
        StructField('_c5', StringType()),  # species
    ])
    reactome_pathways_data = [
        Row(
            _c0='ENSG00000141510',
            _c1='R-HSA-69278',
            _c2='https://reactome.org',
            _c3='Cell Cycle',
            _c4='CC',
            _c5='Homo sapiens',
        ),
    ]
    pathways_df = spark.createDataFrame(reactome_pathways_data, reactome_pathways_schema)

    reactome_etl_schema = StructType([
        StructField('id', StringType()),
        StructField('label', StringType()),
        StructField('path', ArrayType(ArrayType(StringType()))),
    ])
    reactome_etl_data = [
        Row(id='R-HSA-69278', label='Cell Cycle', path=[['R-HSA-1', 'R-HSA-69278']]),
        Row(id='R-HSA-1', label='Cell Cycle Root', path=[[None]]),
    ]
    etl_df = spark.createDataFrame(reactome_etl_data, reactome_etl_schema)

    result = _build_reactome(pathways_df, etl_df)
    rows = result.collect()
    assert len(rows) == 1
    row = rows[0]
    assert row.id == 'ENSG00000141510'
    pathway_ids = {p.pathwayId for p in row.pathways}
    assert 'R-HSA-69278' in pathway_ids


# ---------------------------------------------------------------------------
# 8. HGNC + Ensembl merge
# ---------------------------------------------------------------------------


def test_merge_hgnc_ensembl_prefers_hgnc_name(spark):
    """Merged dataframe uses HGNC approvedName/Symbol when available."""
    ensembl_schema = StructType([
        StructField('id', StringType()),
        StructField('biotype', StringType()),
        StructField('approvedName', StringType()),
        StructField('approvedSymbol', StringType()),
        StructField(
            'genomicLocation',
            StructType([
                StructField('chromosome', StringType()),
                StructField('start', LongType()),
                StructField('end', LongType()),
                StructField('strand', IntegerType()),
            ]),
        ),
    ])
    ensembl_data = [
        Row(
            id='ENSG00000141510',
            biotype='protein_coding',
            approvedName='Ensembl approved name',
            approvedSymbol='TP53_ensembl',
            genomicLocation=Row(chromosome='17', start=7661779, end=7687538, strand=-1),
        ),
    ]
    ensembl_df = spark.createDataFrame(ensembl_data, ensembl_schema)

    hgnc_schema = StructType([
        StructField('ensemblId', StringType()),
        StructField('approvedSymbol', StringType()),
        StructField('approvedName', StringType()),
        StructField(
            'hgncId',
            ArrayType(
                StructType([
                    StructField('id', StringType()),
                    StructField('source', StringType()),
                ])
            ),
        ),
        StructField(
            'hgncSynonyms',
            ArrayType(
                StructType([
                    StructField('label', StringType()),
                    StructField('source', StringType()),
                ])
            ),
        ),
        StructField(
            'hgncSymbolSynonyms',
            ArrayType(
                StructType([
                    StructField('label', StringType()),
                    StructField('source', StringType()),
                ])
            ),
        ),
        StructField(
            'hgncNameSynonyms',
            ArrayType(
                StructType([
                    StructField('label', StringType()),
                    StructField('source', StringType()),
                ])
            ),
        ),
        StructField(
            'hgncObsoleteSymbols',
            ArrayType(
                StructType([
                    StructField('label', StringType()),
                    StructField('source', StringType()),
                ])
            ),
        ),
        StructField(
            'hgncObsoleteNames',
            ArrayType(
                StructType([
                    StructField('label', StringType()),
                    StructField('source', StringType()),
                ])
            ),
        ),
        StructField('uniprotIds', ArrayType(StringType())),
    ])
    hgnc_data = [
        Row(
            ensemblId='ENSG00000141510',
            approvedSymbol='TP53',
            approvedName='tumor protein p53',
            hgncId=[Row(id='11998', source='HGNC')],
            hgncSynonyms=None,
            hgncSymbolSynonyms=None,
            hgncNameSynonyms=None,
            hgncObsoleteSymbols=None,
            hgncObsoleteNames=None,
            uniprotIds=['P04637'],
        ),
    ]
    hgnc_df = spark.createDataFrame(hgnc_data, hgnc_schema)

    result = _merge_hgnc_ensembl(hgnc_df, ensembl_df)
    rows = result.collect()
    assert len(rows) == 1
    row = rows[0]
    # HGNC values take precedence
    assert row.approvedSymbol == 'TP53'
    assert row.approvedName == 'tumor protein p53'


# ---------------------------------------------------------------------------
# 9. Output schema validation
# ---------------------------------------------------------------------------

REQUIRED_OUTPUT_COLUMNS = {
    'id',
    'approvedSymbol',
    'approvedName',
    'biotype',
    'genomicLocation',
    'pathways',
    'go',
    'constraint',
    'subcellularLocations',
    'targetClass',
    'hallmarks',
}


def test_output_schema_has_required_columns(spark):
    """The target module exposes the required_output_columns constant."""
    from pts.pyspark.target import REQUIRED_OUTPUT_COLUMNS as ROC

    assert REQUIRED_OUTPUT_COLUMNS.issubset(ROC)


# ---------------------------------------------------------------------------
# 10. Subcellular location struct schema alignment
# ---------------------------------------------------------------------------


def test_subcellular_location_struct_schema_alignment(spark):
    """HPA and UniProt subcellular location paths must produce identical struct schemas.

    Both arrays are merged via array_union, which requires matching element types.
    This test catches a field added to one path without the other, without
    enumerating the fields explicitly.
    """
    hpa_schema = StructType([
        StructField('Gene', StringType()),
        StructField('Main location', StringType()),
        StructField('Additional location', StringType()),
        StructField('Extracellular location', StringType()),
    ])
    hpa_df = spark.createDataFrame(
        [('ENSG00000001', 'Cytoplasm', None, None)],
        hpa_schema,
    )

    hpa_sl_schema = StructType([
        StructField('HPA_location', StringType()),
        StructField('termSL', StringType()),
        StructField('labelSL', StringType()),
    ])
    hpa_sl_df = spark.createDataFrame(
        [('Cytoplasm', 'SL-0086', 'Intracellular')],
        hpa_sl_schema,
    )

    uniprot_schema = StructType([
        StructField('uniprotId', StringType()),
        StructField(
            'locations',
            ArrayType(
                StructType([
                    StructField('location', StringType()),
                    StructField('targetModifier', StringType()),
                ])
            ),
        ),
    ])
    uniprot_df = spark.createDataFrame(
        [('P12345', [('Cytoplasm', None), ('Nucleus', 'Isoform 2')])],
        uniprot_schema,
    )

    ssl_schema = StructType([
        StructField('Subcellular location ID', StringType()),
        StructField('Name', StringType()),
        StructField('Category', StringType()),
    ])
    ssl_df = spark.createDataFrame(
        [
            ('SL-0086', 'Cytoplasm', 'Intracellular'),
            ('SL-0191', 'Nucleus', 'Intracellular'),
        ],
        ssl_schema,
    )

    hpa_result = _build_gene_with_location(hpa_df, hpa_sl_df)
    uniprot_result = _map_uniprot_locations_to_ssl(uniprot_df, ssl_df)

    hpa_locations = hpa_result.schema['locations'].dataType
    assert isinstance(hpa_locations, ArrayType)
    hpa_struct = hpa_locations.elementType

    uniprot_locations = uniprot_result.schema['subcellularLocations'].dataType
    assert isinstance(uniprot_locations, ArrayType)
    uniprot_struct = uniprot_locations.elementType

    assert isinstance(hpa_struct, StructType)
    assert isinstance(uniprot_struct, StructType)
    hpa_fields = {field.name: field.dataType for field in hpa_struct.fields}
    uniprot_fields = {field.name: field.dataType for field in uniprot_struct.fields}

    common_keys = hpa_fields.keys() & uniprot_fields.keys()
    type_mismatches = {k: (hpa_fields[k], uniprot_fields[k]) for k in common_keys if hpa_fields[k] != uniprot_fields[k]}
    assert hpa_fields == uniprot_fields, (
        f'Schema mismatch between HPA and UniProt subcellularLocations structs.\n'
        f'HPA-only fields:     {set(hpa_fields) - set(uniprot_fields)}\n'
        f'UniProt-only fields: {set(uniprot_fields) - set(hpa_fields)}\n'
        f'Type mismatches:     {type_mismatches}'
    )


# ---------------------------------------------------------------------------
# 12. Protein classification (raw ChEMBL tables)
# ---------------------------------------------------------------------------


class TestProteinClassification:
    @pytest.fixture
    def chembl(self, spark):
        # a full six-level chain, plus a second level-1 class on the same
        # component: multi-class components are a small minority of the table, so
        # the case is easy to miss by sampling and is pinned explicitly here
        classes = spark.createDataFrame(
            [
                (1, None, 'Enzyme', 1),
                (2, 1, 'Kinase', 2),
                (3, 2, 'Protein Kinase', 3),
                (4, 3, 'TK', 4),
                (5, 4, 'TK group', 5),
                (6, 5, 'TK family', 6),
                (20, None, 'Transporter', 1),
                (21, None, 'Ion channel', 1),
            ],
            'protein_class_id int, parent_id int, pref_name string, class_level int',
        )
        # Component 1 carries three classes. They are listed so that the lowest
        # protein_class_id (6) is NOT the first row by comp_class_id (20), so a
        # test pinning "the lowest id survives" cannot also pass for an
        # implementation that just takes whichever row comes first.
        #
        # Components 2 and 3 carry classes but belong only to the two-component
        # target 102, so nothing they classify may reach the output.
        # Component 4 is classified but has no accession -- the shape that must
        # not produce an accession=NULL record.
        component_class = spark.createDataFrame(
            [(1, 1, 20), (2, 1, 6), (3, 2, 20), (4, 3, 6), (5, 1, 21), (6, 4, 20)],
            'comp_class_id int, component_id int, protein_class_id int',
        )
        sequences = spark.createDataFrame(
            [(1, 'P00001'), (2, 'P00002'), (3, 'P00003'), (4, None)],
            'component_id int, accession string',
        )
        # component 1 sits under two single-component targets, so its accession
        # must be deduplicated; target 102 has two components and is skipped
        components = spark.createDataFrame(
            [(1, 100, 1), (2, 101, 1), (3, 102, 2), (4, 102, 3), (5, 103, 4)],
            'targcomp_id int, tid int, component_id int',
        )
        targets = spark.createDataFrame(
            [
                (100, 'CHEMBL_T1', 'A', 'SINGLE PROTEIN'),
                (101, 'CHEMBL_T2', 'B', 'SINGLE PROTEIN'),
                (102, 'CHEMBL_T3', 'C', 'PROTEIN COMPLEX'),
                (103, 'CHEMBL_T4', 'D', 'SINGLE PROTEIN'),
            ],
            'tid int, chembl_id string, pref_name string, target_type string',
        )
        return {
            'target_dictionary': targets,
            'target_components': components,
            'component_sequences': sequences,
            'component_class': component_class,
            'protein_classification': classes,
        }

    @staticmethod
    def _flat(chembl):
        flat = _flatten_protein_classification(chembl['protein_classification'])
        return {r['leaf_id']: r.asDict() for r in flat.collect()}

    @staticmethod
    def _by_accession(chembl):
        return {r['accession']: r['targetClass'] for r in _build_protein_classification(**chembl).collect()}

    def test_full_six_level_chain_is_flattened(self, chembl):
        deep = self._flat(chembl)[6]
        assert deep['l1'] == 'Enzyme'
        assert deep['l2'] == 'Kinase'
        assert deep['l3'] == 'Protein Kinase'
        assert deep['l4'] == 'TK'
        assert deep['l5'] == 'TK group'
        assert deep['l6'] == 'TK family'

    def test_labels_land_at_their_own_level_not_their_depth(self, chembl):
        # class 20 is level 1 with no parent: only l1 is filled
        flat = self._flat(chembl)
        assert flat[20]['l1'] == 'Transporter'
        assert flat[20]['l2'] is None
        assert flat[20]['l6'] is None

    def test_only_the_positionally_zipped_class_survives(self, chembl):
        # Component 1 carries three classes -- 20, 6 and 21 -- and contributes
        # exactly one, the lowest protein_class_id. That is what the published
        # data holds; see the comment on zipped_class_per_component for why.
        ids = {c['id'] for c in self._by_accession(chembl)['P00001']}
        # 6, not 20 (the first component_class row) and not 21
        assert ids == {6}

    def test_every_ancestor_becomes_its_own_class_entry(self, chembl):
        by_accession = self._by_accession(chembl)
        chain = {c['level']: c['label'] for c in by_accession['P00001'] if c['id'] == 6}
        assert chain == {
            'l1': 'Enzyme',
            'l2': 'Kinase',
            'l3': 'Protein Kinase',
            'l4': 'TK',
            'l5': 'TK group',
            'l6': 'TK family',
        }

    def test_accession_under_two_targets_is_not_duplicated(self, chembl):
        rows = _build_protein_classification(**chembl).collect()
        accessions = [r['accession'] for r in rows]
        assert len(accessions) == len(set(accessions))
        # P00001 is reached through tid 100 and tid 101, both single-component;
        # the surviving class is the same either way, so the six levels of
        # class 6 stay six entries rather than twelve
        classes = {r['accession']: r['targetClass'] for r in rows}['P00001']
        assert len(classes) == 6
        assert len({(c['id'], c['label'], c['level']) for c in classes}) == 6

    def test_multi_component_target_contributes_no_accessions(self, chembl):
        # Target 102 has two components, 2 and 3, and both carry classes. The
        # single-component restriction means neither accession may appear -- see
        # the comment on the filter for why it is kept.
        assert set(self._by_accession(chembl)) == {'P00001'}

    def test_no_junk_null_accession_row_is_emitted(self, chembl):
        # Component 4 is classified but has no accession. No such record appears
        # in the published dataset, so emitting one here would be inventing a row
        # rather than reproducing one.
        assert None not in set(self._by_accession(chembl))

    def test_class_level_above_max_triggers_a_warning(self, spark):
        # If ChEMBL ever adds a class_level 7 tier, a level-7 leaf still gets
        # l1..l6 from its ancestors -- only its own, most specific label
        # silently vanishes. Nothing else catches this, so it must warn.
        from loguru import logger

        classes = spark.createDataFrame(
            [
                (1, None, 'Enzyme', 1),
                (2, 1, 'Kinase', 7),
            ],
            'protein_class_id int, parent_id int, pref_name string, class_level int',
        )
        messages = []
        sink_id = logger.add(messages.append, level='WARNING')
        try:
            _flatten_protein_classification(classes)
        finally:
            logger.remove(sink_id)

        assert any('7' in message for message in messages)

    def test_class_level_at_max_does_not_warn(self, chembl):
        from loguru import logger

        messages = []
        sink_id = logger.add(messages.append, level='WARNING')
        try:
            _flatten_protein_classification(chembl['protein_classification'])
        finally:
            logger.remove(sink_id)

        assert messages == []


# ---------------------------------------------------------------------------
# _build_gene_code / _add_tss
#
# Both were entirely uncovered until the strand encoding was unified. They are
# coupled: `_build_gene_code` decides how strand is spelled and `_add_tss` reads
# that spelling, with no `otherwise` to fall back on. A disagreement between them
# does not raise -- it empties `tss` for every gene -- so the coupling itself is
# what these tests pin, not just the two functions separately.
# ---------------------------------------------------------------------------

GENE_CODE_GFF3_SCHEMA = StructType([StructField(f'_c{i}', StringType()) for i in range(9)])

_CT_ATTRS = 'gene_id=ENSG00000141510.16;transcript_id=ENST00000269305.9;tag=Ensembl_canonical'


def _gene_code_row(chrom='chr17', start=7661779, end=7687538, strand='-', attrs=_CT_ATTRS):
    return Row(
        _c0=chrom,
        _c1='HAVANA',
        _c2='transcript',
        _c3=str(start),
        _c4=str(end),
        _c5='.',
        _c6=strand,
        _c7='.',
        _c8=attrs,
    )


def test_build_gene_code_strand_is_signed_integer(spark):
    """GENCODE's +/- is translated to Ensembl's 1/-1 on the way in."""
    rows = [
        _gene_code_row(strand='-'),
        _gene_code_row(strand='+', attrs=_CT_ATTRS.replace('ENSG00000141510', 'ENSG00000000001')),
    ]
    result = _build_gene_code(spark.createDataFrame(rows, GENE_CODE_GFF3_SCHEMA))

    ct_type = result.schema['canonicalTranscript'].dataType
    assert isinstance(ct_type, StructType)
    assert ct_type['strand'].dataType == IntegerType()

    strands = {r.gene_id: r.canonicalTranscript.strand for r in result.collect()}
    assert strands == {'ENSG00000141510': -1, 'ENSG00000000001': 1}


def test_build_gene_code_unstranded_feature_is_null(spark):
    """An unstranded GENCODE feature yields null rather than a third strand value."""
    rows = [_gene_code_row(strand='.')]
    row = _build_gene_code(spark.createDataFrame(rows, GENE_CODE_GFF3_SCHEMA)).first()
    assert row is not None
    assert row.canonicalTranscript.strand is None


def _with_canonical_transcript(spark, strand, start=100, end=200):
    """One gene row carrying a canonicalTranscript struct, or null when strand is None."""
    schema = StructType([
        StructField('id', StringType()),
        StructField(
            'canonicalTranscript',
            StructType([
                StructField('id', StringType()),
                StructField('chromosome', StringType()),
                StructField('start', LongType()),
                StructField('end', LongType()),
                StructField('strand', IntegerType()),
            ]),
        ),
    ])
    ct = (
        None
        if strand == 'MISSING'
        else Row(
            id='ENST1',
            chromosome='17',
            start=start,
            end=end,
            strand=strand,
        )
    )
    return spark.createDataFrame([Row(id='ENSG1', canonicalTranscript=ct)], schema)


def test_add_tss_forward_strand_uses_start(spark):
    """A forward-strand gene starts transcribing at `start`."""
    row = _add_tss(_with_canonical_transcript(spark, 1)).first()
    assert row is not None
    assert row.tss == 100


def test_add_tss_reverse_strand_uses_end(spark):
    """A reverse-strand gene starts transcribing at `end`."""
    row = _add_tss(_with_canonical_transcript(spark, -1)).first()
    assert row is not None
    assert row.tss == 200


def test_add_tss_null_strand_yields_null(spark):
    """No `otherwise`: an unstranded gene gets no TSS rather than an invented one."""
    row = _add_tss(_with_canonical_transcript(spark, None)).first()
    assert row is not None
    assert row.tss is None


def test_add_tss_missing_canonical_transcript_yields_null(spark):
    """A gene the GENCODE join missed carries no TSS."""
    row = _add_tss(_with_canonical_transcript(spark, 'MISSING')).first()
    assert row is not None
    assert row.tss is None


def test_add_tss_drops_canonical_transcript(spark):
    """The internal canonicalTranscript struct must not reach the output."""
    result = _add_tss(_with_canonical_transcript(spark, 1))
    assert 'canonicalTranscript' not in result.columns


def test_gene_code_and_tss_agree_on_the_strand_encoding(spark):
    """The two halves must speak the same encoding end to end.

    This is the test that would have caught the failure the old code was one
    edit away from: `_add_tss` comparing against a spelling `_build_gene_code`
    no longer produces yields null for every gene, with no error anywhere.
    """
    rows = [
        _gene_code_row(strand='+', start=7661779, end=7687538),
        _gene_code_row(
            strand='-',
            start=1000,
            end=2000,
            attrs=_CT_ATTRS.replace('ENSG00000141510', 'ENSG00000000001'),
        ),
    ]
    gene_code = _build_gene_code(spark.createDataFrame(rows, GENE_CODE_GFF3_SCHEMA))
    result = _add_tss(gene_code.withColumnRenamed('gene_id', 'id'))

    tss = {r.id: r.tss for r in result.collect()}
    assert tss == {'ENSG00000141510': 7661779, 'ENSG00000000001': 2000}
    assert all(v is not None for v in tss.values())
