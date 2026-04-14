"""Tests for the target pyspark module.

Ported from platform-etl-backend target step.
"""

import pytest
from pyspark.sql import Row
from pyspark.sql import functions as f
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.target import (
    _build_gene_ontology,
    _build_genetic_constraints,
    _build_hallmarks,
    _build_hgnc,
    _build_homologues,
    _build_reactome,
    _build_safety,
    _filter_ensembl,
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
    StructField('transcripts', ArrayType(StructType([
        StructField('id', StringType()),
        StructField('biotype', StringType()),
    ]))),
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
    StructField('response', StructType([
        StructField('docs', ArrayType(StructType([
            StructField('ensembl_gene_id', StringType()),
            StructField('hgnc_id', StringType()),
            StructField('symbol', StringType()),
            StructField('name', StringType()),
            StructField('uniprot_ids', ArrayType(StringType())),
            StructField('alias_symbol', ArrayType(StringType())),
            StructField('alias_name', ArrayType(StringType())),
            StructField('prev_symbol', ArrayType(StringType())),
            StructField('prev_name', ArrayType(StringType())),
        ])))
    ]))
])


def test_hgnc_symbol_mapping(spark):
    """HGNC maps ensembl_gene_id to approvedSymbol and approvedName."""
    data = [{
        'response': {
            'docs': [{
                'ensembl_gene_id': 'ENSG00000141510',
                'hgnc_id': 'HGNC:11998',
                'symbol': 'TP53',
                'name': 'tumor protein p53',
                'uniprot_ids': ['P04637'],
                'alias_symbol': None,
                'alias_name': None,
                'prev_symbol': None,
                'prev_name': None,
            }]
        }
    }]
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
    StructField('_c10', StringType()), # dbObjectSynonym
    StructField('_c11', StringType()), # dbObjectType
    StructField('_c12', StringType()), # taxon
    StructField('_c13', StringType()), # date
    StructField('_c14', StringType()), # assignedBy
    StructField('_c15', StringType()), # annotationExtension
    StructField('_c16', StringType()), # geneProductFormId
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
        StructField('proteinIds', ArrayType(StructType([
            StructField('id', StringType()),
            StructField('source', StringType()),
        ]))),
    ])
    ensembl_go_data = [Row(id='ENSG00000141510', approvedSymbol='TP53',
                           proteinIds=[Row(id='P04637', source='uniprot_swissprot')])]
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
# 4. Homologue filtering by species whitelist
# ---------------------------------------------------------------------------

def test_homologue_whitelist_filtering(spark):
    """Only species in whitelist are included in homologues."""
    homology_dict_schema = StructType([
        StructField('#name', StringType()),
        StructField('species', StringType()),
        StructField('taxonomy_id', StringType()),
    ])
    homology_dict_data = [
        Row(**{'#name': 'mus_musculus', 'species': 'mus_musculus', 'taxonomy_id': '10090'}),
        Row(**{'#name': 'rattus_norvegicus', 'species': 'rattus_norvegicus', 'taxonomy_id': '10116'}),
        Row(**{'#name': 'danio_rerio', 'species': 'danio_rerio', 'taxonomy_id': '7955'}),
    ]
    homology_dict_df = spark.createDataFrame(homology_dict_data, homology_dict_schema)

    coding_proteins_schema = StructType([
        StructField('gene_stable_id', StringType()),
        StructField('protein_stable_id', StringType()),
        StructField('species', StringType()),
        StructField('identity', DoubleType()),
        StructField('homology_type', StringType()),
        StructField('homology_gene_stable_id', StringType()),
        StructField('homology_protein_stable_id', StringType()),
        StructField('homology_species', StringType()),
        StructField('homology_identity', DoubleType()),
        StructField('dn', DoubleType()),
        StructField('ds', DoubleType()),
        StructField('goc_score', DoubleType()),
        StructField('wga_coverage', DoubleType()),
        StructField('is_high_confidence', StringType()),
        StructField('homology_id', StringType()),
    ])
    coding_proteins_data = [
        Row(gene_stable_id='ENSG0001', protein_stable_id='P001',
            species='homo_sapiens', identity=100.0,
            homology_type='ortholog_one2one',
            homology_gene_stable_id='ENSMUSG0001',
            homology_protein_stable_id='P002',
            homology_species='mus_musculus',
            homology_identity=88.0, dn=None, ds=None,
            goc_score=None, wga_coverage=None,
            is_high_confidence='1', homology_id='h001'),
        # rat — NOT in whitelist
        Row(gene_stable_id='ENSG0001', protein_stable_id='P001',
            species='homo_sapiens', identity=100.0,
            homology_type='ortholog_one2one',
            homology_gene_stable_id='ENSRNOG0001',
            homology_protein_stable_id='P003',
            homology_species='rattus_norvegicus',
            homology_identity=85.0, dn=None, ds=None,
            goc_score=None, wga_coverage=None,
            is_high_confidence='1', homology_id='h002'),
    ]
    coding_proteins_df = spark.createDataFrame(coding_proteins_data, coding_proteins_schema)

    gene_dict_schema = StructType([
        StructField('id', StringType()),
        StructField('name', StringType()),
    ])
    gene_dict_df = spark.createDataFrame(
        [Row(id='ENSMUSG0001', name='Trp53')],
        gene_dict_schema,
    )

    # Only mouse in whitelist (10090), not rat (10116)
    whitelist = ['10090-mus_musculus']
    result = _build_homologues(homology_dict_df, coding_proteins_df, gene_dict_df, whitelist)
    rows = result.collect()
    species_ids = {r.speciesId for r in rows}
    assert '10090' in species_ids
    assert '10116' not in species_ids


# ---------------------------------------------------------------------------
# 5. Safety evidence aggregation
# ---------------------------------------------------------------------------

def test_safety_evidence_aggregation(spark):
    """Safety evidence is grouped by ENSG id into safetyLiabilities."""
    safety_schema = StructType([
        StructField('id', StringType()),
        StructField('targetFromSourceId', StringType()),
        StructField('event', StringType()),
        StructField('eventId', StringType()),
        StructField('effects', ArrayType(StructType([
            StructField('direction', StringType()),
            StructField('dosing', StringType()),
        ]))),
        StructField('biosamples', ArrayType(StructType([
            StructField('tissueLabel', StringType()),
            StructField('tissueId', StringType()),
            StructField('cellLabel', StringType()),
            StructField('cellFormat', StringType()),
        ]))),
        StructField('datasource', StringType()),
        StructField('literature', StringType()),
        StructField('url', StringType()),
        StructField('studies', ArrayType(StructType([
            StructField('name', StringType()),
            StructField('description', StringType()),
            StructField('type', StringType()),
        ]))),
    ])
    safety_data = [
        Row(id='ENSG00000141510', targetFromSourceId='TP53',
            event='heart failure', eventId='EFO_0003777',
            effects=[Row(direction='Activation/Increase/Upregulation', dosing='general')],
            biosamples=None, datasource='TestSource',
            literature='12345678', url=None, studies=None),
        Row(id='ENSG00000141510', targetFromSourceId='TP53',
            event='liver toxicity', eventId='EFO_0001234',
            effects=None, biosamples=None, datasource='TestSource2',
            literature=None, url=None, studies=None),
        Row(id='ENSG00000012048', targetFromSourceId='BRCA1',
            event='breast cancer', eventId='MONDO_0007254',
            effects=None, biosamples=None, datasource='TestSource',
            literature=None, url=None, studies=None),
    ]
    safety_df = spark.createDataFrame(safety_data, safety_schema)

    # lookup df: ensgId -> name (array)
    lookup_schema = StructType([
        StructField('ensgId', StringType()),
        StructField('name', ArrayType(StringType())),
        StructField('uniprot', ArrayType(StringType())),
        StructField('HGNC', ArrayType(StringType())),
        StructField('symbols', ArrayType(StringType())),
    ])
    lookup_data = [
        Row(ensgId='ENSG00000141510', name=['P04637', 'TP53'],
            uniprot=['P04637'], HGNC=['TP53'], symbols=['TP53']),
    ]
    lookup_df = spark.createDataFrame(lookup_data, lookup_schema)

    # disease df for EFO replacement (empty — no obsolete terms to replace)
    disease_schema = StructType([
        StructField('id', StringType()),
        StructField('obsoleteTerms', ArrayType(StringType())),
    ])
    disease_df = spark.createDataFrame([], disease_schema)

    result = _build_safety(safety_df, lookup_df, disease_df)
    rows = {r.id: r for r in result.collect()}

    # Both ENSG ids should appear
    assert 'ENSG00000141510' in rows
    # TP53 has 2 safety liabilities
    assert len(rows['ENSG00000141510'].safetyLiabilities) == 2


# ---------------------------------------------------------------------------
# 6. Genetic constraints
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
    constraint_data = [{
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
    }]
    df = spark.createDataFrame(constraint_data, constraint_schema)
    result = _build_genetic_constraints(df)
    rows = result.collect()
    assert len(rows) == 1
    row = rows[0]
    assert row.id == 'ENSG00000141510'
    constraint_types = {c.constraintType for c in row.constraint}
    assert constraint_types == {'syn', 'mis', 'lof'}


# ---------------------------------------------------------------------------
# 7. Hallmarks
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
        Row(GENE_SYMBOL='TP53', PUBMED_PMID='12345', HALLMARK='angiogenesis',
            IMPACT='promotes', DESCRIPTION='promotes tumour angiogenesis'),
        Row(GENE_SYMBOL='TP53', PUBMED_PMID='67890', HALLMARK='apoptosis',
            IMPACT='suppresses', DESCRIPTION='induces apoptosis'),
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
# 8. Reactome pathways
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
        Row(_c0='ENSG00000141510', _c1='R-HSA-69278', _c2='https://reactome.org',
            _c3='Cell Cycle', _c4='CC', _c5='Homo sapiens'),
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
# 9. HGNC + Ensembl merge
# ---------------------------------------------------------------------------

def test_merge_hgnc_ensembl_prefers_hgnc_name(spark):
    """Merged dataframe uses HGNC approvedName/Symbol when available."""
    ensembl_schema = StructType([
        StructField('id', StringType()),
        StructField('biotype', StringType()),
        StructField('approvedName', StringType()),
        StructField('approvedSymbol', StringType()),
        StructField('genomicLocation', StructType([
            StructField('chromosome', StringType()),
            StructField('start', LongType()),
            StructField('end', LongType()),
            StructField('strand', IntegerType()),
        ])),
    ])
    ensembl_data = [
        Row(id='ENSG00000141510', biotype='protein_coding',
            approvedName='Ensembl approved name',
            approvedSymbol='TP53_ensembl',
            genomicLocation=Row(chromosome='17', start=7661779, end=7687538, strand=-1)),
    ]
    ensembl_df = spark.createDataFrame(ensembl_data, ensembl_schema)

    hgnc_schema = StructType([
        StructField('ensemblId', StringType()),
        StructField('approvedSymbol', StringType()),
        StructField('approvedName', StringType()),
        StructField('hgncId', ArrayType(StructType([
            StructField('id', StringType()),
            StructField('source', StringType()),
        ]))),
        StructField('hgncSynonyms', ArrayType(StructType([
            StructField('label', StringType()),
            StructField('source', StringType()),
        ]))),
        StructField('hgncSymbolSynonyms', ArrayType(StructType([
            StructField('label', StringType()),
            StructField('source', StringType()),
        ]))),
        StructField('hgncNameSynonyms', ArrayType(StructType([
            StructField('label', StringType()),
            StructField('source', StringType()),
        ]))),
        StructField('hgncObsoleteSymbols', ArrayType(StructType([
            StructField('label', StringType()),
            StructField('source', StringType()),
        ]))),
        StructField('hgncObsoleteNames', ArrayType(StructType([
            StructField('label', StringType()),
            StructField('source', StringType()),
        ]))),
        StructField('uniprotIds', ArrayType(StringType())),
    ])
    hgnc_data = [
        Row(ensemblId='ENSG00000141510',
            approvedSymbol='TP53',
            approvedName='tumor protein p53',
            hgncId=[Row(id='11998', source='HGNC')],
            hgncSynonyms=None,
            hgncSymbolSynonyms=None,
            hgncNameSynonyms=None,
            hgncObsoleteSymbols=None,
            hgncObsoleteNames=None,
            uniprotIds=['P04637']),
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
# 10. Output schema validation
# ---------------------------------------------------------------------------

REQUIRED_OUTPUT_COLUMNS = {
    'id',
    'approvedSymbol',
    'approvedName',
    'biotype',
    'transcripts',
    'genomicLocation',
    'pathways',
    'geneOntology',
    'constraint',
    'safety',
    'tractability',
    'homologues',
    'subcellularLocations',
    'targetClass',
    'hallmarks',
    'chemicalProbes',
    'tep',
}


def test_output_schema_has_required_columns(spark):
    """The target module exposes the required_output_columns constant."""
    from pts.pyspark.target import REQUIRED_OUTPUT_COLUMNS as ROC
    assert REQUIRED_OUTPUT_COLUMNS.issubset(ROC)
