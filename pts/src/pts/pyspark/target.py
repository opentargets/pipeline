"""Target dataset generation.

Ported from platform-etl-backend target step. Integrates ~15 data sources to
produce the canonical target index used by the Open Targets Platform.

Scala sources ported:
    - Target.scala (main assembly)
    - Ensembl.scala
    - GeneCode.scala
    - GeneOntology.scala
    - GeneticConstraints.scala
    - GeneWithLocation.scala
    - Hallmarks.scala
    - Hgnc.scala
    - Ncbi.scala
    - Ortholog.scala
    - ProjectScores.scala
    - ProteinClassification.scala
    - Reactome.scala
    - Safety.scala
    - Tep.scala
    - Tractability.scala
    - Uniprot.scala
"""

from __future__ import annotations

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame
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
from pyspark.sql.window import Window

from pts.pyspark.common.session import Session
from pts.pyspark.common.utils import safe_array_union as _safe_array_union

# ---------------------------------------------------------------------------
# Public constant — tested for schema compliance
# ---------------------------------------------------------------------------
REQUIRED_OUTPUT_COLUMNS = {
    'id',
    'approvedSymbol',
    'approvedName',
    'biotype',
    'transcripts',
    'transcriptIds',
    'canonicalExons',
    'genomicLocation',
    'pathways',
    'go',
    'constraint',
    'safety',
    'tractability',
    'homologues',
    'subcellularLocations',
    'targetClass',
    'hallmarks',
    'chemicalProbes',
    'tep',
    'synonyms',
    'symbolSynonyms',
    'nameSynonyms',
    'functionDescriptions',
    'proteinIds',
    'dbXrefs',
    'alternativeGenes',
    'obsoleteSymbols',
    'obsoleteNames',
    'canonicalTranscript',
    'tss',
}

# Canonical chromosomes (mirroring Ensembl.scala)
INCLUDE_CHROMOSOMES = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']

# Cancer hallmarks from Hallmarks.scala
CANCER_HALLMARKS = {
    'proliferative signalling',
    'invasion and metastasis',
    'suppression of growth',
    'angiogenesis',
    'change of cellular energetics',
    'genome instability and mutations',
    'escaping programmed cell death',
    'tumour promoting inflammation',
    'cell replicative immortality',
    'escaping immune response to cancer',
}

# Protein-ID source priority for deduplication (lower = higher priority)
_PROTEIN_SOURCE_PRIORITY = {
    'uniprot_swissprot': 1,
    'uniprot_trembl': 2,
    'uniprot': 3,
    'ensembl_PRO': 4,
}


# ===========================================================================
# Entry point
# ===========================================================================


def target(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Generate the target index dataset.

    Args:
        source: Mapping of logical input names to paths.
        destination: Dict with keys 'target' and 'gene_essentiality' output paths.
        settings: Step settings (hgncOrthologSpecies list, etc.).
        properties: Spark properties.
    """
    spark = Session(app_name='target', properties=properties).spark

    logger.info('Reading target inputs')

    # --- read inputs --------------------------------------------------------
    ensembl_raw = spark.read.parquet(source['ensembl'])
    gene_code_raw = spark.read.option('sep', '\t').option('comment', '#').csv(source['gene_code'])
    hgnc_raw = spark.read.option('multiline', 'true').json(source['hgnc'])
    hallmarks_raw = spark.read.option('sep', '\t').option('header', 'true').csv(source['hallmarks'])
    ncbi_raw = spark.read.option('sep', '\t').option('header', 'true').csv(source['ncbi'])
    go_human_raw = spark.read.option('sep', '\t').option('comment', '!').csv(source['gene_ontology_human'])
    go_rna_raw = spark.read.option('sep', '\t').option('comment', '!').csv(source['gene_ontology_rna'])
    go_rna_lookup_raw = spark.read.option('sep', '\t').csv(source['gene_ontology_rna_lookup'])
    go_eco_raw = spark.read.option('sep', '\t').option('comment', '!').csv(source['gene_ontology_eco_lookup'])
    tep_raw = spark.read.json(source['tep'])
    hpa_raw = spark.read.option('sep', '\t').option('header', 'true').csv(source['hpa'])
    hpa_sl_raw = spark.read.parquet(source['hpa_sl'])
    ps_ids_raw = spark.read.parquet(source['project_scores_ids'])
    ps_matrix_raw = spark.read.parquet(source['project_scores_essentiality_matrix'])
    chembl_raw = spark.read.json(source['chembl'])
    genetic_constraints_raw = spark.read.option('sep', '\t').option('header', 'true').csv(source['genetic_constraints'])
    homology_dict_raw = spark.read.option('sep', '\t').option('header', 'true').csv(source['homology_dictionary'])
    homology_coding_proteins_raw = (
        spark.read
        .option('recursiveFileLookup', 'true')
        .option('sep', '\t')
        .option('header', 'true')
        .csv(source['homology_coding_proteins'])
    )
    homology_gene_dict_raw = spark.read.option('recursiveFileLookup', 'true').parquet(
        source['homology_gene_dictionary']
    )
    reactome_pathways_raw = spark.read.option('sep', '\t').csv(source['reactome_pathways'])
    reactome_etl_raw = spark.read.parquet(source['reactome_etl'])
    tractability_raw = spark.read.option('sep', '\t').option('header', 'true').csv(source['tractability'])
    safety_raw = spark.read.parquet(source['safety_evidence'])
    diseases_raw = spark.read.parquet(source['diseases'])
    chemical_probes_raw = spark.read.parquet(source['chemical_probes'])
    gene_essentiality_raw = spark.read.parquet(source['gene_essentiality'])

    # Uniprot is a gzipped XML flat-file — we read a pre-processed parquet
    # produced by the pts_target pre-processing step (uniprot XML→parquet).
    # The pre-processing step writes a parquet with UniprotEntry fields.
    uniprot_raw = spark.read.parquet(source['uniprot'])
    uniprot_ssl_raw = spark.read.option('sep', '\t').option('header', 'true').csv(source['uniprot_ssl'])

    # --- species whitelist from settings ------------------------------------
    hgnc_ortholog_species: list[str] = settings.get(
        'hgncOrthologSpecies',
        [
            '9606-human',
            '9598-chimpanzee',
            '9544-macaque',
            '10090-mouse',
            '10116-rat',
            '9986-rabbit',
            '10141-guineapig',
            '9615-dog',
            '9823-pig',
            '8364-frog',
            '7955-zebrafish',
            '7227-fly',
            '6239-worm',
        ],
    )

    # --- intermediate DataFrames --------------------------------------------
    logger.info('Building GeneCode canonical transcripts')
    gene_code = _build_gene_code(gene_code_raw)

    logger.info('Building Ensembl genes')
    ensembl_df = _build_ensembl(ensembl_raw, gene_code)

    logger.info('Building HGNC')
    hgnc_df = _build_hgnc(hgnc_raw)

    logger.info('Building Hallmarks')
    hallmarks_df = _build_hallmarks(hallmarks_raw)

    logger.info('Building NCBI')
    ncbi_df = _build_ncbi(ncbi_raw)

    logger.info('Building Gene Ontology')
    go_df = _build_gene_ontology(go_human_raw, go_rna_raw, go_rna_lookup_raw, go_eco_raw, ensembl_df)

    logger.info('Building TEP')
    tep_df = _build_tep(tep_raw)

    logger.info('Building HPA subcellular locations')
    hpa_df = _build_gene_with_location(hpa_raw, hpa_sl_raw)

    logger.info('Building ProjectScores')
    project_scores_df = _build_project_scores(ps_ids_raw, ps_matrix_raw)

    logger.info('Building ProteinClassification')
    protein_class_df = _build_protein_classification(chembl_raw)

    logger.info('Building GeneticConstraints')
    genetic_constraints_df = _build_genetic_constraints(genetic_constraints_raw)

    logger.info('Building Homologues')
    homology_df = _build_homologues(
        homology_dict_raw,
        homology_coding_proteins_raw,
        homology_gene_dict_raw,
        hgnc_ortholog_species,
    )

    logger.info('Building Reactome')
    reactome_df = _build_reactome(reactome_pathways_raw, reactome_etl_raw)

    logger.info('Building Tractability')
    tractability_df = _build_tractability(tractability_raw)

    # --- Uniprot (pre-processed parquet schema mirrors UniprotEntry) --------
    logger.info('Building Uniprot')
    uniprot_df = _build_uniprot(uniprot_raw, uniprot_ssl_raw)

    # --- Merge ---------------------------------------------------------------
    logger.info('Merging HGNC + Ensembl + GO + ProjectScores + Hallmarks')
    hgnc_ensembl = _merge_hgnc_ensembl(hgnc_df, ensembl_df)

    hgnc_ensembl_go = (
        hgnc_ensembl
        .join(go_df, hgnc_ensembl['id'] == go_df['ensemblId'], 'left_outer')
        .drop('ensemblId')
        .join(project_scores_df, 'id', 'left_outer')
        .join(hallmarks_df, 'approvedSymbol', 'left_outer')
    )

    logger.info('Adding Uniprot + ProteinClassification + HPA')
    uniprot_with_class = _add_protein_classification_to_uniprot(uniprot_df, protein_class_df)
    uniprot_by_ensembl = (
        _add_ensembl_ids_to_uniprot(hgnc_df, uniprot_with_class)
        .withColumnRenamed('proteinIds', 'pid')
        .join(hpa_df, 'id', 'left_outer')
        .withColumn(
            'subcellularLocations',
            _safe_array_union(f.col('subcellularLocations'), f.col('locations')),
        )
        .drop('locations')
    )

    logger.info('Building interim target DataFrame')
    target_interim = (
        hgnc_ensembl_go
        .join(uniprot_by_ensembl, 'id', 'left_outer')
        .withColumn('proteinIds', _safe_array_union(f.col('proteinIds'), f.col('pid')))
        .withColumn(
            'dbXrefs',
            _safe_array_union(f.col('hgncId'), f.col('dbXrefs'), f.col('signalP'), f.col('xRef')),
        )
        .withColumn('symbolSynonyms', _safe_array_union(f.col('symbolSynonyms'), f.col('hgncSymbolSynonyms')))
        .withColumn('nameSynonyms', _safe_array_union(f.col('nameSynonyms'), f.col('hgncNameSynonyms')))
        .withColumn(
            'synonyms',
            _safe_array_union(f.col('synonyms'), f.col('symbolSynonyms'), f.col('nameSynonyms')),
        )
        .withColumn('obsoleteSymbols', _safe_array_union(f.col('hgncObsoleteSymbols')))
        .withColumn('obsoleteNames', _safe_array_union(f.col('hgncObsoleteNames')))
        .drop(
            'pid',
            'hgncId',
            'hgncSynonyms',
            'hgncNameSynonyms',
            'hgncSymbolSynonyms',
            'hgncObsoleteNames',
            'hgncObsoleteSymbols',
            'uniprotIds',
            'signalP',
            'xRef',
        )
        .persist()
    )

    logger.info('Building ENSG→symbol lookup')
    ensg_lookup = _generate_ensg_lookup(target_interim).persist()

    logger.info('Assembling final target DataFrame')
    targets_df = (
        target_interim
        .join(genetic_constraints_df, 'id', 'left_outer')
        .transform(lambda df: _add_tep(df, tep_df, ensg_lookup))
        .transform(_filter_and_sort_protein_ids)
        .transform(_remove_redundant_xrefs)
        .transform(lambda df: _add_chemical_probes(df, chemical_probes_raw, ensg_lookup))
        .transform(lambda df: _add_orthologues(df, homology_df))
        .transform(lambda df: _add_tractability(df, tractability_df))
        .transform(lambda df: _add_ncbi_synonyms(df, ncbi_df))
        .transform(lambda df: _add_target_safety(df, safety_raw, ensg_lookup, diseases_raw))
        .transform(lambda df: _add_reactome(df, reactome_df))
        .transform(_remove_duplicated_synonyms)
        .transform(_add_tss)
    )

    logger.info(f'Writing target output to {destination["target"]}')
    targets_df.write.mode('overwrite').parquet(destination['target'])

    logger.info('Building gene essentiality output')
    gene_essentiality_df = _build_gene_essentiality_output(gene_essentiality_raw, ensg_lookup)
    logger.info(f'Writing gene essentiality output to {destination["gene_essentiality"]}')
    gene_essentiality_df.write.mode('overwrite').parquet(destination['gene_essentiality'])


# ===========================================================================
# GeneCode.scala → _build_gene_code
# ===========================================================================


def _build_gene_code(df: DataFrame) -> DataFrame:
    """Extract canonical transcripts from GFF3 gene-code file.

    Args:
        df: Raw GFF3 DataFrame (tab-separated, no header, no comment lines).

    Returns:
        DataFrame with [id, canonicalTranscript{id, chromosome, start, end, strand}].
    """
    return (
        df
        .filter((f.col('_c2') == 'transcript') & f.col('_c8').contains('Ensembl_canonical'))
        .select(
            f.regexp_extract(f.col('_c8'), r'gene_id=(.*?);', 1).alias('gene_id_raw'),
            f.regexp_extract(f.col('_c8'), r'transcript_id=(.*?);', 1).alias('transcript_id_raw'),
            f.regexp_extract(f.col('_c0'), r'([0-9]{1,2}|X|Y|M)', 1).alias('chromosome_raw'),
            f.col('_c3').cast(LongType()).alias('start'),
            f.col('_c4').cast(LongType()).alias('end'),
            f.col('_c6').alias('strand'),
        )
        .select(
            f.regexp_extract(f.col('gene_id_raw'), r'(.*?)\.', 1).alias('id'),
            f.regexp_extract(f.col('transcript_id_raw'), r'(.*?)\.', 1).alias('ct_id'),
            f.when(f.col('chromosome_raw') == 'M', 'MT').otherwise(f.col('chromosome_raw')).alias('chromosome'),
            'start',
            'end',
            'strand',
        )
        .distinct()
        .select(
            'id',
            f.struct(
                f.col('ct_id').alias('id'),
                f.col('chromosome'),
                f.col('start'),
                f.col('end'),
                f.col('strand'),
            ).alias('canonicalTranscript'),
        )
        .withColumnRenamed('id', 'gene_id')
    )


# ===========================================================================
# Ensembl.scala → _filter_ensembl / _build_ensembl
# ===========================================================================


def _filter_ensembl(df: DataFrame) -> DataFrame:
    """Filter Ensembl genes to canonical chromosomes + reviewed genes.

    Args:
        df: Raw Ensembl DataFrame with at minimum: id, chromosome, uniprot_swissprot.

    Returns:
        Filtered DataFrame.
    """
    return df.filter(f.col('id').startswith('ENSG')).filter(
        f.col('chromosome').isin(INCLUDE_CHROMOSOMES) | f.col('uniprot_swissprot').isNotNull()
    )


def _build_ensembl(df: DataFrame, gene_code: DataFrame) -> DataFrame:
    """Build the Ensembl gene DataFrame.

    Applies canonical chromosome filter, joins canonical transcript from GeneCode,
    parses protein IDs and signalP, and deduplicates by id.

    Args:
        df: Raw Ensembl parquet (from intermediate/target/ensembl/homo_sapiens.parquet).
        gene_code: Canonical transcript lookup from _build_gene_code.

    Returns:
        Ensembl DataFrame with structured fields.
    """
    # Parse transcripts array if present in the schema
    has_transcripts = 'transcripts' in df.columns

    ensembl = (
        _filter_ensembl(df)
        .select(
            f.trim(f.col('id')).alias('id'),
            f.regexp_replace(f.col('biotype'), r'(?i)tec', '').alias('biotype'),
            f.col('description'),
            f.col('end').cast(LongType()).alias('end'),
            f.col('start').cast(LongType()).alias('start'),
            f.col('strand').cast(IntegerType()).alias('strand'),
            f.col('chromosome'),
            f.col('approvedSymbol'),
            (
                f.col('transcripts')
                if has_transcripts
                else f.lit(None).cast(ArrayType(StructType([StructField('id', StringType())])))
            ).alias('transcripts_raw'),
            # transcriptIds: flat array of transcript IDs (Ensembl.scala line 45)
            (f.col('transcripts.id') if has_transcripts else f.lit(None).cast(ArrayType(StringType()))).alias(
                'transcriptIds'
            ),
            # exons: per-transcript exon arrays for canonicalExons computation
            (f.col('transcripts.exons') if has_transcripts else f.lit(None)).alias('exons_raw'),
            # translations: flatten transcripts[*].translations[*] into a top-level array
            # so _refactor_ensembl_protein_ids can build ensembl_PRO protein IDs
            (f.flatten(f.col('transcripts.translations')) if has_transcripts else f.lit(None)).alias('translations'),
            f.col('SignalP').alias('signalP') if 'SignalP' in df.columns else f.lit(None).alias('signalP'),
            f.col('uniprot_trembl') if 'uniprot_trembl' in df.columns else f.lit(None).alias('uniprot_trembl'),
            f.col('uniprot_swissprot') if 'uniprot_swissprot' in df.columns else f.lit(None).alias('uniprot_swissprot'),
        )
        .orderBy('id')
        .dropDuplicates(['id'])
    )

    # Join canonical transcript
    ensembl = ensembl.join(
        gene_code.withColumnRenamed('gene_id', 'ct_gene_id'),
        (ensembl['id'] == f.col('ct_gene_id')) & (ensembl['chromosome'] == f.col('canonicalTranscript.chromosome')),
        'left_outer',
    ).drop('ct_gene_id')

    # Build genomicLocation struct
    ensembl = ensembl.withColumn(
        'genomicLocation',
        f.struct(
            f.col('chromosome'),
            f.col('start'),
            f.col('end'),
            f.col('strand'),
        ),
    )

    # approvedName from description
    ensembl = (
        ensembl
        .withColumn('_desc_parts', f.split(f.col('description'), r'\['))
        .withColumn('approvedName', f.regexp_replace(f.element_at(f.col('_desc_parts'), 1), r'(?i)tec', ''))
        .drop('_desc_parts', 'description')
    )

    # Parse transcripts: extract id, biotype, uniprotId, etc.
    id_source_schema = ArrayType(
        StructType([
            StructField('id', StringType()),
            StructField('source', StringType()),
        ])
    )

    # Build proteinIds from uniprot columns
    ensembl = _refactor_ensembl_protein_ids(ensembl)

    # Build signalP — cast to explicit array type first to handle VOID (all-null) columns
    ensembl = ensembl.withColumn('signalP', f.col('signalP').cast(ArrayType(StringType())))
    ensembl = ensembl.withColumn(
        'signalP',
        f.when(
            f.col('signalP').isNotNull() & (f.size(f.col('signalP')) >= 0),
            f.transform(f.col('signalP'), lambda c: f.struct(c.alias('id'), f.lit('signalP').alias('source'))),
        ).cast(id_source_schema),
    )

    # Build canonicalExons: flat array of (start, end) pairs for the canonical transcript
    # Mirrors addCanonicalExons in Ensembl.scala
    # Use expr because array_position(col, col) requires SQL expression syntax in PySpark
    ensembl = ensembl.withColumn(
        'exonIndex',
        f.expr('array_position(transcriptIds, canonicalTranscript.id)'),
    )
    ensembl = ensembl.withColumn(
        '_canon_exons_raw',
        f.when(
            f.col('exonIndex') > 0,
            f.element_at(f.col('exons_raw'), f.col('exonIndex').cast(IntegerType())),
        ),
    )
    ensembl = ensembl.withColumn(
        'canonicalExons',
        f.when(
            f.col('_canon_exons_raw').isNotNull(),
            f.flatten(
                f.transform(
                    f.col('_canon_exons_raw'),
                    lambda x: f.array(x.getField('start'), x.getField('end')),
                )
            ),
        ),
    ).drop('exonIndex', 'exons_raw', '_canon_exons_raw')

    # Parse transcripts into the structured format (also sets isEnsemblCanonical)
    ensembl = _parse_ensembl_transcripts(ensembl)

    return ensembl.select(
        'id',
        'biotype',
        'approvedName',
        'alternativeGenes',
        'genomicLocation',
        'approvedSymbol',
        'proteinIds',
        'transcriptIds',
        'canonicalExons',
        'transcripts',
        'signalP',
        'canonicalTranscript',
    )


def _refactor_ensembl_protein_ids(df: DataFrame) -> DataFrame:
    """Build proteinIds from uniprot_swissprot, uniprot_trembl columns.

    Also sets alternativeGenes to null (filled later).
    """
    id_source_schema = ArrayType(
        StructType([
            StructField('id', StringType()),
            StructField('source', StringType()),
        ])
    )

    has_translations = 'translations' in df.columns

    swiss = f.when(
        f.size(f.col('uniprot_swissprot')) >= 0,
        f.transform(
            f.col('uniprot_swissprot'),
            lambda c: f.struct(c.alias('id'), f.lit('uniprot_swissprot').alias('source')),
        ),
    ).cast(id_source_schema)
    trembl = f.when(
        f.size(f.col('uniprot_trembl')) >= 0,
        f.transform(
            f.col('uniprot_trembl'),
            lambda c: f.struct(c.alias('id'), f.lit('uniprot_trembl').alias('source')),
        ),
    ).cast(id_source_schema)

    df = df.withColumn('_swiss', swiss).withColumn('_trembl', trembl)

    if has_translations:
        ensembl_pro = f.when(
            f.size(f.col('translations')) >= 0,
            f.transform(
                f.col('translations'),
                lambda c: f.struct(c.getField('id').alias('id'), f.lit('ensembl_PRO').alias('source')),
            ),
        ).cast(id_source_schema)
        df = df.withColumn('_ensembl_pro', ensembl_pro)
        protein_ids = _safe_array_union(f.col('_swiss'), f.col('_trembl'), f.col('_ensembl_pro'))
        df = df.withColumn('proteinIds', protein_ids).drop(
            '_swiss',
            '_trembl',
            '_ensembl_pro',
            'translations',
            'uniprot_swissprot',
            'uniprot_trembl',
        )
    else:
        protein_ids = _safe_array_union(f.col('_swiss'), f.col('_trembl'))
        df = df.withColumn('proteinIds', protein_ids).drop('_swiss', '_trembl', 'uniprot_swissprot', 'uniprot_trembl')

    return df.withColumn('alternativeGenes', f.lit(None).cast(ArrayType(StringType())))


def _parse_ensembl_transcripts(df: DataFrame) -> DataFrame:
    """Parse raw transcripts array into structured transcript objects."""
    transcript_schema = ArrayType(
        StructType([
            StructField('transcriptId', StringType()),
            StructField('biotype', StringType()),
            StructField('uniprotId', StringType()),
            StructField('isUniprotReviewed', BooleanType()),
            StructField('translationId', StringType()),
            StructField('alphafoldId', StringType()),
            StructField('uniprotIsoformId', StringType()),
            StructField('isEnsemblCanonical', BooleanType()),
        ])
    )

    if 'transcripts_raw' not in df.columns:
        return df.withColumn('transcripts', f.lit(None).cast(transcript_schema))

    canon_id = f.col('canonicalTranscript.id')

    parsed = f.when(
        f.col('transcripts_raw').isNotNull(),
        f.transform(
            f.col('transcripts_raw'),
            lambda tr: f.struct(
                tr.getField('id').alias('transcriptId'),
                tr.getField('biotype').alias('biotype'),
                f
                .when(tr.getField('uniprot_swissprot').isNotNull(), f.element_at(tr.getField('uniprot_swissprot'), 1))
                .when(tr.getField('uniprot_trembl').isNotNull(), f.element_at(tr.getField('uniprot_trembl'), 1))
                .alias('uniprotId'),
                f
                .when(tr.getField('uniprot_swissprot').isNotNull(), f.lit(True))
                .when(tr.getField('uniprot_trembl').isNotNull(), f.lit(False))
                .alias('isUniprotReviewed'),
                f.when(
                    tr.getField('translations').isNotNull(),
                    f.element_at(tr.getField('translations'), 1).getField('id'),
                ).alias('translationId'),
                f.when(
                    tr.getField('alphafold').isNotNull(),
                    f.element_at(tr.getField('alphafold'), 1),
                ).alias('alphafoldId'),
                f.when(
                    tr.getField('uniprot_isoform').isNotNull(),
                    f.element_at(tr.getField('uniprot_isoform'), 1),
                ).alias('uniprotIsoformId'),
                # isEnsemblCanonical: true when this transcript is the canonical one
                f
                .when(canon_id.isNotNull(), tr.getField('id') == canon_id)
                .cast(BooleanType())
                .alias('isEnsemblCanonical'),
            ),
        ),
    ).cast(transcript_schema)

    return df.withColumn('transcripts', parsed).drop('transcripts_raw')


# ===========================================================================
# Hgnc.scala → _build_hgnc
# ===========================================================================


def _build_hgnc(df: DataFrame) -> DataFrame:
    """Build the HGNC mapping DataFrame.

    Args:
        df: Raw HGNC JSON with nested response.docs array.

    Returns:
        DataFrame with [ensemblId, hgncId, approvedSymbol, approvedName,
        hgncSynonyms, hgncSymbolSynonyms, hgncNameSynonyms,
        hgncObsoleteSymbols, hgncObsoleteNames, uniprotIds].
    """
    label_source_schema = ArrayType(
        StructType([
            StructField('label', StringType()),
            StructField('source', StringType()),
        ])
    )
    id_source_schema = ArrayType(
        StructType([
            StructField('id', StringType()),
            StructField('source', StringType()),
        ])
    )

    docs = df.select(f.explode(f.col('response.docs')).alias('doc')).select('doc.*')

    def _array_to_label_source(col_expr, source_val):
        return f.when(
            col_expr.isNotNull(),
            f.transform(col_expr, lambda c: f.struct(c.alias('label'), f.lit(source_val).alias('source'))),
        ).cast(label_source_schema)

    hgnc = (
        docs
        .select(
            f.col('ensembl_gene_id').alias('ensemblId'),
            f.split(f.col('hgnc_id'), ':').alias('_hgnc_parts'),
            f.col('symbol').alias('approvedSymbol'),
            f.col('name').alias('approvedName'),
            f.col('uniprot_ids').alias('uniprotIds'),
            _safe_array_union(
                f.col('prev_name'),
                f.col('prev_symbol'),
                f.col('alias_symbol'),
                f.col('alias_name'),
            ).alias('_all_synonyms'),
            _safe_array_union(f.col('alias_symbol')).alias('_symbol_synonyms'),
            _safe_array_union(f.col('alias_name')).alias('_name_synonyms'),
            _safe_array_union(f.col('prev_symbol')).alias('_obsolete_symbols'),
            _safe_array_union(f.col('prev_name')).alias('_obsolete_names'),
        )
        .withColumn(
            'hgncId',
            f.array(
                f.struct(
                    f.element_at(f.col('_hgnc_parts'), 2).alias('id'),
                    f.element_at(f.col('_hgnc_parts'), 1).alias('source'),
                )
            ).cast(id_source_schema),
        )
        .drop('_hgnc_parts')
    )

    for src_col, dst_col in [
        ('_all_synonyms', 'hgncSynonyms'),
        ('_symbol_synonyms', 'hgncSymbolSynonyms'),
        ('_name_synonyms', 'hgncNameSynonyms'),
        ('_obsolete_symbols', 'hgncObsoleteSymbols'),
        ('_obsolete_names', 'hgncObsoleteNames'),
    ]:
        hgnc = hgnc.withColumn(
            dst_col,
            _array_to_label_source(f.col(src_col), 'HGNC'),
        ).drop(src_col)

    return hgnc.dropDuplicates(['ensemblId'])


# ===========================================================================
# Hallmarks.scala → _build_hallmarks
# ===========================================================================


def _build_hallmarks(df: DataFrame) -> DataFrame:
    """Build the hallmarks DataFrame.

    Args:
        df: Raw COSMIC hallmarks TSV with GENE_SYMBOL, PUBMED_PMID, HALLMARK, IMPACT, DESCRIPTION.

    Returns:
        DataFrame with [approvedSymbol, hallmarks{attributes, cancerHallmarks}].
    """
    cancer_hallmarks_list = list(CANCER_HALLMARKS)

    processed = df.select(
        f.col('GENE_SYMBOL').alias('gene_symbol'),
        f.col('PUBMED_PMID').cast(LongType()).alias('pmid'),
        f.col('HALLMARK').alias('hallmark'),
        f.col('IMPACT').alias('impact'),
        f.col('DESCRIPTION').alias('description'),
    ).withColumn('is_cancer_hallmark', f.col('hallmark').isin(cancer_hallmarks_list))

    cancer_df = (
        processed
        .filter(f.col('is_cancer_hallmark'))
        .select(
            'gene_symbol',
            f.struct(
                f.col('pmid'),
                f.col('description'),
                f.col('impact'),
                f.col('hallmark').alias('label'),
            ).alias('cancerHallmarks'),
        )
        .groupBy('gene_symbol')
        .agg(f.collect_set('cancerHallmarks').alias('cancerHallmarks'))
    )

    attrs_df = (
        processed
        .filter(~f.col('is_cancer_hallmark'))
        .select(
            'gene_symbol',
            f.struct(
                f.col('pmid'),
                f.col('description'),
                f.col('hallmark').alias('name'),
            ).alias('attributes'),
        )
        .groupBy('gene_symbol')
        .agg(f.collect_set('attributes').alias('attributes'))
    )

    return (
        processed
        .select('gene_symbol')
        .distinct()
        .join(cancer_df, 'gene_symbol', 'left_outer')
        .join(attrs_df, 'gene_symbol', 'left_outer')
        .select(
            f.col('gene_symbol').alias('approvedSymbol'),
            f.struct(
                f.col('attributes'),
                f.col('cancerHallmarks'),
            ).alias('hallmarks'),
        )
    )


# ===========================================================================
# Ncbi.scala → _build_ncbi
# ===========================================================================


def _build_ncbi(df: DataFrame) -> DataFrame:
    """Build NCBI Entrez synonym DataFrame.

    Args:
        df: Raw NCBI gene_info TSV.

    Returns:
        DataFrame with [id, synonyms, symbolSynonyms, nameSynonyms] (LabelAndSource arrays).
    """
    label_source_schema = ArrayType(
        StructType([
            StructField('label', StringType()),
            StructField('source', StringType()),
        ])
    )

    sep = r'\|'

    ncbi = (
        df
        .select(
            f.split(f.col('Symbol'), sep).alias('sy'),
            f.split(f.col('dbXrefs'), sep).alias('id_parts'),
            f.split(f.col('Synonyms'), sep).alias('s'),
            f.split(f.col('Other_designations'), sep).alias('od'),
        )
        .withColumn('id_part', f.explode(f.col('id_parts')))
        .filter(f.col('id_part').startswith('Ensembl'))
        .withColumn('id_split', f.split(f.col('id_part'), ':'))
        .withColumn('ensg', f.explode(f.col('id_split')))
        .filter(f.col('ensg').startswith('ENSG'))
        .select(
            f.col('ensg').alias('id'),
            _safe_array_union(f.col('s'), f.col('od'), f.col('sy')).alias('synonyms'),
            _safe_array_union(f.col('s'), f.col('sy')).alias('symbolSynonyms'),
            _safe_array_union(f.col('od')).alias('nameSynonyms'),
        )
        .groupBy('id')
        .agg(
            f.flatten(f.collect_set('synonyms')).alias('synonyms'),
            f.flatten(f.collect_set('symbolSynonyms')).alias('symbolSynonyms'),
            f.flatten(f.collect_set('nameSynonyms')).alias('nameSynonyms'),
        )
    )

    def _to_label_source(col_name):
        return f.when(
            f.col(col_name).isNotNull(),
            f.transform(
                f.filter(f.col(col_name), lambda c: c != '-'),
                lambda c: f.struct(c.alias('label'), f.lit('NCBI_entrez').alias('source')),
            ),
        ).cast(label_source_schema)

    for col_name in ['synonyms', 'symbolSynonyms', 'nameSynonyms']:
        ncbi = ncbi.withColumn(col_name, _to_label_source(col_name))

    return ncbi


# ===========================================================================
# GeneOntology.scala → _build_gene_ontology
# ===========================================================================


def _build_gene_ontology(
    human: DataFrame,
    rna: DataFrame,
    rna_lookup: DataFrame,
    eco_lookup: DataFrame,
    ensembl_df: DataFrame,
) -> DataFrame:
    """Build gene ontology annotations grouped by Ensembl ID.

    Args:
        human: Raw human GAF file (17 columns, tab-separated).
        rna: Raw RNA GAF file.
        rna_lookup: RNACentral → Ensembl mapping TSV.
        eco_lookup: ECO lookup GPA file.
        ensembl_df: Ensembl DataFrame (used to map UniprotKB → ENSG).

    Returns:
        DataFrame with [ensemblId, go[{id, source, evidence, aspect, geneProduct, ecoId}]].
    """
    col_names = [
        'database',
        'dbObjectId',
        'dbObjectSymbol',
        'qualifier',
        'goId',
        'source',
        'evidence',
        'withOrFrom',
        'aspect',
        'dbObjectName',
        'dbObjectSynonym',
        'dbObjectType',
        'taxon',
        'date',
        'assignedBy',
        'annotationExtension',
        'geneProductFormId',
    ]

    def _extract_go_cols(df):
        n = len(df.columns)
        actual_names = col_names[:n]
        renamed = df.toDF(*actual_names)
        return renamed.select(
            f.col('dbObjectId'),
            f.col('goId'),
            f.col('source'),
            f.col('evidence'),
            f.col('aspect'),
            f.col('dbObjectId').alias('geneProduct'),
        )

    human_go = _extract_go_cols(human).orderBy('goId')

    rna_go = (
        _extract_go_cols(rna)
        .withColumn('dbObjectId', f.element_at(f.split(f.col('dbObjectId'), '_', 0), 1))
        .orderBy('goId')
    )

    # RNA lookup: rnaCentralId → ensemblId
    rna_lu_cols = ['rnaCentralId', 'database', 'externalId', 'ncbiTaxonId', 'rnaType', 'ensemblId']
    n_rna = len(rna_lookup.columns)
    rna_lu = (
        rna_lookup
        .toDF(*rna_lu_cols[:n_rna])
        .select('rnaCentralId', 'ensemblId')
        .filter(f.col('ensemblId').startswith('ENSG0'))
        .withColumn('ensemblId', f.regexp_extract(f.col('ensemblId'), r'ENSG[0-9]+', 0))
    )

    # Ensembl lookup: uniprotId → ensemblId
    # Mirrors Scala: map uniprot proteinIds + approvedSymbol → ensemblId
    human_lu = (
        ensembl_df
        .filter(f.col('proteinIds').isNotNull())
        .select(
            f.col('id').alias('ensemblId'),
            f.col('approvedSymbol'),
            f.col('proteinIds'),
        )
        # Explode proteinIds to get one row per protein accession
        .withColumn('pid', f.explode(_safe_array_union(f.col('proteinIds.id'), f.array(f.col('approvedSymbol')))))
        .drop('proteinIds', 'approvedSymbol')
        .withColumnRenamed('pid', 'uniprotId')
    )

    # ECO lookup: (goId, geneProduct, evidence) → ecoId
    if eco_lookup.columns and len(eco_lookup.columns) >= 12:
        eco_lu = eco_lookup.select(
            f.col('_c3').alias('goId'),
            f.col('_c1').alias('geneProduct'),
            f.element_at(f.split(f.col('_c11'), '='), 2).alias('evidence'),
            f.col('_c5').alias('ecoId'),
        ).distinct()
    else:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        eco_lu = spark.createDataFrame(
            [],
            StructType([
                StructField('goId', StringType()),
                StructField('geneProduct', StringType()),
                StructField('evidence', StringType()),
                StructField('ecoId', StringType()),
            ]),
        )

    def _nest_go(df):
        return (
            df
            .drop('dbObjectId')
            .withColumnRenamed('goId', 'id')
            .select(
                'ensemblId',
                f.struct('id', 'source', 'evidence', 'aspect', 'geneProduct', 'ecoId').alias('go'),
            )
            .groupBy('ensemblId')
            .agg(f.collect_set('go').alias('go'))
            .withColumn('go', f.coalesce(f.col('go'), f.array()))
        )

    rna_with_ensembl = (
        rna_go
        .join(rna_lu, rna_go['dbObjectId'] == rna_lu['rnaCentralId'])
        .join(eco_lu, ['goId', 'geneProduct', 'evidence'], 'left_outer')
        .drop('rnaCentralId')
        .transform(_nest_go)
    )

    human_with_ensembl = (
        human_go
        .join(human_lu, human_go['dbObjectId'] == human_lu['uniprotId'])
        .join(eco_lu, ['goId', 'geneProduct', 'evidence'], 'left_outer')
        .drop('uniprotId')
        .transform(_nest_go)
    )

    return (
        rna_with_ensembl
        .withColumnRenamed('go', 'go_rna')
        .join(human_with_ensembl, 'ensemblId', 'outer')
        .select(
            'ensemblId',
            f.array_union(
                f.coalesce(f.col('go'), f.array()),
                f.coalesce(f.col('go_rna'), f.array()),
            ).alias('go'),
        )
    )


# ===========================================================================
# GeneWithLocation.scala → _build_gene_with_location
# ===========================================================================


def _build_gene_with_location(df: DataFrame, sl_df: DataFrame) -> DataFrame:
    """Build subcellular locations from HPA data.

    Args:
        df: Raw HPA TSV with Gene, Main location, Additional location, Extracellular location.
        sl_df: Subcellular location SSL ontology (HPA_location → termSL, labelSL).

    Returns:
        DataFrame with [id, locations[{location, source, termSL, labelSL}]].
    """

    def _to_loc_source(col_expr, source_val):
        return f.when(
            col_expr.isNotNull(),
            f.transform(
                f.split(col_expr, ';'),
                lambda c: f.struct(f.trim(c).alias('location'), f.lit(source_val).alias('source')),
            ),
        )

    return (
        df
        .select(
            f.col('Gene').alias('id'),
            _to_loc_source(f.col('Main location'), 'HPA_main').alias('HPA_main'),
            _to_loc_source(f.col('Additional location'), 'HPA_additional').alias('HPA_additional'),
            _to_loc_source(
                f.col('Extracellular location'),
                'HPA_extracellular_location',
            ).alias('HPA_extracellular_location'),
        )
        .withColumn(
            'all_locations',
            f.explode(
                _safe_array_union(f.col('HPA_main'), f.col('HPA_additional'), f.col('HPA_extracellular_location')),
            ),
        )
        .select('id', f.col('all_locations.location').alias('location'), f.col('all_locations.source').alias('source'))
        .join(sl_df, f.col('location') == sl_df['HPA_location'], 'left_outer')
        .withColumn('targetModifier', f.lit(None).cast(StringType()))
        .select(
            'id',
            f.struct('location', 'source', 'termSL', 'labelSL', 'targetModifier').alias('locations'),
        )
        .groupBy('id')
        .agg(f.collect_list('locations').alias('locations'))
    )


# ===========================================================================
# ProjectScores.scala → _build_project_scores
# ===========================================================================


def _build_project_scores(ids_df: DataFrame, matrix_df: DataFrame) -> DataFrame:
    """Build project scores (DepMap) cross-references.

    Args:
        ids_df: Gene identifiers mapping (gene_id, ensembl_gene_id, hgnc_symbol).
        matrix_df: Binary dependency scores matrix (Gene column + per-cell-line columns).

    Returns:
        DataFrame with [id, xRef[{id, source}]].
    """
    id_source_schema = ArrayType(
        StructType([
            StructField('id', StringType()),
            StructField('source', StringType()),
        ])
    )

    ps_ids = ids_df.filter(f.col('ensembl_gene_id').isNotNull()).select(
        f.col('gene_id').alias('ps_gene_id'),
        f.col('ensembl_gene_id'),
        f.col('hgnc_symbol'),
    )

    # Build a proper total using stack expression
    data_cols = [c for c in matrix_df.columns if c != 'Gene']
    if data_cols:
        stack_expr = f'stack({len(data_cols)}, {", ".join([f"`{c}`" for c in data_cols])}) as val'
        total_df = (
            matrix_df
            .select('Gene', f.expr(stack_expr).alias('val'))
            .groupBy('Gene')
            .agg(f.sum('val').alias('total'))
            .filter(f.col('total') > 0)
        )

    return total_df.join(ps_ids, total_df['Gene'] == ps_ids['hgnc_symbol']).select(
        f.col('ensembl_gene_id').alias('id'),
        f
        .array(
            f.struct(
                f.col('ps_gene_id').alias('id'),
                f.lit('ProjectScore').alias('source'),
            )
        )
        .cast(id_source_schema)
        .alias('xRef'),
    )


# ===========================================================================
# ProteinClassification.scala → _build_protein_classification
# ===========================================================================


def _build_protein_classification(df: DataFrame) -> DataFrame:
    """Build protein target classification from ChEMBL.

    Args:
        df: Raw ChEMBL target JSONL.

    Returns:
        DataFrame with [accession, targetClass[{id, label, level}]].
    """
    accession_pc = df.select(
        f.explode(
            f.arrays_zip(
                f.col('_metadata.protein_classification'),
                f.col('target_components.accession'),
            )
        ).alias('s')
    ).select(
        f.col('s.accession').alias('accession'),
        f.col('s.protein_classification.*'),
    )

    levels = [f'l{i}' for i in range(1, 7)]

    def _to_struct(level):
        return f.struct(
            f.col('protein_class_id').alias('id'),
            f.col(level).alias('label'),
            f.lit(level).alias('level'),
        )

    expanded = accession_pc
    for lvl in levels:
        expanded = expanded.withColumn(lvl, _to_struct(lvl))

    return (
        expanded
        .select('accession', f.array(*levels).alias('levels'))
        .groupBy('accession')
        .agg(f.flatten(f.collect_set('levels')).alias('levels'))
        .select('accession', f.explode('levels').alias('l'))
        .select('accession', f.col('l.*'))
        .filter(f.col('label').isNotNull())
        .select('accession', f.struct('id', 'label', 'level').alias('pc'))
        .groupBy('accession')
        .agg(f.collect_set('pc').alias('targetClass'))
    )


# ===========================================================================
# GeneticConstraints.scala → _build_genetic_constraints
# ===========================================================================


def _build_genetic_constraints(df: DataFrame) -> DataFrame:
    """Build genetic constraints (gnomAD).

    Args:
        df: Raw gnomAD constraint metrics TSV.

    Returns:
        DataFrame with [id, constraint[{constraintType, score, exp, obs, oe, oeLower, oeUpper,
        upperRank, upperBin, upperBin6}]].
    """
    filtered = df.filter((f.col('canonical') == 'true') & (f.col('transcript_type') != 'NA'))

    # Compute sextile bin for lof
    w = Window.orderBy(f.col('`lof.oe_ci.upper_rank`').cast(IntegerType()))
    filtered = filtered.withColumn(
        'lof_upper_bin6',
        f.when(
            f.col('`lof.oe_ci.upper_rank`') != 'NA',
            f.ntile(6).over(w) - 1,
        ).otherwise(None),
    )

    return filtered.select(
        f.col('gene_id').cast(StringType()).alias('id'),
        f.array(
            f.struct(
                f.lit('syn').alias('constraintType'),
                f.col('`syn.z_score`').cast(FloatType()).alias('score'),
                f.col('`syn.exp`').cast(FloatType()).alias('exp'),
                f.col('`syn.obs`').cast(IntegerType()).alias('obs'),
                f.col('`syn.oe`').cast(FloatType()).alias('oe'),
                f.col('`syn.oe_ci.lower`').cast(FloatType()).alias('oeLower'),
                f.col('`syn.oe_ci.upper`').cast(FloatType()).alias('oeUpper'),
                f.lit(None).cast(IntegerType()).alias('upperRank'),
                f.lit(None).cast(IntegerType()).alias('upperBin'),
                f.lit(None).cast(IntegerType()).alias('upperBin6'),
            ),
            f.struct(
                f.lit('mis').alias('constraintType'),
                f.col('`mis.z_score`').cast(FloatType()).alias('score'),
                f.col('`mis.exp`').cast(FloatType()).alias('exp'),
                f.col('`mis.obs`').cast(IntegerType()).alias('obs'),
                f.col('`mis.oe`').cast(FloatType()).alias('oe'),
                f.col('`mis.oe_ci.lower`').cast(FloatType()).alias('oeLower'),
                f.col('`mis.oe_ci.upper`').cast(FloatType()).alias('oeUpper'),
                f.lit(None).cast(IntegerType()).alias('upperRank'),
                f.lit(None).cast(IntegerType()).alias('upperBin'),
                f.lit(None).cast(IntegerType()).alias('upperBin6'),
            ),
            f.struct(
                f.lit('lof').alias('constraintType'),
                f.col('`lof.pLI`').cast(FloatType()).alias('score'),
                f.col('`lof.exp`').cast(FloatType()).alias('exp'),
                f.col('`lof.obs`').cast(IntegerType()).alias('obs'),
                f.col('`lof.oe`').cast(FloatType()).alias('oe'),
                f.col('`lof.oe_ci.lower`').cast(FloatType()).alias('oeLower'),
                f.col('`lof.oe_ci.upper`').cast(FloatType()).alias('oeUpper'),
                f.col('`lof.oe_ci.upper_rank`').cast(IntegerType()).alias('upperRank'),
                f.col('`lof.oe_ci.upper_bin_decile`').cast(IntegerType()).alias('upperBin'),
                f.col('lof_upper_bin6').cast(IntegerType()).alias('upperBin6'),
            ),
        ).alias('constraint'),
    )


# ===========================================================================
# Ortholog.scala → _build_homologues
# ===========================================================================


def _build_homologues(
    homology_dict: DataFrame,
    coding_proteins: DataFrame,
    gene_dict: DataFrame,
    target_species: list[str],
) -> DataFrame:
    """Build homologue/ortholog DataFrame.

    Args:
        homology_dict: Ensembl vertebrates species dictionary.
        coding_proteins: Ensembl compara homologies TSV (protein + ncrna).
        gene_dict: Gene ID → gene name mapping (pre-processed parquet).
        target_species: Whitelisted species in format "TAXID-species_name".

    Returns:
        DataFrame with homolog fields including speciesId, speciesName, homologyType, etc.
    """
    # Extract tax IDs from whitelist
    tax_ids = [s.split('-')[0] for s in target_species]
    priority_df_data = [(s.split('-')[0], i) for i, s in enumerate(target_species)]

    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    priority_df = spark.createDataFrame(priority_df_data, ['speciesId', 'priority']).withColumn(
        'priority', f.col('priority').cast('int')
    )

    homo_dict = homology_dict.select(
        f.col('#name').alias('name'),
        f.col('species').alias('speciesName'),
        f.col('taxonomy_id'),
        f.array(*[f.lit(t) for t in tax_ids]).alias('whitelist'),
    ).filter(f.array_contains(f.col('whitelist'), f.col('taxonomy_id')))

    gene_dict_mapped = gene_dict.select(
        f.col('id').alias('homology_gene_stable_id'),
        f
        .when(f.col('name').isNotNull() & (f.col('name') != ''), f.col('name'))  # noqa: PLC1901
        .otherwise(f.col('id'))
        .alias('targetGeneSymbol'),
    )

    reference = 'homo_sapiens'

    # homo_sapiens homologies
    homo_sapiens_h = coding_proteins.filter(f.col('species') == reference)

    # paralogs and cross-species
    other_h = (
        coding_proteins.filter(
            (
                (f.col('species') == reference)
                & ((f.col('homology_type') == 'other_paralog') | (f.col('homology_type') == 'within_species_paralog'))
            )
            | ((f.col('species') != reference) & (f.col('homology_species') == reference))
        )
        # swap homo_sapiens ↔ homology columns
        .select(
            f.col('homology_gene_stable_id').alias('gene_stable_id'),
            f.col('homology_protein_stable_id').alias('protein_stable_id'),
            f.col('homology_species').alias('species'),
            f.col('homology_identity').alias('identity'),
            f.col('homology_type'),
            f.col('gene_stable_id').alias('homology_gene_stable_id'),
            f.col('protein_stable_id').alias('homology_protein_stable_id'),
            f.col('species').alias('homology_species'),
            f.col('identity').alias('homology_identity'),
            f.col('dn'),
            f.col('ds'),
            f.col('goc_score'),
            f.col('wga_coverage'),
            f.col('is_high_confidence'),
            f.col('homology_id'),
        )
    )

    all_homologies = homo_sapiens_h.unionByName(other_h)

    return (
        all_homologies
        .join(homo_dict, all_homologies['homology_species'] == homo_dict['speciesName'])
        .join(gene_dict_mapped, 'homology_gene_stable_id', 'left_outer')
        .select(
            f.col('gene_stable_id').alias('id'),
            f.col('taxonomy_id').alias('speciesId'),
            f.col('name').alias('speciesName'),
            f.col('homology_type').alias('homologyType'),
            f.col('homology_gene_stable_id').alias('targetGeneId'),
            f.col('is_high_confidence').alias('isHighConfidence'),
            f.col('targetGeneSymbol'),
            f.col('identity').cast(DoubleType()).alias('queryPercentageIdentity'),
            f.col('homology_identity').cast(DoubleType()).alias('targetPercentageIdentity'),
        )
        .join(f.broadcast(priority_df), 'speciesId', 'left_outer')
    )


# ===========================================================================
# Reactome.scala → _build_reactome
# ===========================================================================


def _build_reactome(reactome_pathways: DataFrame, reactome_etl: DataFrame) -> DataFrame:
    """Build reactome pathways grouped by Ensembl ID.

    Args:
        reactome_pathways: Ensembl2Reactome.txt (6-column TSV).
        reactome_etl: Reactome ETL output (id, label, path).

    Returns:
        DataFrame with [id, pathways[{pathwayId, pathway, topLevelTerm}]].
    """
    rp_cols = ['ensemblId', 'reactomeId', 'url', 'eventName', 'eventCode', 'species']

    rp = (
        reactome_pathways
        .toDF(*rp_cols)
        .filter(f.col('species') == 'Homo sapiens')
        .filter(f.col('ensemblId').startswith('ENSG'))
        .select('ensemblId', 'reactomeId', 'eventName')
    )

    top_level_terms = (
        reactome_etl
        .select(
            f.col('id'),
            f.element_at(f.element_at(f.col('path'), 1), 1).alias('tlId'),
        )
        .join(
            reactome_etl.select(f.col('id').alias('tlId'), f.col('label').alias('topLevelTerm')),
            'tlId',
        )
        .select(f.col('id').alias('reactomeId'), 'topLevelTerm')
    )

    return (
        rp
        .join(top_level_terms, 'reactomeId')
        .groupBy('ensemblId')
        .agg(
            f.collect_set(
                f.struct(
                    f.col('reactomeId').alias('pathwayId'),
                    f.col('eventName').alias('pathway'),
                    f.col('topLevelTerm'),
                )
            ).alias('pathways')
        )
        .withColumnRenamed('ensemblId', 'id')
    )


# ===========================================================================
# Tractability.scala → _build_tractability
# ===========================================================================


def _build_tractability(df: DataFrame) -> DataFrame:
    """Build tractability assessments.

    Args:
        df: Raw tractability TSV. Columns with pattern *_B{N}_* are tractability buckets.

    Returns:
        DataFrame with [ensemblGeneId, tractability[{modality, id, value}]].
    """
    import re

    bucket_cols = [c for c in df.columns if re.match(r'.*_B\d+_.*', c)]
    tractability = df.select('ensembl_gene_id', *bucket_cols)
    data_cols = [c for c in tractability.columns if c != 'ensembl_gene_id']

    for col_name in data_cols:
        parts = col_name.split('_')
        tractability = tractability.withColumn(
            col_name,
            f.struct(
                f.lit(parts[0]).alias('modality'),
                f.lit(parts[-1]).alias('id'),
                f.when(f.col(f'`{col_name}`') == 1, True).otherwise(False).alias('value'),
            ),
        )

    return tractability.select(
        f.col('ensembl_gene_id').alias('ensemblGeneId'),
        f.array(*data_cols).alias('tractability'),
    )


# ===========================================================================
# Uniprot (pre-processed parquet) → _build_uniprot
# ===========================================================================


def _build_uniprot(df: DataFrame, ssl_df: DataFrame) -> DataFrame:
    """Build Uniprot DataFrame from pre-processed parquet.

    The pre-processing step in pts_pre_target converts the Uniprot XML flat-file
    into a parquet with UniprotEntry fields:
        accessions, names, synonyms, symbolSynonyms, functions, dbXrefs, locations.

    Args:
        df: Pre-processed Uniprot parquet.
        ssl_df: Subcellular location SSL ontology mapping.

    Returns:
        DataFrame with [uniprotId, synonyms, symbolSynonyms, nameSynonyms,
        functionDescriptions, proteinIds, subcellularLocations, dbXrefs].
    """
    label_source_schema = ArrayType(
        StructType([
            StructField('label', StringType()),
            StructField('source', StringType()),
        ])
    )
    id_source_schema = ArrayType(
        StructType([
            StructField('id', StringType()),
            StructField('source', StringType()),
        ])
    )
    base = df.filter(f.size(f.col('accessions')) > 0).select(
        f.element_at(f.col('accessions'), 1).alias('uniprotId'),
        _safe_array_union(f.col('names'), f.col('synonyms')).alias('_name_syns'),
        _safe_array_union(f.col('symbolSynonyms')).alias('_symbol_syns'),
        _safe_array_union(
            _safe_array_union(f.col('names')),
            _safe_array_union(f.col('symbolSynonyms')),
        ).alias('_synonyms'),
        f.col('functions').alias('functionDescriptions'),
        f.col('dbXrefs'),
        f.col('accessions'),
        f.col('locations'),
    )

    # Handle dbXrefs: "ID SOURCE" → struct(id, source)
    base = base.withColumn(
        'dbXrefs',
        f.when(
            f.col('dbXrefs').isNotNull(),
            f.transform(
                f.col('dbXrefs'),
                lambda c: f.struct(
                    f.element_at(f.split(c, ' '), 1).alias('id'),
                    f.element_at(f.split(c, ' '), 2).alias('source'),
                ),
            ),
        ).cast(id_source_schema),
    )

    # Map locations → subcellularLocations using SSL ontology
    base = _map_uniprot_locations_to_ssl(base, ssl_df)

    # Build proteinIds (old accessions → uniprot_obsolete)
    base = base.withColumn(
        'proteinIds',
        f.when(
            f.col('accessions').isNotNull(),
            f.transform(
                f.col('accessions'),
                lambda c: f.struct(f.trim(c).alias('id'), f.lit('uniprot_obsolete').alias('source')),
            ),
        ).cast(id_source_schema),
    )

    # Add synonyms as LabelAndSource
    for src_col, dst_col in [
        ('_synonyms', 'synonyms'),
        ('_symbol_syns', 'symbolSynonyms'),
        ('_name_syns', 'nameSynonyms'),
    ]:
        base = base.withColumn(
            dst_col,
            f.when(
                f.col(src_col).isNotNull(),
                f.transform(f.col(src_col), lambda c: f.struct(c.alias('label'), f.lit('uniprot').alias('source'))),
            ).cast(label_source_schema),
        ).drop(src_col)

    return base.drop('accessions', 'locations')


def _map_uniprot_locations_to_ssl(df: DataFrame, ssl_df: DataFrame) -> DataFrame:
    """Map pre-parsed Uniprot location structs to SSL ontology terms."""
    first_words_regex = r'^([\w\s]+)'
    last_after_comma_regex = r'.*,\s([\w\s]+)'

    ssl_onto = (
        ssl_df.select(
            f.col('Subcellular location ID').alias('termSL'),
            f.col('Name').alias('ssl_match'),
            f.col('Category').alias('labelSL'),
        )
        if 'Subcellular location ID' in ssl_df.columns
        else ssl_df.select(
            f.col('termSL'),
            f.col('HPA_location').alias('ssl_match'),
            f.col('labelSL'),
        )
    )

    loc_df = (
        df
        .select('uniprotId', f.explode('locations').alias('loc_struct'))
        .select(
            'uniprotId',
            f.col('loc_struct.location').alias('location'),
            f.col('loc_struct.targetModifier').alias('targetModifier'),
        )
        .withColumn('loc1', f.trim(f.regexp_extract('location', first_words_regex, 0)))
        .withColumn('loc3', f.trim(f.regexp_extract('location', last_after_comma_regex, 1)))
        .withColumn(
            'ssl_match',
            f
            .when(f.col('loc1') != '', f.col('loc1'))  # noqa: PLC1901
            .when(f.col('loc3') != '', f.col('loc3'))  # noqa: PLC1901
            .otherwise(f.lit(None)),
        )
        .drop('loc1', 'loc3')
        .filter(f.col('location').isNotNull())
        .join(f.broadcast(ssl_onto), 'ssl_match', 'left_outer')
        .drop('ssl_match')
        .withColumn('source', f.lit('uniprot'))
        .select('uniprotId', f.struct('location', 'source', 'termSL', 'labelSL', 'targetModifier').alias('loc'))
        .groupBy('uniprotId')
        .agg(f.collect_list('loc').alias('subcellularLocations'))
    )

    return df.drop('locations').join(loc_df, 'uniprotId', 'left_outer')


# ===========================================================================
# Safety.scala → _build_safety
# ===========================================================================


def _build_safety(
    safety_df: DataFrame,
    ensg_lookup: DataFrame,
    diseases_df: DataFrame,
) -> DataFrame:
    """Build target safety liabilities.

    Args:
        safety_df: Pre-processed safety evidence parquet.
        ensg_lookup: ENSG ID lookup table (ensgId, name array).
        diseases_df: Disease index (id, obsoleteTerms).

    Returns:
        DataFrame with [id, safetyLiabilities[{event, eventId, effects, ...}]].
    """
    # Add missing ENSG IDs for entries that only have symbol (e.g. ToxCast)
    enriched = (
        safety_df
        .join(ensg_lookup, f.array_contains(f.col('name'), f.col('targetFromSourceId')), 'left_outer')
        .drop(*[c for c in ensg_lookup.columns if c != 'ensgId'])
        .withColumn('temp_id', f.coalesce(f.col('id'), f.col('ensgId')))
        .drop('id', 'ensgId')
        .withColumnRenamed('temp_id', 'id')
    )

    # Replace obsolete EFOs
    disease_mapping = diseases_df.select(
        f.col('id').alias('diseaseId'), f.explode(f.col('obsoleteTerms')).alias('obsoleteTerm')
    )

    enriched = (
        enriched
        .join(disease_mapping, enriched['eventId'] == disease_mapping['obsoleteTerm'], 'left_outer')
        .withColumn('eventId', f.coalesce(f.col('diseaseId'), f.col('eventId')))
        .drop('obsoleteTerm', 'diseaseId')
    )

    return (
        enriched
        .select(
            'id',
            f.struct(
                'event',
                'eventId',
                'effects',
                'biosamples',
                'datasource',
                'literature',
                'url',
                'studies',
            ).alias('safety'),
        )
        .groupBy('id')
        .agg(f.collect_set('safety').alias('safetyLiabilities'))
    )


# ===========================================================================
# Tep.scala → _build_tep
# ===========================================================================


def _build_tep(df: DataFrame) -> DataFrame:
    """Build TEP (Target Enabling Package) DataFrame.

    Args:
        df: Raw TEP JSON.

    Returns:
        DataFrame with [targetFromSourceId, description, therapeuticArea, url].
    """
    return df.select(
        f.trim(f.col('targetFromSourceId')).alias('targetFromSourceId'),
        f.trim(f.col('description')).alias('description'),
        f.trim(f.col('therapeuticArea')).alias('therapeuticArea'),
        f.trim(f.col('url')).alias('url'),
    )


# ===========================================================================
# Assembly helpers (mirrors Target.scala private methods)
# ===========================================================================


def _merge_hgnc_ensembl(hgnc_df: DataFrame, ensembl_df: DataFrame) -> DataFrame:
    """Merge HGNC and Ensembl; HGNC fields take precedence.

    Args:
        hgnc_df: HGNC DataFrame from _build_hgnc.
        ensembl_df: Ensembl DataFrame from _build_ensembl.

    Returns:
        Merged DataFrame.
    """
    e = ensembl_df.withColumnRenamed('approvedName', '_an').withColumnRenamed('approvedSymbol', '_as')

    return (
        e
        .join(hgnc_df, e['id'] == hgnc_df['ensemblId'], 'left_outer')
        .withColumn('approvedName', f.coalesce(f.col('approvedName'), f.col('_an'), f.lit('')))
        .withColumn('approvedSymbol', f.coalesce(f.col('approvedSymbol'), f.col('_as'), f.col('id')))
        .drop('_an', '_as', 'ensemblId')
    )


def _add_ensembl_ids_to_uniprot(hgnc_df: DataFrame, uniprot_df: DataFrame) -> DataFrame:
    """Group Uniprot entries by Ensembl ID using HGNC as the mapping key.

    Args:
        hgnc_df: HGNC DataFrame with [ensemblId, uniprotIds, ...].
        uniprot_df: Uniprot DataFrame with [uniprotId, ...].

    Returns:
        Uniprot data grouped by Ensembl id (column renamed to 'id').
    """
    id_source_schema = ArrayType(
        StructType([
            StructField('id', StringType()),
            StructField('source', StringType()),
        ])
    )

    mapping = hgnc_df.select('ensemblId', f.explode('uniprotIds').alias('uniprotId')).withColumn(
        'uniprotProteinId',
        f.struct(
            f.col('uniprotId').alias('id'),
            f.lit('uniprot_obsolete').alias('source'),
        ),
    )

    return (
        mapping
        .join(uniprot_df, 'uniprotId')
        .groupBy('ensemblId')
        .agg(
            f.flatten(f.collect_set('synonyms')).alias('synonyms'),
            f.flatten(f.collect_set('symbolSynonyms')).alias('symbolSynonyms'),
            f.flatten(f.collect_set('nameSynonyms')).alias('nameSynonyms'),
            f.flatten(f.collect_set('functionDescriptions')).alias('functionDescriptions'),
            f.flatten(f.collect_set('proteinIds')).alias('proteinIds'),
            f.flatten(f.collect_set('subcellularLocations')).alias('subcellularLocations'),
            f.flatten(f.collect_set('dbXrefs')).alias('dbXrefs'),
            f.collect_set('uniprotProteinId').alias('uniprotProteinId'),
            f.flatten(f.collect_set('targetClass')).alias('targetClass'),
        )
        .withColumn(
            'targetClass',
            f.when(f.size('targetClass') < 1, f.lit(None)).otherwise(f.col('targetClass')),
        )
        .withColumnRenamed('ensemblId', 'id')
        .withColumn(
            'proteinIds',
            _safe_array_union(f.col('proteinIds'), f.col('uniprotProteinId').cast(id_source_schema)),
        )
        .drop('uniprotId', 'uniprotProteinId')
    )


def _add_protein_classification_to_uniprot(
    uniprot_df: DataFrame,
    protein_class_df: DataFrame,
) -> DataFrame:
    """Add protein classification (targetClass) to Uniprot entries."""
    pc_with_uniprot = (
        uniprot_df
        .select(
            'uniprotId',
            f.explode(_safe_array_union(f.array(f.col('uniprotId')), f.col('proteinIds.id'))).alias('pid'),
        )
        .withColumn('pid', f.trim('pid'))
        .join(protein_class_df, f.col('pid') == protein_class_df['accession'], 'left_outer')
        .drop('accession')
        .groupBy('uniprotId')
        .agg(f.flatten(f.collect_set('targetClass')).alias('targetClass'))
    )

    return uniprot_df.join(pc_with_uniprot, 'uniprotId', 'left_outer')


def _generate_ensg_lookup(df: DataFrame) -> DataFrame:
    """Generate ENSG → symbol / uniprot / HGNC lookup table."""
    safe_union = _safe_array_union

    return (
        df
        .select(
            'id',
            f.col('proteinIds.id').alias('pid'),
            f.array(f.col('approvedSymbol')).alias('as_arr'),
            f.filter(f.col('synonyms'), lambda c: c.getField('source') == 'uniprot').alias('uniprot'),
            f.filter(f.col('synonyms'), lambda c: c.getField('source') == 'HGNC').alias('HGNC_syns'),
            f.array_distinct(
                safe_union(
                    f.col('proteinIds.id'),
                    f.col('symbolSynonyms.label'),
                    f.col('obsoleteSymbols.label') if 'obsoleteSymbols' in df.columns else f.array(),
                    f.array(f.col('approvedSymbol')),
                )
            ).alias('symbols'),
        )
        .select(
            'id',
            f.flatten(f.array(f.col('pid'), f.col('as_arr'))).alias('name'),
            f.col('uniprot.label').alias('uniprot'),
            f.col('HGNC_syns.label').alias('HGNC'),
            'symbols',
        )
        .select(
            f.col('id').alias('ensgId'),
            f.col('name'),
            'uniprot',
            'HGNC',
            'symbols',
        )
    )


def _build_gene_essentiality_output(
    essentiality_df: DataFrame,
    ensg_lookup: DataFrame,
) -> DataFrame:
    """Build gene essentiality output mapped to ENSG IDs.

    Ports addGeneEssentiality from Target.scala.

    Args:
        essentiality_df: Gene essentiality intermediate with targetSymbol column.
        ensg_lookup: ENSG→symbol lookup from _generate_ensg_lookup.

    Returns:
        DataFrame with [id, geneEssentiality] where geneEssentiality is a list
        of essentiality structs per ENSG ID.
    """
    lookup = (
        ensg_lookup
        .select('ensgId', 'name')
        .withColumn('approvedTarget', f.explode('name'))
        .drop('name')
        .orderBy('approvedTarget')
    )
    essentiality_cols = [c for c in essentiality_df.columns if c != 'targetSymbol']
    essentiality_with_ensg = (
        essentiality_df
        .join(lookup, lookup['approvedTarget'] == essentiality_df['targetSymbol'], 'inner')
        .drop(*[c for c in lookup.columns if c != 'ensgId'])
        .drop('targetSymbol')
    )
    return (
        essentiality_with_ensg
        .select(
            f.col('ensgId').alias('id'),
            f.struct(*[f.col(c) for c in essentiality_cols]).alias('ts'),
        )
        .groupBy('id')
        .agg(f.collect_list('ts').alias('geneEssentiality'))
    )


def _add_tep(df: DataFrame, tep_df: DataFrame, lookup: DataFrame) -> DataFrame:
    """Join TEP data using symbol→ENSG lookup."""
    lut = (
        lookup.select(f.col('ensgId').alias('id'), 'symbols').withColumn('symbol', f.explode('symbols')).drop('symbols')
    )

    tep_fields = ['targetFromSourceId', 'description', 'therapeuticArea', 'url']
    tep_with_id = (
        tep_df
        .join(lut, lut['symbol'] == tep_df['targetFromSourceId'])
        .withColumn('tep', f.struct(*tep_fields))
        .select('id', 'tep')
    )

    return df.join(tep_with_id, 'id', 'left_outer')


def _filter_and_sort_protein_ids(df: DataFrame) -> DataFrame:
    """Deduplicate proteinIds and sort by source preference."""

    def _sort_protein_ids(accessions):
        """UDF: deduplicate and sort by source priority."""
        if not accessions:
            return []
        seen = {}
        delimiter = '|||'
        for entry in accessions:
            parts = entry.split(delimiter, 1)
            if len(parts) < 2:
                continue
            pid, src = parts[0].strip(), parts[1]
            if pid not in seen:
                seen[pid] = src

        def _priority(item):
            pid, src = item
            if 'swiss' in src:
                return (1, pid)
            if 'trembl' in src:
                return (2, pid)
            if src.endswith('uniprot'):
                return (3, pid)
            if 'ensembl' in src:
                return (4, pid)
            return (5, pid)

        sorted_items = sorted(seen.items(), key=_priority)
        return [f'{pid}|||{src}' for pid, src in sorted_items]

    from pyspark.sql.functions import udf
    from pyspark.sql.types import ArrayType, StringType

    sort_udf = udf(_sort_protein_ids, ArrayType(StringType()))

    delimiter = '|||'
    ensembl_id = 'eid'
    protein_id = 'proteinIds'

    deduped = (
        df
        .select(f.col('id').alias(ensembl_id), f.explode(f.col(protein_id)).alias('p'))
        .select(ensembl_id, f.col('p.*'))
        .withColumn('arr', f.array_join(f.array(f.trim(f.col('id')), f.col('source')), delimiter))
        .groupBy(ensembl_id)
        .agg(f.collect_list('arr').alias('accessions'))
        .select(f.col(ensembl_id), sort_udf(f.col('accessions')).alias('accessions'))
        .select(f.col(ensembl_id), f.explode('accessions').alias('accessions'))
        .withColumn('id_and_source', f.split('accessions', r'\|\|\|'))
        .select(
            f.col(ensembl_id),
            f.element_at(f.col('id_and_source'), 1).alias('id'),
            f.element_at(f.col('id_and_source'), 2).alias('source'),
        )
        .select(f.col(ensembl_id), f.struct('id', 'source').alias(protein_id))
        .groupBy(ensembl_id)
        .agg(f.collect_list(protein_id).alias(protein_id))
        .withColumnRenamed(ensembl_id, 'id')
    )

    return df.drop(protein_id).join(deduped, 'id', 'left_outer')


def _remove_redundant_xrefs(df: DataFrame) -> DataFrame:
    """Remove GO and Ensembl entries from dbXrefs."""
    cols = [c for c in df.columns if c != 'dbXrefs']
    cols.append("filter(dbXrefs, s -> s.source != 'GO' and s.source != 'Ensembl') as dbXrefs")
    return df.selectExpr(*cols)


def _add_chemical_probes(df: DataFrame, cp_df: DataFrame, lookup: DataFrame) -> DataFrame:
    """Join chemical probes using symbol lookup."""
    cp_with_id = cp_df.join(lookup, f.array_contains(f.col('name'), f.col('targetFromSourceId'))).drop(*[
        c for c in lookup.columns if c != 'ensgId'
    ])

    cp_cols = [c for c in cp_df.columns if c != 'ensgId']
    cp_grouped = (
        cp_with_id
        .select(
            f.col('ensgId').alias('id'),
            f.struct(*cp_cols).alias('probe'),
        )
        .groupBy('id')
        .agg(f.collect_list('probe').alias('chemicalProbes'))
    )

    return df.join(f.broadcast(cp_grouped), 'id', 'left_outer')


def _add_orthologues(df: DataFrame, orthologs: DataFrame) -> DataFrame:
    """Add homologues to target DataFrame."""
    gene_symbols = df.select('id', 'approvedSymbol').cache()

    paralog_symbols = gene_symbols.withColumnRenamed('approvedSymbol', 'paralogGeneSymbol').withColumnRenamed(
        'id', 'paralogId'
    )

    homo_df = (
        orthologs
        .join(f.broadcast(gene_symbols), 'id')
        .join(
            f.broadcast(paralog_symbols),
            f.col('paralogId') == f.col('targetGeneId'),
            'left_outer',
        )
        .withColumn(
            'targetGeneSymbol',
            f.coalesce(f.col('paralogGeneSymbol'), f.col('targetGeneSymbol'), f.col('approvedSymbol')),
        )
        .drop('approvedSymbol', 'paralogGeneSymbol', 'paralogId')
    )

    homo_cols = [c for c in homo_df.columns if c != 'id']
    grouped = (
        homo_df
        .select('id', f.struct(*homo_cols).alias('homologues'))
        .groupBy('id')
        .agg(f.collect_list('homologues').alias('homologues'))
        # Sort by priority (ascending = closest species first)
        .withColumn('homologues', f.expr('array_sort(homologues, (x, y) -> x.priority - y.priority)'))
    )

    return df.join(grouped, 'id', 'left_outer').drop('humanGeneId')


def _add_tractability(df: DataFrame, tractability: DataFrame) -> DataFrame:
    """Join tractability data."""
    return df.join(tractability, f.col('ensemblGeneId') == f.col('id'), 'left_outer').drop('ensemblGeneId')


def _add_ncbi_synonyms(df: DataFrame, ncbi: DataFrame) -> DataFrame:
    """Add NCBI Entrez synonyms."""
    ncbi_renamed = (
        ncbi
        .withColumnRenamed('synonyms', '_s')
        .withColumnRenamed('nameSynonyms', '_ns')
        .withColumnRenamed('symbolSynonyms', '_ss')
    )

    return (
        df
        .join(ncbi_renamed, 'id', 'left_outer')
        .withColumn('symbolSynonyms', _safe_array_union(f.col('symbolSynonyms'), f.col('_ss')))
        .withColumn('nameSynonyms', _safe_array_union(f.col('nameSynonyms'), f.col('_ns')))
        .withColumn('synonyms', _safe_array_union(f.col('synonyms'), f.col('_s'), f.col('_ns'), f.col('_ss')))
        .drop('_s', '_ss', '_ns')
    )


def _add_target_safety(
    df: DataFrame,
    safety_raw: DataFrame,
    ensg_lookup: DataFrame,
    diseases_df: DataFrame,
) -> DataFrame:
    """Add target safety liabilities."""
    safety = _build_safety(safety_raw, ensg_lookup, diseases_df)
    return df.join(safety, 'id', 'left_outer')


def _add_reactome(df: DataFrame, reactome: DataFrame) -> DataFrame:
    """Add Reactome pathways."""
    return df.join(reactome, 'id', 'left_outer')


def _remove_duplicated_synonyms(df: DataFrame) -> DataFrame:
    """Deduplicate synonym arrays."""
    for col_name in ['synonyms', 'symbolSynonyms', 'nameSynonyms', 'obsoleteNames', 'obsoleteSymbols']:
        if col_name in df.columns:
            df = df.withColumn(col_name, f.array_distinct(f.col(col_name)))
    return df


def _add_tss(df: DataFrame) -> DataFrame:
    """Add transcription start site column."""
    return df.withColumn(
        'tss',
        f.when(f.col('canonicalTranscript.strand') == '+', f.col('canonicalTranscript.start')).when(
            f.col('canonicalTranscript.strand') == '-', f.col('canonicalTranscript.end')
        ),
    )
