"""Evidence parser for Gene2Phenotype's disease panels."""

from collections import OrderedDict
from typing import Any

from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.common.ontology import add_efo_mapping
from pts.pyspark.common.session import Session

G2P_mutationCsq2functionalCsq = OrderedDict([
    ('uncertain', 'SO_0002220'),
    ('cis-regulatory or promotor mutation', 'SO_0001566'),
    ('5_prime or 3_prime UTR mutation', 'SO_0001622'),
    ('increased gene product level', 'SO_0002315'),
    ('decreased gene product level', 'SO_0002316'),
    ('altered gene product structure', 'SO_0002318'),
    ('absent gene product', 'SO_0002317'),
])

G2P_INPUT_SCHEMA = (
    t.StructType()
    .add('g2p id', t.StringType())
    .add('gene symbol', t.StringType())
    .add('gene mim', t.IntegerType())
    .add('hgnc id', t.IntegerType())
    .add('previous gene symbols', t.StringType())
    .add('disease name', t.StringType())
    .add('disease mim', t.StringType())
    .add('disease MONDO', t.StringType())
    .add('allelic requirement', t.StringType())
    .add('cross cutting modifier', t.StringType())
    .add('confidence', t.StringType())
    .add('variant consequence', t.StringType())
    .add('variant types', t.StringType())
    .add('molecular mechanism', t.StringType())
    .add('molecular mechanism support', t.StringType())
    .add('molecular mechanism categorisation', t.StringType())
    .add('molecular mechanism evidence', t.StringType())
    .add('phenotypes', t.StringType())
    .add('publications', t.StringType())
    .add('additional mined publications', t.StringType())
    .add('panel', t.StringType())
    .add('comments', t.StringType())
    .add('date of last review', t.StringType())
    .add('review', t.StringType())
)


def gene2phenotype(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    spark = Session(app_name='gene2phenotype', properties=properties)

    # Pop OnToma LUT paths from the sources dict
    ontoma_disease_label_lut = source.pop('ontoma_disease_label_lut')
    ontoma_disease_id_lut = source.pop('ontoma_disease_id_lut')

    logger.info(f'load data from {source}')
    g2p_panel_files = list(source.values())
    gene2phenotype_df = spark.load_data(g2p_panel_files, format='csv', header=True, schema=G2P_INPUT_SCHEMA)

    evidence_df = process_gene2phenotype(gene2phenotype_df)
    logger.info('map gene2phenotype disease labels')
    mapped_evidence_df = add_efo_mapping(
        spark=spark.spark,
        evidence_df=evidence_df,
        disease_label_lut_path=ontoma_disease_label_lut,
        disease_id_lut_path=ontoma_disease_id_lut,
    )

    logger.info(f'write gene2phenotype evidence strings to {destination}')
    mapped_evidence_df.write.parquet(destination, mode='overwrite')


def process_gene2phenotype(gene2phenotype_df: DataFrame) -> DataFrame:
    """Format raw G2P data into evidence strings."""
    return gene2phenotype_df.select(
        # Split pubmed IDs to list when not null:
        f.when(f.col('publications').isNotNull(), f.split(f.col('publications'), ';')).alias('literature'),
        # Renaming a few columns:
        f.col('gene symbol').alias('targetFromSourceId'),
        f.col('panel').alias('studyId'),
        f.col('confidence'),
        # Parsing allelic requirements:
        f.when(
            f.col('allelic requirement').isNotNull(),
            f.array(f.col('allelic requirement')),
        ).alias('allelicRequirements'),
        # Parsing disease from source identifier:
        f.when(f.col('disease MONDO').isNotNull(), f.col('disease MONDO'))
        .when(
            (~f.col('disease mim').contains('No disease mim')) & (f.col('disease mim').isNotNull()),
            f.concat(f.lit('OMIM:'), f.col('disease mim')),
        )
        .otherwise(f.lit(None))
        .alias('diseaseFromSourceId'),
        # Cleaning disease names:
        f.regexp_replace(f.col('disease name'), r'.+-related ', '').alias('diseaseFromSource'),
        # Map functional consequences:
        parse_functional_consequence(G2P_mutationCsq2functionalCsq)('variant consequence').alias(
            'variantFunctionalConsequenceId'
        ),
        # Adding constant columns:
        f.lit('gene2phenotype').alias('datasourceId'),
        f.lit('genetic_literature').alias('datatypeId'),
    )


def parse_functional_consequence(mapping: dict[str, str]) -> f.udf:
    """Map semicolon separated list of consequence terms to SO codes."""
    order = list(mapping.values())

    def __translate(col):
        if col is None:
            return None

        # Split the string and clean up any whitespace
        consequence_terms = [term.strip() for term in col.split(';') if term.strip()]
        if not consequence_terms:
            return None

        # Get mapped SO terms that exist in the mapping
        mapped_terms = [
            mapping.get(consequence_term) for consequence_term in consequence_terms if consequence_term in mapping
        ]
        if not mapped_terms:
            return None

        # Return the element of mapped_terms that has the highest rank in the original mapping
        try:
            return max(mapped_terms, key=lambda x: order.index(x))
        except ValueError:
            return None

    return f.udf(__translate, t.StringType())
