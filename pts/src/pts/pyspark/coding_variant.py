"""Parser to generate view on coding variants and their effects."""

from typing import Any

from loguru import logger
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.window import Window

from pts.pyspark.common import Session


def coding_variant(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Generate view on coding variants with their functional context."""
    session = Session(app_name='coding_variant_view', properties=properties)

    # Extracting variants with amino acid change:
    variants_with_amino_acid_effect = process_variants(session.spark, source['variant'])

    # Getting target/uniprot map:
    target_uniprot_map = process_target_index(session.spark, source['target'])

    # Reading disease dataset:
    diseases = process_diseases(session.spark, source['disease'])

    # Extracting evidence:
    evidence_dataset = process_evidence(
        session.spark,
        diseases,
        source['evidence_eva'],
        source['evidence_eva_somatic'],
        source['evidence_uniprot_variant'],
    )

    # Extracting GWAS associations:
    gwas_evidence = process_gwas_associations(
        session.spark,
        variants_with_amino_acid_effect,
        diseases,
        source['evidence_gwas_credible_set'],
        source['credible_set'],
    )

    # Getting molQTL datasets:
    molqtl_credsets = process_qtls(session.spark, source['credible_set'])

    # Extracting pharmacogenetics evidence:
    pharmacogenetics = process_pharmacogenetics(session.spark, source['pharmacogenetics'])

    # Pooling all evidence together:
    variant_evidence = (
        evidence_dataset
        .unionByName(gwas_evidence, allowMissingColumns=True)
        .unionByName(pharmacogenetics, allowMissingColumns=True)
        .unionByName(molqtl_credsets, allowMissingColumns=True)
    )

    logger.info(f'pooled {variant_evidence.count()} evidences from all sources')
    logger.info('starting main process')

    # this window specification is used to group diseases for each variant and accumulate scores:
    wspec = (
        Window
        .partitionBy('variantId', 'diseaseId')
        .orderBy(f.col('score').desc())  # ty:ignore[missing-argument]
        .rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    )
    wspec_datasource = (
        Window
        .partitionBy('variantId', 'datasourceId')
        .orderBy(f.col('score').desc())  # ty:ignore[missing-argument]
        .rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    )

    (
        variant_evidence
        .withColumns({
            'harmonic_sum': calculate_harmonic_sum(f.collect_list(f.col('score')).over(wspec)),
            'datasourceCount': f.size(f.collect_set('evidenceId').over(wspec_datasource)),
            'datasourceNiceName': f
            .when(f.col('datasourceId') == 'eva', f.lit('ClinVar'))
            .when(f.col('datasourceId') == 'eva_somatic', f.lit('ClinVar-somatic'))
            .when(f.col('datasourceId') == 'uniprot_variants', f.lit('UniProt Variants'))
            .when(f.col('datasourceId') == 'mol_qtl', f.lit('molQTL'))
            .when(f.col('datasourceId') == 'gwas_credible_sets', f.lit('GWAS credible sets'))
            .otherwise(f.col('datasourceId')),
        })
        .groupby('variantId')
        .agg(
            # Collecting all therapeutic areas:
            f.array_distinct(f.flatten(f.collect_list('therapeuticAreas'))).alias('therapeuticAreas'),
            # Collecting disease ids and scores:
            f.collect_set(f.struct('diseaseId', 'harmonic_sum')).alias('diseases'),
            # Collecting evidence sources:
            f.collect_set(
                f.struct(
                    'datasourceCount',
                    'datasourceId',
                    'datasourceNiceName',
                )
            ).alias('datasources'),
            # Collecting if evidence is nullable:
            f.collect_set('zeroEvidence').alias('zeroEvidence'),
        )
        # Drop empty disease objects:
        .withColumn('diseases', f.filter('diseases', lambda disease: disease.diseaseId.isNotNull()))
        # Order disease objects by the harmonic sum and extract disease id:
        .withColumn(
            'diseases', f.transform(order_struct_list('diseases', 'harmonic_sum'), lambda disease: disease.diseaseId)
        )
        .join(variants_with_amino_acid_effect, on='variantId', how='right')
        .filter(
            # Dropping varians without any evidence:
            f.col('dataSources').isNotNull()  # ty:ignore[missing-argument]
        )
        # Removing source annotation if variant has only zero evidence:
        .withColumn(
            'datasources',
            f.when(f.array_contains(f.col('zeroEvidence'), False), f.col('datasources')).otherwise(f.lit([])),
        )
        .join(target_uniprot_map, on='targetId', how='left')
        .withColumn(
            'uniprotAccessions',
            f.when(
                f.col('uniprotAccessions').isNotNull() & f.col('refUniprotAccessions').isNotNull(),  # ty:ignore[missing-argument]
                f.array_intersect(f.col('refUniprotAccessions'), f.col('uniprotAccessions')),
            ).otherwise(f.coalesce(f.col('uniprotAccessions'), f.col('refUniprotAccessions'))),
        )
        .drop('zeroEvidence', 'refUniprotAccessions')
        .write.mode('overwrite')
        .parquet(destination)
    )


def process_target_index(spark: SparkSession, source: str) -> DataFrame:
    """Read target index and extract relevant columns.

    Args:
        spark (SparkSession): Spark session.
        source (str): Path to the target dataset.

    Returns:
        DataFrame: DataFrame containing targetId and uniprotAccessions.
    """
    d = (
        spark.read
        .parquet(source)
        .select(
            f.col('id').alias('targetId'),
            f.transform(
                f.filter('proteinIds', lambda ref: ref.source == 'uniprot_swissprot'), lambda ref: ref.id
            ).alias('refUniprotAccessions'),
        )
        .filter(f.size('refUniprotAccessions') != 0)
    )
    logger.info(f'processed {d.count()} targets')
    return d


def calculate_harmonic_sum(input_array: Column) -> Column:
    """Calculate the harmonic sum of an array.

    Args:
        input_array (Column): input array of doubles

    Returns:
        Column: column of harmonic sums

    Examples:
        >>> from pyspark.sql import Row
        >>> df = spark.createDataFrame([
        ...     Row([0.3, 0.8, 1.0]),
        ...     Row([0.7, 0.2, 0.9]),
        ...     ], ["input_array"]
        ... )
        >>> df.select("*", f.round(calculate_harmonic_sum(f.col("input_array")), 2).alias("harmonic_sum")).show()
        +---------------+------------+
        |    input_array|harmonic_sum|
        +---------------+------------+
        |[0.3, 0.8, 1.0]|        0.75|
        |[0.7, 0.2, 0.9]|        0.67|
        +---------------+------------+
        <BLANKLINE>
    """
    # Remove null values from the input array
    filtered_array = f.filter(input_array, lambda x: x.isNotNull())

    return f.when(
        f.size(filtered_array) > 0,
        f.aggregate(
            f.arrays_zip(
                f.sort_array(filtered_array, False).alias('score'),
                f.sequence(f.lit(1), f.size(filtered_array)).alias('pos'),
            ),
            f.lit(0.0),
            lambda acc, x: acc + x['score'] / f.pow(x['pos'], 2) / f.lit(sum(1 / ((i + 1) ** 2) for i in range(1000))),
        ),
    ).otherwise(f.lit(None))


def parse_amino_acid_change(aa: Column) -> tuple[Column, Column, Column]:
    """Extract reference, alternate amino acids and position from the amino acid change string.

    Args:
        aa (Column): Column containing the amino acid change string.

    Returns:
        tuple[Column, Column, Column]: Tuple containing the reference amino acid,
        alternate amino acid, and position.
    """
    ref_aa = f.regexp_extract(aa, r'^([-\*A-Za-z]+)', 1).alias('referenceAminoAcid')
    alt_aa = f.regexp_extract(aa, r'([-\*A-Zaz+]+$)', 1).alias('alternateAminoAcid')
    position = f.regexp_extract(aa, r'(\d+)', 1).cast(t.IntegerType()).alias('aminoAcidPosition')

    return (ref_aa, position, alt_aa)


def extract_method_value(variant_effect: Column, method: str) -> Column:
    """Extract the value of a specific method from the variant effect.

    Args:
        variant_effect (Column): Column containing the variant effect.
        method (str): The method to extract the value for.

    Returns:
        Column: A struct containing the value and method name.
    """
    # Extract prediction:
    method_predictions = f.filter(variant_effect, lambda ve: ve.method == method)

    # Exract the maximum value:
    value = f.when(
        f.size(method_predictions) > 0, f.array_max(f.transform(method_predictions, lambda mp: mp.normalisedScore))
    ).otherwise(f.lit(None))

    return f.struct(value.alias('value'), f.lit(method).alias('method'))


def parse_variant_effect(variant_effect: Column) -> Column:
    """Parse the variant effect to extract AlphaMissense and VEP effects.

    Args:
        variant_effect (Column): Column containing the variant effect.

    Returns:
        Column: A struct containing the AlphaMissense and VEP effects.
    """
    # we need to extract variant effect separately:
    a_missense = extract_method_value(variant_effect, 'AlphaMissense')
    vep = extract_method_value(variant_effect, 'VEP')

    # Preferentially pick a_missense:
    return f.when(a_missense.getItem('value').isNotNull(), a_missense).otherwise(vep)  # ty:ignore[missing-argument]


def process_variants(spark: SparkSession, source: str) -> DataFrame:
    """Get all protein coding variants with their consequences.

    Args:
        spark (SparkSession): Spark session.
        source (str): Path to the variant dataset.

    Returns:
        DataFrame: DataFrame containing variants with amino acid effects.
    """
    d = (
        spark.read
        .parquet(source)
        # Filter for certain variant consequence types:
        .withColumn('variantEffect', f.filter('variantEffect', lambda ve: ve.method == 'AlphaMissense'))
        # Exploding transcript consequences to extract amino acid consequences:
        .withColumn('exploded_consequences', f.explode('transcriptConsequences'))
        # Drop all consequences without amino acid change:
        .filter(f.col('exploded_consequences.aminoAcidChange').isNotNull())  # ty:ignore[missing-argument]
        .select(
            'variantId',
            # Filter for AlphaMissense consequence or return None:
            f
            .when(f.size(f.col('variantEffect')) > 0, f.col('variantEffect')[0].score)
            .otherwise(f.lit(None))
            .alias('variantEffect'),
            'exploded_consequences.targetId',
            'exploded_consequences.variantFunctionalConsequenceIds',
            # DO NOT explode ambigious uniprot accessions:
            f.expr(
                'array_sort(exploded_consequences.uniprotAccessions, '
                '(left, right) -> CASE '
                'WHEN length(left) <= length(right) THEN -1 '
                'ELSE 1 END)'
            ).alias('uniprotAccessions'),
            # Extracting amino acids and positions:
            *parse_amino_acid_change(f.col('exploded_consequences.aminoAcidChange')),
        )
    )
    logger.info(f'processed {d.count()} variants')
    return d


def process_evidence(
    spark: SparkSession,
    diseases: DataFrame,
    source_evidence_eva: str,
    source_evidence_eva_somatic: str,
    source_evidence_uniprot_variant: str,
) -> DataFrame:
    """Get all evidence supported by variants.

    Args:
        spark (SparkSession): Spark session.
        diseases (DataFrame): DataFrame containing diseases.
        source_evidence_eva (str): Path to the eva evidence dataset.
        source_evidence_eva_somatic (str): Path to the eva somatic evidence dataset.
        source_evidence_uniprot_variant (str): Path to the uniprot variants dataset.

    Returns:
        DataFrame: DataFrame containing evidence supported by variants.
    """
    # Extracting variant effect:
    evidence_columns: list[str | Column] = [
        'datasourceId',
        'variantId',
        'score',
        f.col('id').alias('evidenceId'),
        f.when(f.col('score') == 0, f.lit(True)).otherwise(f.lit(False)).alias('zeroEvidence'),
        f.when(f.col('score') == 0, f.lit(None)).otherwise(f.col('diseaseId')).alias('diseaseId'),
    ]

    # Reading evidence dataset:
    d = (
        spark.read
        .parquet(source_evidence_eva, source_evidence_eva_somatic, source_evidence_uniprot_variant)
        # Filtering for evidence supported by variants:
        .filter(f.col('variantId').isNotNull())  # ty:ignore[missing-argument]
        .select(*evidence_columns)
        # Adding therapeutic areas:
        .join(diseases, on='diseaseId', how='left')
    )
    logger.info(f'processed {d.count()} evidences')
    return d


def order_struct_list(column: str, field: str, ascending: bool = True) -> Column:
    """Orders a list column of structs based on a specific field in the struct.

    Args:
        column (str): The name of column containing a list of structs.
        field (str): The field in the struct to sort by.
        ascending (bool): Whether to sort in ascending order. Defaults to True.

    Returns:
        Column: A column with the ordered list of structs.
    """
    return f.expr(f"""
        array_sort({column}, (left, right) ->
            CASE
                WHEN left.{field} < right.{field} THEN {1 if ascending else -1}
                WHEN left.{field} > right.{field} THEN {-1 if ascending else 1}
                ELSE 0
            END
        )
    """)


def process_pharmacogenetics(spark: SparkSession, source: str) -> DataFrame:
    """Get all pharmacogenetics evidence.

    Args:
        spark (SparkSession): Spark session.
        source (str): Path to the pharmacogenetics dataset.

    Returns:
        DataFrame: DataFrame containing pharmacogenetics evidence.
    """
    d = (
        spark.read
        .parquet(source)
        .select(
            'variantId',
            'datasourceId',
            f.monotonically_increasing_id().cast(t.StringType()).alias('evidenceId'),
            f.lit(False).alias('zeroEvidence'),
        )
        .filter(f.col('variantId').isNotNull())  # ty:ignore[missing-argument]
    )
    logger.info(f'processed {d.count()} pharmacogenetics evidences')
    return d


def process_qtls(spark: SparkSession, source: str) -> DataFrame:
    """Get all QTL evidence.

    Args:
        spark (SparkSession): Spark session.
        source (str): Path to the credible_set dataset.

    Returns:
        DataFrame: DataFrame containing QTL evidence.
    """
    d = (
        # Reading credible sets and explode:
        spark.read
        .parquet(source)
        .withColumn('col', f.explode('locus'))
        # Selecting only qtl evidence:
        .filter(f.col('studyType') != 'gwas')
        .select(
            f.col('col.variantId').alias('VariantId'),
            f.lit('mol_qtl').alias('datasourceId'),
            f.monotonically_increasing_id().cast(t.StringType()).alias('evidenceId'),
            f.lit(False).alias('zeroEvidence'),
        )
    )
    logger.info(f'processed {d.count()} qtl evidences')
    return d


def process_gwas_associations(
    spark: SparkSession,
    variants_with_amino_acid_effect: DataFrame,
    diseases: DataFrame,
    source_evidence_gwas_credible_set: str,
    source_credible_set: str,
) -> DataFrame:
    """Get all GWAS associations with credible sets.

    Args:
        spark (SparkSession): Spark session.
        variants_with_amino_acid_effect (DataFrame): DataFrame containing variants with amino acid
            effects.
        diseases (DataFrame): DataFrame containing diseases.
        source_evidence_gwas_credible_set (str): Path to the gwas credible set evidence dataset.
        source_credible_set (str): Path to the credible set dataset.

    Returns:
        DataFrame: DataFrame containing GWAS associations with credible sets.
    """
    d = (
        # Reading evidence:
        spark.read
        .parquet(source_evidence_gwas_credible_set)
        # Group by studyLocusId and select to strongest evidence:
        .withColumn(
            'maxScore',
            f.max('score').over(
                Window.partitionBy('studyLocusId', 'diseaseId').rowsBetween(
                    Window.unboundedPreceding, Window.unboundedFollowing
                )
            ),
        )
        # Each StudyLocusId/diseaseId pair should have only one max score:
        .filter(f.col('score') == f.col('maxScore'))
        # joining with credible sets:
        .join(spark.read.parquet(source_credible_set), on='studyLocusId', how='inner')
        # Exploding credible sets:
        .withColumn('locus', f.explode('locus'))
        # Join with variants to get correct gene:
        .select(
            'locus.variantId',
            'diseaseId',
            'score',
            'datasourceId',
            f.col('studyLocusId').alias('evidenceId'),
            f.when(f.col('score') == 0, f.lit(True)).otherwise(f.lit(False)).alias('zeroEvidence'),
        )
        # Subsetting to only those variants that are in the variants_with_amino_acid_effect:
        .join(variants_with_amino_acid_effect.select('variantId'), how='inner', on=['variantId'])
        # Adding therapeutic areas:
        .join(diseases, on='diseaseId', how='left')
    )
    logger.info(f'processed {d.count()} gwas credible set association evidences')
    return d


def process_diseases(spark: SparkSession, source) -> DataFrame:
    """Get all diseases.

    Args:
        spark (SparkSession): Spark session.
        source (str): Path to the disease dataset.

    Returns:
        DataFrame: DataFrame containing disease IDs and therapeutic areas.
    """
    d = spark.read.parquet(source).select(f.col('id').alias('diseaseId'), f.col('therapeuticAreas'))
    logger.info(f'processed {d.count()} diseases')
    return d
