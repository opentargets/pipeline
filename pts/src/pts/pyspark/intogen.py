"""This script generates disease/target evidence based on INTOGEN data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.common.session import Session

if TYPE_CHECKING:
    from pyspark.sql import Column

DOI_TO_PMID_MAPPING = {
    '10.1038/ng.2529': '23334666',
    '10.1038/s41588-023-01321-1': '36928603',
    '10.1038/ng.3940': '28825729',
    '10.1101/162784': '32025007',
    '10.1172/JCI128227': '31483290',
    '10.1038/s41591-019-0582-4': '31570822',
    '10.1002/hep.27198': '24798001',
    '10.1038/ng.3520': '26928227',
    '10.1038/s41588-018-0078-z': '29610475',
}


def parse_source(source_column: Column) -> Column:
    processed = f.when(source_column.startswith('WEB'), f.lit(None)).otherwise(f.trim(f.split(source_column, ':')[1]))

    map_udf = f.udf(lambda v: DOI_TO_PMID_MAPPING.get(v, v), t.StringType())
    mapped = map_udf(processed)

    return f.when(mapped.isNotNull(), f.array(mapped))


def generate_evidence(genes: DataFrame, cohorts: DataFrame, disease_mapping: DataFrame) -> DataFrame:
    """Generate INTOGEN-derived evidence.

    The function transforms the raw INTOGEN `genes` table and cohort metadata into a Spark
    DataFrame that matches the project's evidence schema fragments. It:
      - joins gene-level driver calls to cohort records,
      - maps gene ROLE ('Act' / 'LoF') to canonical consequence IDs ('SO_0002053' / 'SO_0002054'),
      - constructs a single-element `mutatedSamples` array-of-struct with fields required by downstream
        pipelines (functionalConsequenceId, numberSamplesTested, numberMutatedSamples),
      - parses the `REFERENCE` column into a `literature` array (maps some DOIs to PMIDs),
      - selects cohort and gene-level metadata and normalises column names expected by later steps,
      - left-joins to `disease_mapping` to attach a mapped `diseaseFromSourceMappedId`.

    Args:
        genes: Spark DataFrame with INTOGEN gene/driver rows. Expected columns (at least):
            - SYMBOL: gene symbol used as targetFromSourceId
            - ROLE: role annotation (e.g., 'Act' or 'LoF')
            - SAMPLES: numeric sample counts (used for both numberSamplesTested and numberMutatedSamples)
            - METHODS: comma-separated methods string (turned into `significantDriverMethods`)
            - QVALUE_COMBINATION: numeric score (resourceScore)
            - REFERENCE: citation string parsed by parse_source()
            - CANCER, CANCER_NAME: cohort identifiers (used to join disease mapping)
        cohorts: Spark DataFrame containing cohort metadata. Expected columns:
            - COHORT: cohort key that matches genes.COHORT
            - COHORT_NICK: short cohort name
            - COHORT_NAME: human-readable cohort description
            (the function drops the SAMPLES column from cohorts before joining in the caller)
        disease_mapping: Spark DataFrame mapping cohort acronyms to EFO ids. Expected columns:
            - Cancer_type_acronym (matches genes.CANCER)
            - EFO_id (mapped onto diseaseFromSourceMappedId)

    Returns:
        DataFrame: Spark DataFrame containing generated evidence-like rows with columns including:
            - datatypeId (literal 'somatic_mutation')
            - datasourceId (literal 'intogen')
            - targetFromSourceId (from SYMBOL)
            - cohortId, cohortShortName, cohortDescription
            - significantDriverMethods (array of strings)
            - mutatedSamples (array<struct> with functionalConsequenceId, numberSamplesTested, numberMutatedSamples)
            - diseaseFromSource (original cohort disease string)
            - diseaseFromSourceMappedId (mapped EFO id; may be null if no mapping)
            - resourceScore (from QVALUE_COMBINATION)
            - literature (array<string> of identifiers; some DOIs mapped to PMIDs)

    Notes:
        - `mutatedSamples` is constructed as a single-element array of structs to match the project's
          expected evidence schema for mutation summaries. If INTOGEN provides multiple distinct sample
          groups per gene, consider expanding to multiple array elements.
        - `parse_source` maps some DOIs to PMIDs and returns an array when a reference is present.
        - Columns not explicitly produced by this function are intentionally omitted; downstream
          producers should add any additional required fields.
        - The function does not write output — callers should persist the returned DataFrame.
    """
    # Joining cancer driver genes with cohorts:
    return (
        genes.join(cohorts, on='COHORT', how='inner')
        .select(
            # Adding constant columns:
            f.lit('somatic_mutation').alias('datatypeId'),
            f.lit('intogen').alias('datasourceId'),
            # Extracting gene fields:
            f.col('SYMBOL').alias('targetFromSourceId'),
            # Extracting cohort specific annotation:
            f.col('COHORT').alias('cohortId'),
            f.col('COHORT_NICK').alias('cohortShortName'),
            f.col('COHORT_NAME').alias('cohortDescription'),
            # Splitting methods:
            f.split(f.col('METHODS'), ',').alias('significantDriverMethods'),
            # Generating mutated samples column:
            f.array(
                f.struct(
                    # Parse functional consequences:
                    f.when(f.col('ROLE') == 'Act', 'SO_0002053')  # Gain of function
                    .when(f.col('ROLE') == 'LoF', 'SO_0002054')  # Loss of function
                    .otherwise(None)
                    .alias('functionalConsequenceId'),
                    # Extract samples:
                    f.col('SAMPLES').alias('numberSamplesTested'),
                    f.col('SAMPLES').alias('numberMutatedSamples'),
                )
            ).alias('mutatedSamples'),
            # Adding disease name:
            f.col('CANCER').alias('Cancer_type_acronym'),
            f.col('CANCER_NAME').alias('diseaseFromSource'),
            # Adding score:
            f.col('QVALUE_COMBINATION').alias('resourceScore'),
            parse_source(f.col('REFERENCE')).alias('literature'),
        )
        .join(
            disease_mapping.select(
                'Cancer_type_acronym',
                f.trim(f.col('EFO_id')).alias('diseaseFromSourceMappedId'),
            ),
            on='Cancer_type_acronym',
            how='left',
        )
        .drop('Cancer_type_acronym')
    )


def intogen(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Intogen evidence generation step."""
    # Initialise session:
    session = Session(app_name='orphanet', properties=properties)

    # read input files
    genes = session.spark.read.csv(source['genes'], sep=r'\t', header=True, inferSchema=True)
    cohorts = session.spark.read.csv(source['cohorts'], sep=r'\t', header=True, inferSchema=True).drop('SAMPLES')
    disease_mapping = session.spark.read.csv(source['disease_mapping'], sep=r'\t', header=True)

    # generate intogen evidence
    generate_evidence(genes, cohorts, disease_mapping).write.parquet(destination)
