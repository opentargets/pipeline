"""Pharmacogenetics processing.

This module processes ClinPGx pharmacogenetics data:
1. Parses phenotype text using OpenAI to extract concise phenotype descriptions
2. Adds variant IDs from genotype information
3. Maps phenotypes to EFO disease ontology
4. Maps drug names to ChEMBL drug IDs
5. Enriches with isDirectTarget flag using mechanism of action data
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from openai import OpenAI, OpenAIError
from otter.storage.synchronous.handle import StorageHandle
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from pts.pyspark.common.ontology import add_efo_mapping
from pts.pyspark.common.session import Session


def pharmacogenetics(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Process pharmacogenetics data from ClinPGx.

    Args:
        source: Dictionary with paths to:
            - clinpgx: ClinPGx annotation JSON
            - phenotypes: Phenotypes lookup JSON
            - ontoma_disease_label_lut: OnToma disease label lookup parquet
            - chembl_molecule: ChEMBL molecule parquet for drug ID mapping
            - drug_mechanism_of_action: Mechanism of action parquet for target lookup
        destination: Dictionary with output paths for:
            - associations: Output parquet path
            - phenotypes: Updated phenotypes JSON path
        settings: Settings including openai_token_filename
        properties: Spark configuration options
    """
    spark = Session(app_name='pharmacogenetics', properties=properties)
    # Read OpenAI API key from the source path (automatically resolved by PySpark task)
    openai_token_filename = settings.get('openai_token_filename')
    if not openai_token_filename:
        raise ValueError('openai_token_filename field missing in settings')
    openai_key = Path(openai_token_filename).read_text().strip()

    logger.info(f'load data from {source}')
    pgx_phenotypes_df = spark.load_data(source['phenotypes'], format='json')
    pgx_df = spark.load_data(source['clinpgx'], format='json')
    chembl_molecule_df = spark.load_data(source['chembl_molecule'])
    moa_df = spark.load_data(source['drug_mechanism_of_action'])

    logger.info('overwrite phenotypeText column with parsed phenotypes')
    # Collect texts that appear in ClinPGx but not yet in the phenotypes lookup.
    # A text present in the lookup is considered parsed, even if its phenotypeText
    # is [] (empty extraction is valid).
    unparsed_texts = (
        pgx_df.select('genotypeAnnotationText').distinct().join(
            pgx_phenotypes_df.select('genotypeAnnotationText').distinct(),
            on='genotypeAnnotationText',
            how='left_anti')
        .toPandas()['genotypeAnnotationText'].to_list()
    )
    annotated_pgx_df = annotate_phenotype(pgx_df, pgx_phenotypes_df)
    if len(unparsed_texts) == 0:
        logger.info('all phenotypes have been parsed')
    else:
        logger.warning(f'{len(unparsed_texts)} phenotypes have not been parsed')
        # Retries and a timeout, rather than the client's defaults: a transient failure
        # against a third-party API is likely over a long enough list, and costs that text
        # its extraction. `max_retries` covers connection errors, timeouts and 429/5xx with
        # exponential backoff; `timeout` stops one hung request stalling the step.
        #
        # `br` is excluded from `Accept-Encoding` because the Dataproc image's Brotli 1.1.0
        # has no `output_buffer_limit`, so the `BrotliDecoder` in httpx2 (bundled with
        # openai==3.2.0) raises `TypeError` and the sdk reports it as `APIConnectionError`.
        client = OpenAI(
            api_key=openai_key,
            max_retries=2,
            timeout=30.0,
            default_headers={'Accept-Encoding': 'gzip, deflate'},
        )
        max_workers = int(settings.get('openai_concurrency', 10))
        logger.info(f'parsing {len(unparsed_texts)} phenotypes with concurrency={max_workers}')
        new_phenotypes_df = parse_phenotypes(
            spark=spark,
            texts_to_parse=unparsed_texts,
            openai_client=client,
            max_workers=max_workers,
        )
        updated_phenotypes_df = update_phenotypes_lut(new_phenotypes_df, pgx_phenotypes_df)
        logger.info(f'save updated phenotypes to {destination["phenotypes"]}')
        StorageHandle(destination['phenotypes']).write_text(
            updated_phenotypes_df.toPandas().to_json(orient='records')
        )
        annotated_pgx_df = annotate_phenotype(pgx_df, updated_phenotypes_df)

    logger.info('parse variantId')
    pgx_w_variantid_df = add_variantid_column(annotated_pgx_df)
    logger.info('add efo mappings')
    mapped_pgx_df = add_efo_mapping(
        spark=spark.spark,
        evidence_df=pgx_w_variantid_df,
        label_col_name='phenotypeText',
        disease_label_lut_path=source['ontoma_disease_label_lut'],
        id_col_name=None,
    ).withColumnRenamed('diseaseFromSourceMappedId', 'phenotypeFromSourceId')

    logger.info('map drug IDs and enrich with target information')
    enriched_pgx_df = enrich_with_drug_and_target_info(
        mapped_pgx_df,
        chembl_molecule_df,
        moa_df,
    )

    partition_count = settings.get('partition_count')

    logger.info(f'save associations to {destination["associations"]}')
    out = enriched_pgx_df.coalesce(partition_count) if partition_count is not None else enriched_pgx_df
    out.write.parquet(destination['associations'], mode='overwrite')


def parse_phenotype_with_gpt(
    genotype_text: str, openai_client: OpenAI, gpt_model: str = 'gpt-5-nano-2025-08-07'
) -> list[str] | None:
    """Query the OpenAI API to extract the phenotype from the genotype text."""
    prompt = f"""
        Context: We want to analyse ClinPGx clinical annotations. Their data includes a column,"genotypeAnnotationText",
        which typically informs about efficacy,side effects, or patient response variability given a specific genotype.
        The data is presented in a lengthy and complex format, making it challenging to quickly grasp the key phenotypic
        outcomes.

        Aim: To parse the observed effect in a short string so that the effect can be easily interpreted at a glance.
        The goal is to extract the essence of the pharmacogenetic relationship. This extraction helps in summarizing the
        data for faster and more efficient analysis.

        Please analyse the following examples from the "genotypeAnnotationText" column and extract the key phenotype as
        a concise description. Format the result as a JSON array. Each JSON must only contain one field:
        "gptExtractedPhenotype".

        Examples for extraction:
        1. "Patients with the CTT/del genotype (one copy of the CFTR F508del variant) and cystic fibrosis may have "
           "increased response when treated with ivacaftor/tezacaftor combination as compared to patients with the "
           "CTT/CTT genotype." -> Expected extraction: "increased response"
        2. "Patients with the AC genotype may have "
           "increased risk for gastrointestinal toxicity with taxane and platinum regimens as compared to "
           "patients with the CC genotype." -> Expected extraction: "risk of gastrointestinal toxicity"
        3. "Patients with the rs2032582 AA genotype may be more likely to respond to tramadol "
           "treatment as compared to patients with the CC genotype." -> Expected extraction: "increased response"
        4. "Patients receiving methotrexate to treat acute lymphoblastic leukemia (ALL), and the "
           "rs4149056 TT genotype may be less likely to require glucarpidase treatment as compared to "
           "patients with the CC or CT genotypes." -> Expected extraction: "less likely to require glucarpidase"
        5. "Patients with the TT genotype and hormone insensitive breast cancer may experience "
           "increased risk of chemotherapy-induced amenorrhea when treated with goserelin or combinations of "
           "cyclophosphamide, docetaxel, doxorubicin, epirubicin, and fluorouracil compared to patients with the "
           "CT genotype." -> Expected extraction: "risk of chemotherapy-induced amenorrhea"
        6. "Patients with the GG genotype and cancer may have an increased risk for drug toxicity and an "
           "increased response to treatment with cisplatin or carboplatin as compared to patients with the AA or AG "
           "genotype. Other genetic and clinical factors may also influence a patient's risk for toxicity and "
           "response to platinum-based chemotherapy." -> Expected extraction: "drug toxicity" and "increased response"

        Based on these examples, please extract the phenotype from the following text:

        "{genotype_text}"
    """
    try:
        completion = openai_client.responses.create(
            model=gpt_model,
            text={'format': {'type': 'json_object'}},
            input=[
                {
                    'role': 'system',
                    'content': 'you are an expert in clinical pharmacology designed to output JSON.',
                },
                {'role': 'user', 'content': prompt},
            ],
            reasoning={'effort': 'minimal'},
        )

    except OpenAIError as e:
        # The call itself was previously outside any `try` -- the one below guards only the
        # parsing of a response, which by definition has already arrived. So a connection
        # error, a rate limit or an expired key propagated out of the step and discarded the
        # whole run's work. Returning None instead puts this text on the same footing as one
        # the model declined to extract: `parse_phenotypes` skips it, the row keeps a null
        # phenotypeText, and the step continues.
        logger.warning(f'{type(e).__name__} extracting phenotype, leaving it unparsed: {e}')
        return None
    try:
        generated_text = completion.output_text
        if not generated_text:
            logger.warning(f'No content generated for text: {genotype_text}')
            return None
        json_obj = json.loads(generated_text)
        if 'gptExtractedPhenotype' not in json_obj:
            # Distinguished from an empty extraction, which is legitimate: 1,411 of the
            # curated entries have an empty phenotypeText because the annotation genuinely
            # describes no phenotype. A missing key is different -- the model answered in a
            # shape the prompt did not ask for. Defaulting it to [] recorded that as a
            # successful extraction, so the text entered the lookup table permanently empty
            # and was never attempted again.
            logger.warning(f'no gptExtractedPhenotype in the response for: {genotype_text[:80]}')
            return None
        return json_obj['gptExtractedPhenotype']
    except Exception as e:
        logger.error(f'Error parsing phenotype: {e}')
        return None


def parse_phenotypes(
    spark: Session, texts_to_parse: list[str], openai_client: OpenAI, max_workers: int = 10
) -> DataFrame:
    """Parse the phenotypes from the given texts by calling the OpenAI API.

    Texts that cannot be extracted are left out rather than failing the step: they keep a
    null `phenotypeText`, exactly as they would have before the API was consulted. The
    counts are logged because that degradation is otherwise invisible -- a step that
    extracts nothing and one that extracts everything both succeed.

    Concurrency is bounded by ``max_workers`` (default 10) via a thread pool.
    """
    results_dict: dict[str, list[str]] = {}
    total = len(texts_to_parse)
    if total == 0:
        logger.info('no phenotypes to parse')
    else:
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # future_to_text maps each Future back to its input text (genotypeAnnotationText)
            future_to_text = {
                executor.submit(parse_phenotype_with_gpt, text, openai_client): text
                for text in texts_to_parse
            }
            # as_completed yields futures in completion order (fastest first)
            for i, future in enumerate(as_completed(future_to_text), 1):
                text = future_to_text[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.warning(f'{type(e).__name__} extracting phenotype, leaving it unparsed: {e}')
                    result = None
                if isinstance(result, list):
                    results_dict[text] = result
                elif isinstance(result, str):
                    results_dict[text] = [result]
                if i % 50 == 0 or i == total:
                    logger.info(f'progress {i}/{total}')
        elapsed = time.perf_counter() - t_start
        logger.info(f'parsed {len(results_dict)}/{total} in {elapsed:.1f}s')

    unresolved = total - len(results_dict)
    if unresolved == total and texts_to_parse:
        logger.error(
            f'extracted none of {total} phenotypes; the API was unreachable, '
            f'unauthorised or rate limited throughout. Those rows keep a null phenotypeText '
            f'and the step continues.'
        )
    elif unresolved:
        logger.warning(f'extracted {len(results_dict)} of {total} phenotypes')
    else:
        logger.info(f'extracted all {len(results_dict)} phenotypes')

    return spark.spark.createDataFrame(
        list(results_dict.items()),
        StructType([
            StructField('genotypeAnnotationText', StringType(), True),
            StructField('phenotypeText', ArrayType(StringType()), True),
        ]),
    )


def update_phenotypes_lut(
    new_phenotypes_df: DataFrame,
    extracted_phenotypes_df: DataFrame,
) -> DataFrame:
    """Adds the new phenotypes to the extracted phenotypes table."""
    return extracted_phenotypes_df.unionByName(new_phenotypes_df).distinct()


def annotate_phenotype(pgx_evidence_df: DataFrame, extracted_phenotypes_df: DataFrame) -> DataFrame:
    """This module overwrites the `phenotypeText`, which comes from ClinPGx directly, but it is usually too verbose.

    Args:
        pgx_evidence_df: Dataframe with the PGx evidence submitted by EVA.
        extracted_phenotypes_df: Dataframe containing the phenotypes extracted from `genotypeAnnotationText`.
    """
    return (
        pgx_evidence_df
        .drop('phenotypeText', 'phenotypeFromSourceId')
        .join(extracted_phenotypes_df, on='genotypeAnnotationText', how='left')
        .withColumn('phenotypeText', f.explode_outer('phenotypeText'))
        .distinct()
    )


def add_variantid_column(input_df: DataFrame) -> DataFrame:
    """Based on the content of the genotypeId column, adds a variantId column to the dataset."""
    return (
        input_df
        # split genotypeId column into chr pos ref alt columns
        .select(
            'genotypeId',
            f.from_csv(
                f.col('genotypeId'),
                'chr string, pos string, ref string, alt string',
                {'sep': '_'},
            ).alias('genotype_split'),
        )
        .select('genotypeId', 'genotype_split.*')
        .toDF('genotypeId', 'chr', 'pos', 'ref', 'alt')
        # split alt column and explode
        .withColumn('alt', f.explode(f.split(f.col('alt'), ',')))
        .filter(~(f.col('ref') == f.col('alt')))
        .select(
            'genotypeId',
            f.concat_ws('_', f.col('chr'), f.col('pos'), f.col('ref'), f.col('alt')).alias('variantId'),
        )
        .join(input_df, on='genotypeId', how='right')
    )


def enrich_with_drug_and_target_info(
    pgx_df: DataFrame,
    molecule_df: DataFrame,
    moa_df: DataFrame,
) -> DataFrame:
    """Enrich pharmacogenetics data with drug IDs and isDirectTarget flag.

    This function:
    1. Explodes the drugs array to process each drug individually
    2. Maps drug names to ChEMBL drug IDs
    3. Determines if the variant target is a direct target of the drug (using mechanism of action)
    4. Groups back by the original row structure

    Args:
        pgx_df: Pharmacogenetics DataFrame with drugs array.
        molecule_df: ChEMBL molecule DataFrame for drug ID mapping.
        moa_df: Mechanism of action DataFrame for target lookup.

    Returns:
        Enriched DataFrame with drugId and isDirectTarget.
    """
    # Get drug name to ID lookup table
    drug_name_lut = _get_drug_name_lut(molecule_df)

    # Get drug to target lookup table from mechanism of action
    drug_target_lut = _get_drug_target_lut(moa_df)

    # Add operational row ID for grouping back later
    pgx_expanded = (
        pgx_df
        .withColumn('_operationalRowId', f.monotonically_increasing_id())
        .withColumn('drug', f.explode('drugs'))
        .withColumn('drugFromSource', f.col('drug.drugFromSource'))
        .drop('drugs')
    )

    # Map drug IDs using the lookup table
    pgx_with_drug_id = _map_drug_id(pgx_expanded, drug_name_lut)

    # Join with drug-target lookup and flag direct targets
    pgx_enriched = (
        pgx_with_drug_id
        .join(drug_target_lut, on='drugId', how='left')
        .withColumn(
            'isDirectTarget',
            f.when(
                f.array_contains(f.col('drugTargetIds'), f.col('targetFromSourceId')),
                f.lit(True),
            ).otherwise(f.lit(False)),
        )
        .drop('drugTargetIds')
        .distinct()
    )

    # Group back by the original row, collecting drugs into an array
    grouping_cols = [
        '_operationalRowId',
        'datasourceId',
        'datasourceVersion',
        'datatypeId',
        'directionality',
        'evidenceLevel',
        'genotype',
        'genotypeAnnotationText',
        'genotypeId',
        'haplotypeFromSourceId',
        'haplotypeId',
        'literature',
        'pgxCategory',
        'phenotypeFromSourceId',
        'phenotypeText',
        'variantAnnotation',
        'studyId',
        'targetFromSourceId',
        'variantFunctionalConsequenceId',
        'variantRsId',
        'variantId',
        'isDirectTarget',
    ]

    # Filter to only include columns that exist in the dataframe
    existing_cols = [c for c in grouping_cols if c in pgx_enriched.columns]

    return (
        pgx_enriched
        .groupBy(*existing_cols)
        .agg(f.collect_list(f.struct(f.col('drugFromSource'), f.col('drugId'))).alias('drugs'))
        .drop('_operationalRowId')
    )


def _get_drug_name_lut(molecule_df: DataFrame) -> DataFrame:
    """Create a lookup table mapping drug names to ChEMBL drug IDs.

    When multiple IDs exist for the same name, selects one deterministically
    by sorting and taking the last one (to match Scala behavior).

    Args:
        molecule_df: ChEMBL molecule DataFrame with id and name columns.

    Returns:
        DataFrame with drugFromSource and drugId columns.
    """
    return (
        molecule_df
        .select(f.col('id'), f.lower(f.col('name')).alias('drugFromSource'))
        .filter(f.col('drugFromSource').isNotNull())
        .groupBy('drugFromSource')
        .agg(f.collect_set('id').alias('ids'))
        .select(
            f.col('drugFromSource'),
            f.element_at(f.sort_array(f.col('ids'), asc=False), 1).alias('drugId'),
        )
    )


def _map_drug_id(pgx_df: DataFrame, drug_name_lut: DataFrame) -> DataFrame:
    """Map drug names to ChEMBL drug IDs.

    Args:
        pgx_df: Pharmacogenetics DataFrame with drugFromSource column.
        drug_name_lut: Lookup table with drugFromSource and drugId columns.

    Returns:
        DataFrame with drugId column added.
    """
    return pgx_df.withColumn('drugFromSource', f.lower(f.col('drugFromSource'))).join(
        drug_name_lut, on='drugFromSource', how='left'
    )


def _get_drug_target_lut(moa_df: DataFrame) -> DataFrame:
    """Create a lookup table mapping drug IDs to their target IDs.

    Extracts target information from the mechanism of action data,
    exploding chemblIds to get all drug IDs associated with each mechanism.

    Args:
        moa_df: Mechanism of action DataFrame with chemblIds and targets columns.

    Returns:
        DataFrame with drugId and drugTargetIds columns.
    """
    return (
        moa_df
        .filter(f.col('targets').isNotNull() & (f.size(f.col('targets')) >= 1))
        .select(
            f.explode(f.col('chemblIds')).alias('drugId'),
            f.col('targets'),
        )
        .groupBy('drugId')
        .agg(f.array_distinct(f.flatten(f.collect_list('targets'))).alias('drugTargetIds'))
    )
