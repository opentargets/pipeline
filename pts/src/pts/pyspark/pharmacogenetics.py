"""This module adds a more granular description of the phenotype observed in the ClinPGX evidence."""

import json
from pathlib import Path
from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from openai import OpenAI
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
    spark = Session(app_name='pharmacogenetics', properties=properties)
    # Read OpenAI API key from the source path (automatically resolved by PySpark task)
    openai_token_filename = settings.get('openai_token_filename')
    if not openai_token_filename:
        raise ValueError('openai_token_filename field missing in settings')
    openai_key = Path(openai_token_filename).read_text().strip()

    logger.info(f'load data from {source}')
    pgx_phenotypes_df = spark.load_data(source['phenotypes'], format='json')
    pgx_df = spark.load_data(source['clinpgx'], format='json')

    logger.info('overwrite phenotypeText column with parsed phenotypes')
    annotated_pgx_df = annotate_phenotype(pgx_df, pgx_phenotypes_df)
    unparsed_texts = (
        annotated_pgx_df
        .filter(f.col('phenotypeText').isNull())  # ty:ignore[missing-argument]
        .select('genotypeAnnotationText')
        .distinct()
        .toPandas()['genotypeAnnotationText']
        .to_list()
    )
    if len(unparsed_texts) == 0:
        logger.info('all phenotypes have been parsed')
    else:
        logger.warning(f'{len(unparsed_texts)} phenotypes have not been parsed')
        client = OpenAI(api_key=openai_key)
        new_phenotypes_df = parse_phenotypes(
            spark=spark,
            texts_to_parse=unparsed_texts,
            openai_client=client,
        )
        updated_phenotypes_df = update_phenotypes_lut(new_phenotypes_df, pgx_phenotypes_df)
        logger.info(f'save updated phenotypes to {destination["phenotypes"]}')
        updated_phenotypes_df.toPandas().to_json(destination['phenotypes'], orient='records')
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
    logger.info(f'save associations to {destination["associations"]}')
    mapped_pgx_df.write.parquet(destination['associations'], mode='overwrite')


def parse_phenotype_with_gpt(
    genotype_text: str, openai_client: OpenAI, gpt_model: str = 'gpt-3.5-turbo-1106'
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
    completion = openai_client.chat.completions.create(
        model=gpt_model,
        response_format={'type': 'json_object'},
        messages=[
            {'role': 'system', 'content': 'you are an expert in clinical pharmacology designed to output JSON.'},
            {'role': 'user', 'content': prompt},
        ],
        seed=42,
    )
    try:
        generated_text = completion.choices[0].message.content
        if not generated_text:
            logger.warning(f'No content generated for text: {genotype_text}')
            return None
        json_obj = json.loads(generated_text)
        return json_obj.get('gptExtractedPhenotype', [])
    except Exception as e:
        logger.error(f'Error parsing phenotype: {e}')
        return None


def parse_phenotypes(spark: Session, texts_to_parse: list[str], openai_client: OpenAI) -> DataFrame:
    """Parse the phenotypes from the given texts by calling the OpenAI API."""
    results_dict = {}
    for text in texts_to_parse:
        result = parse_phenotype_with_gpt(text, openai_client)
        if isinstance(result, list):
            results_dict[text] = result
        elif isinstance(result, str):
            results_dict[text] = [result]
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
    return extracted_phenotypes_df.unionByName(new_phenotypes_df)


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
            f.from_csv(f.col('genotypeId'), 'chr string, pos string, ref string, alt string', {'sep': '_'}).alias(
                'genotype_split'
            ),
        )
        .select('genotypeId', 'genotype_split.*')
        .toDF('genotypeId', 'chr', 'pos', 'ref', 'alt')
        # split alt column and explode
        .withColumn('alt', f.explode(f.split(f.col('alt'), ',')))
        .filter(~(f.col('ref') == f.col('alt')))
        .select(
            'genotypeId', f.concat_ws('_', f.col('chr'), f.col('pos'), f.col('ref'), f.col('alt')).alias('variantId')
        )
        .join(input_df, on='genotypeId', how='right')
    )
