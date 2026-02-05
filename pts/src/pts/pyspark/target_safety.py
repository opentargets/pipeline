"""This module puts together data from different sources that describe target safety liabilities."""

from functools import partial, reduce
from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import Column, DataFrame

from pts.pyspark.common.session import Session


def target_safety(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """This module puts together data from different sources that describe target safety liabilities."""
    spark = Session(app_name='target_safety', properties=properties)

    logger.info(f'load data from {source}')
    adverse_events_df = spark.load_data(source['adverse_events'], format='csv', sep='\t', header=True)
    safety_risks_df = spark.load_data(source['safety_risks'], format='csv', sep='\t', header=True)
    toxcast_df = spark.load_data(source['toxcast'], format='csv', sep='\t', header=True)
    aopwiki_df = spark.load_data(source['aopwiki'], format='json')
    brennan_df = spark.load_data(source['brennan'], format='json')
    pharmacogenetics_df = spark.load_data(source['pharmacogenetics'])

    logger.info('transforming target safety evidence')
    safety_dfs = [
        process_adverse_events(adverse_events_df),
        process_safety_risk(safety_risks_df),
        process_toxcast(toxcast_df),
        process_aop(aopwiki_df),
        process_pharmacogenetics(pharmacogenetics_df),
        process_brennan(brennan_df),
    ]

    logger.info('combine safety evidence')
    evidence_unique_cols = [
        'id',
        'targetFromSourceId',
        'event',
        'eventId',
        'datasource',
        'effects',
        'literature',
        'url',
    ]
    union_by_diff_schema = partial(DataFrame.unionByName, allowMissingColumns=True)
    safety_df = (
        reduce(union_by_diff_schema, safety_dfs)
        # Collect biosample and study metadata by grouping on the unique evidence fields
        .groupBy(evidence_unique_cols)
        .agg(
            f.collect_set(f.col('biosample')).alias('biosamples'),
            f.collect_set(f.col('study')).alias('studies'),
            f.collect_set(f.col('supporting_variation')).alias('supporting_variation'),
        )
        .withColumn('biosamples', f.when(f.size('biosamples') != 0, f.col('biosamples')))
        # Add the supporting variation to the study metadata for the PGx evidence
        .withColumn(
            'studies',
            f.when(
                f.col('datasource') == 'ClinPGx',
                f.transform(
                    f.col('studies'),
                    lambda x: f.struct(
                        f.concat(
                            f.lit('Genetic variation linked to this safety liability: '),
                            f.array_join(f.col('supporting_variation'), ', '),
                        ).alias('description'),
                        x['name'].alias('name'),
                        x['type'].alias('type'),
                    ),
                ),
            ).otherwise(f.col('studies')),
        )
        .withColumn('studies', f.when(f.size('studies') != 0, f.col('studies')))
        .drop('supporting_variation')
        .distinct()
    )
    logger.info(f'save associations to {destination}')
    safety_df.write.mode('overwrite').parquet(destination)


def process_aop(aopwik_df: DataFrame) -> DataFrame:
    """Loads and processes the AOPWiki input JSON."""
    return (
        aopwik_df.withColumn(
            'study',
            f.struct(
                f.lit(None).cast('string').alias('description'),
                f.lit(None).cast('string').alias('name'),
                f.lit('cell-based').alias('type'),
            ),
        )
        # data bug: some events have the substring "NA" at the start - removal and trim the string
        .withColumn('event', f.trim(f.regexp_replace(f.col('event'), '^NA', '')))
        # data bug: effects.direction need to be in lowercase, this field is an enum
        .withColumn(
            'effects',
            f.transform(
                f.col('effects'),
                lambda x: f.struct(
                    f.when(
                        x.direction == 'Activation',
                        f.lit('Activation/Increase/Upregulation'),
                    )
                    .when(
                        x.direction == 'Inhibition',
                        f.lit('Inhibition/Decrease/Downregulation'),
                    )
                    .alias('direction'),
                    x.dosing.alias('dosing'),
                ),
            ),
        )
        # Convert biosamples array into struct for consistent parsing with other sources
        .withColumn('biosample', f.explode_outer('biosamples'))
        .withColumn('supporting_variation', f.lit(None).cast('string'))  # Add missing column for schema consistency
    )


def process_adverse_events(adverse_events_df: DataFrame) -> DataFrame:
    """Loads and processes the adverse events input TSV.

    Ex. input record:
        biologicalSystem | gastrointestinal
        effect           | activation_general
        efoId            | EFO_0009836
        ensemblId        | ENSG00000133019
        pmid             | 23197038
        ref              | Bowes et al. (2012)
        symptom          | bronchoconstriction
        target           | CHRM3
        uberonCode       | UBERON_0005409
        url              | null

    Ex. output record:
        id          | ENSG00000133019
        event       | bronchoconstriction
        datasource  | Bowes et al. (2012)
        eventId     | EFO_0009836
        literature  | 23197038
        url         | null
        biosample   | {gastrointestinal, UBERON_0005409, null, null, null}
        effects     | [{Activation/Increase/Upregulation, general}]
    """
    source_to_study_type = {
        'Lynch et al. (2017)': 'preclinical',
        'Bowes et al. (2012)': 'preclinical',
        'Urban et al. (2012)': 'clinical',
    }
    ae_df = (
        adverse_events_df.select(
            f.col('ensemblId').alias('id'),
            f.col('symptom').alias('event'),
            f.col('efoId').alias('eventId'),
            f.col('ref').alias('datasource'),
            f.col('pmid').alias('literature'),
            'url',
            f.struct(
                f.col('biologicalSystem').alias('tissueLabel'),
                f.col('uberonCode').alias('tissueId'),
                f.lit(None).alias('cellLabel'),
                f.lit(None).alias('cellFormat'),
                f.lit(None).alias('cellId'),
            ).alias('biosample'),
            f.split(f.col('effect'), '_').alias('effects'),
        )
        .withColumn(
            'effects',
            f.struct(
                f.when(
                    f.col('effects')[0].contains('activation'),
                    f.lit('Activation/Increase/Upregulation'),
                )
                .when(
                    f.col('effects')[0].contains('inhibition'),
                    f.lit('Inhibition/Decrease/Downregulation'),
                )
                .alias('direction'),
                f.element_at(f.col('effects'), 2).alias('dosing'),
            ),
        )
        .withColumn('studyType', f.col('datasource'))
        .replace(to_replace=source_to_study_type, subset=['studyType'])
        .withColumn(
            'study',
            f.struct(
                f.col('event').alias('description'),
                f.col('eventId').alias('name'),
                f.col('studyType').alias('type'),
            ),
        )
        .drop('studyType')
    )

    # Multiple dosing effects need to be grouped in the same record.
    effects_df = ae_df.groupBy('id', 'event', 'datasource').agg(f.collect_set(f.col('effects')).alias('effects'))
    return (
        ae_df.drop('effects')
        .join(effects_df, on=['id', 'event', 'datasource'], how='left')
        .withColumn('supporting_variation', f.lit(None).cast('string'))  # Add missing column for schema consistency
    )


def process_brennan(brennan_df: DataFrame) -> DataFrame:
    """Loads and processes the Brennan input JSON prepared by the Target Safety team."""
    return (
        brennan_df.withColumn(
            'effects',
            f.array(
                f.struct(
                    f.when(
                        f.col('effects.direction') == 'Activation',
                        f.lit('Activation/Increase/Upregulation'),
                    )
                    .when(
                        f.col('effects.direction') == 'Inhibition',
                        f.lit('Inhibition/Decrease/Downregulation'),
                    )
                    .otherwise(f.col('effects.direction'))
                    .alias('direction'),
                    f.col('effects.dosing'),
                )
            ),
        )
        # Explicitly create the study struct with the correct field names to prevent schema inference issues
        .withColumn(
            'study',
            f.struct(
                f.col('studies.description').alias('description'),
                f.col('studies.name').alias('name'),
                f.col('studies.type').alias('type'),
            ),
        )
        .withColumnRenamed('biosamples', 'biosample')
        .withColumn('supporting_variation', f.lit(None).cast('string'))  # Add missing column for schema consistency
        .drop('Type', 'studies')  # Drop the original studies column
    )


def process_safety_risk(safety_risk_df: DataFrame) -> DataFrame:
    """Loads and processes the safety risk information input TSV.

    Ex. input record:
        biologicalSystem | cardiovascular sy...
        ensemblId        | ENSG00000132155
        event            | heart disease
        eventId          | EFO_0003777
        liability        | Important for the...
        pmid             | 21283106
        ref              | Force et al. (2011)
        target           | RAF1
        uberonId         | UBERON_0004535

    Ex. output record:
        id         | ENSG00000132155
        event      | heart disease
        eventId    | EFO_0003777
        literature | 21283106
        datasource | Force et al. (2011)
        biosample  | {cardiovascular s...
        study      | {Important for th...
    """
    return (
        safety_risk_df.withColumn(
            'studyType',
            f.when(f.col('ref').contains('Force'), 'preclinical').when(f.col('ref').contains('Lamore'), 'cell-based'),
        )
        .select(
            f.col('ensemblId').alias('id'),
            'event',
            'eventId',
            f.col('pmid').alias('literature'),
            f.col('ref').alias('datasource'),
            f.struct(
                f.col('biologicalSystem').alias('tissueLabel'),
                f.col('uberonId').alias('tissueId'),
                f.lit(None).alias('cellLabel'),
                f.lit(None).alias('cellFormat'),
                f.lit(None).alias('cellId'),
            ).alias('biosample'),
            f.struct(
                f.col('liability').alias('description'),
                f.lit(None).alias('name'),
                f.col('studyType').alias('type'),
            ).alias('study'),
        )
        .withColumn(
            'event',
            f.when(f.col('datasource').contains('Force'), 'heart disease').when(
                f.col('datasource').contains('Lamore'), 'cardiac arrhythmia'
            ),
        )
        .withColumn(
            'eventId',
            f.when(f.col('datasource').contains('Force'), 'EFO_0003777').when(
                f.col('datasource').contains('Lamore'), 'EFO_0004269'
            ),
        )
        .withColumn('supporting_variation', f.lit(None).cast('string'))  # Add missing column for schema consistency
    )


def process_toxcast(toxcast_df: DataFrame) -> DataFrame:
    """Loads and processes the ToxCast input table.

    Ex. input record:
        assay_component_endpoint_name | ACEA_ER_80hr
        assay_component_desc          | ACEA_ER_80hr, is ...
        biological_process_target     | cell proliferation
        tissue                        | null
        cell_format                   | cell line
        cell_short_name               | T47D
        assay_format_type             | cell-based
        official_symbol               | ESR1
        eventId                       | null

    Ex. output record:
    targetFromSourceId | ESR1
    event              | cell proliferation
    eventId            | null
    biosample          | {null, null, T47D...
    datasource         | ToxCast
    url                | https://www.epa.g...
    study              | {ACEA_ER_80hr, AC...
    """
    return toxcast_df.select(
        f.trim(f.col('official_symbol')).alias('targetFromSourceId'),
        f.col('biological_process_target').alias('event'),
        'eventId',
        f.struct(
            f.col('tissue').alias('tissueLabel'),
            f.lit(None).alias('tissueId'),
            f.col('cell_short_name').alias('cellLabel'),
            f.col('cell_format').alias('cellFormat'),
            f.lit(None).alias('cellId'),
        ).alias('biosample'),
        f.lit('ToxCast').alias('datasource'),
        f.lit('https://www.epa.gov/chemical-research/exploring-toxcast-data-downloadable-data').alias('url'),
        f.struct(
            f.col('assay_component_desc').alias('description'),
            f.col('assay_component_endpoint_name').alias('name'),
            f.col('assay_format_type').alias('type'),
        ).alias('study'),
    ).withColumn('supporting_variation', f.lit(None).cast('string'))  # Add missing column for schema consistency


def process_pharmacogenetics(pgx_df: DataFrame) -> DataFrame:
    """Given the pharmacogenetics evidence, extract the evidence related to target toxicity."""
    clinpgx_url_template = 'https://www.clinpgx.org/search?query='
    return (
        pgx_df
        # Only interested in the evidence that is related to toxicity
        .filter(f.col('pgxCategory') == 'toxicity')
        .filter((f.col('targetFromSourceId').isNotNull()) & (f.col('phenotypeText').isNotNull()))
        # Safety liabilities extraction
        .filter(
            ~(
                f.col('phenotypeText').contains('no significant association')
                | f.col('phenotypeText').contains('not associated with')
            )
        )
        .withColumn('event', clean_phenotype_to_describe_safety_event(f.col('phenotypeText')))
        .filter(f.col('event') != 'drug response')
        # Explode evidence by the variation that supports the liability - this will be used to build the study metadata
        .withColumn(
            'supporting_variation',
            f.explode(
                f.array_union(f.array('genotypeId'), f.array('haplotypeId')),
            ),
        )
        # Define unaggregated target/event pairs
        .select(
            f.col('targetFromSourceId').alias('id'),
            'event',
            f.lit(None).cast('string').alias('eventId'),  # Add missing eventId column
            f.lit('ClinPGx').alias('datasource'),
            f.concat(f.lit(clinpgx_url_template), f.col('targetFromSourceId')).alias('url'),
            # To build study metadata later - each study is a drug after which the phenotype was observed
            f.explode(f.col('drugs.drugFromSource')).alias('drugFromSource'),
            'supporting_variation',
        )
        .withColumn(
            'study',
            f.struct(
                f.lit(None).cast('string').alias('description'),
                f.concat(f.col('drugFromSource'), f.lit(' induced effect')).alias('name'),
                f.lit('clinical').alias('type'),
            ),
        )
    )


def clean_phenotype_to_describe_safety_event(phenotype_col_name: Column) -> Column:
    words_to_remove = (
        r' but not absent,|but not absent|, but not absent,|, but not absent|, but not non-existent,'
        r'(but not absent) |improved|risk and severity of|increased risk and increased severity of|'
        r'reduced risk and reduced severity of|decreased risk and reduced severity of|similar|'
        r'increased|decreased|not have altered risk of |greater|reduced|phenotype|risk of|'
        r'risk for|risk of developing|less likely to |more likely to |less severe|more severe|'
        r'smaller|likelihood of|severity of|developing|experiencing|experience|develop|'
        r'treatment related|drug-induced|oxcarbazepine-induced|aspirin induced|higher|lower|'
        r'INCREASED|Increased|including|drug toxicity, particularly |unknown likelihood of experiencing|'
        r'high frequency of'
    )
    context_stop_words = (
        r'(as a result of taking|based on|when administered|during treatment|'
        r'when taking|due to unintentional|with docetaxel and thalidomide)'
    )
    pattern = rf'^(.*?)(?:{context_stop_words}.*)?$'
    replacements = {
        'toxicity': 'drug toxicity',
        'response': 'drug response',
        'altered drug toxicity': 'drug toxicity',
        'risk and drug toxicity': 'drug toxicity',
        'risk': 'drug toxicity',
    }

    cleaned_col = f.regexp_replace(phenotype_col_name, words_to_remove, '')
    cleaned_col = f.regexp_extract(cleaned_col, pattern, 1)
    cleaned_col = f.trim(f.regexp_replace(cleaned_col, r'\s+', ' '))

    for original, replacement in replacements.items():
        cleaned_col = f.when(cleaned_col == original, f.lit(replacement)).otherwise(cleaned_col)

    return cleaned_col
