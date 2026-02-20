import json
import re
from functools import reduce
from typing import Any

from loguru import logger
from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.common.session import Session

LITERATURE_MAPPING = [
    ('https://www.biorxiv.org/content/10.1101/2021.09.11.459904v1', ['ppr393667']),
    ('https://pubmed.ncbi.nlm.nih.gov/30297964/', ['30297964']),
    ('https://www.biorxiv.org/content/10.1101/2021.08.23.457400v1', ['PPR387167']),
    ('https://doi.org/10.1101/2020.06.27.175679', ['34031600']),
    ('https://pubmed.ncbi.nlm.nih.gov/31422865/', ['31422865']),
    ('https://pubmed.ncbi.nlm.nih.gov/30449619/', ['30449619']),
]

GENES_SCHEMA = t.StructType([
    t.StructField('Gene', t.StringType(), True),
    t.StructField('TSS', t.StringType(), True),
    t.StructField('Phenotype', t.DoubleType(), True),
    t.StructField('P Value', t.FloatType(), True),
    t.StructField('Gene Score', t.FloatType(), True),
    t.StructField('Hit Class', t.StringType(), True),
    t.StructField('P Value (-Log10)', t.DoubleType(), True),
    t.StructField('P Value (Signed -Log10)', t.DoubleType(), True),
    t.StructField('Gene Rank', t.FloatType(), True),
    t.StructField('Gene Score (Size)', t.FloatType(), True),
    t.StructField('Phenotype (Size)', t.FloatType(), True),
    t.StructField('P Value (-Log10) (Size)', t.FloatType(), True),
])

EXPERIMENT_SCHEMA = t.StructType([
    t.StructField('title', t.StringType()),
    t.StructField('experiment', t.StringType()),
    t.StructField('analysis', t.StringType()),
])


def crispr_screens(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Process CRISPR Brain inputs into target/disease evidence.

    Expected `source` keys:
      - screens: path to gzipped JSON screens payload
      - studies_dir: directory with per-study CSV.GZ files
      - disease_mapping: TSV with studyId -> EFOs and contrast

    Args:
        source (dict[str,str]): pts source configuration
        destination (str): evidence output
        settings (dict[str,str]): pts evidence specific configuration
        properties (dict[str,str]): pts spark session configuration
    """
    session = Session(app_name='crispr_screens', properties=properties)

    # Register udf:
    session.spark.udf.register('parsing_experiment', parsing_experiment, EXPERIMENT_SCHEMA)

    logger.info(f'loading data from: {source}')
    studies_glob = f'{source["studies_dir"].rstrip("/")}/**.csv.gz'
    logger.debug(f'reading study files from: {studies_glob}')
    genes_df = session.load_data(studies_glob, format='csv', header=True, schema=GENES_SCHEMA)
    disease_mapping_df = session.load_data(source['disease_mapping'], format='csv', sep='\t', header=True)

    screens_obj = json.loads(session.spark.read.text(source['screens'], wholetext=True).collect()[0][0])

    logger.info('processing crispr screens into evidence')
    screen_rows = [{**v, **v.get('metadata', {})} for v in screens_obj.values() if isinstance(v, dict)]
    screens_df = session.spark.createDataFrame(screen_rows)
    screens_df = screens_df.withColumn('studySummary', f.expr('parsing_experiment("Description")'))
    studies_df = (
        screens_df
        .join(
            # Build literature references
            session.spark.createDataFrame(
                LITERATURE_MAPPING,
                ['Reference Link', 'literature'],
            ),
            on='Reference Link',
            how='left',
        )
        .select(
            f.lit('affected_pathway').alias('datatypeId'),
            f.lit('crispr_screen').alias('datasourceId'),
            f.lit('crispr_brain').alias('projectId'),
            f.col('Screen Name').alias('studyId'),
            f.col('Libraries Screened').alias('crisprScreenLibrary'),
            f.col('studySummary.title').alias('studyOverview'),
            f.col('Cell Type').alias('cellType'),
            f.when(f.col('Genotype') != 'WT', f.col('Genotype')).otherwise(None).alias('geneticBackground'),
            'Phenotype',
            'literature',
        )
        # Build screen disease annotation
        .join(disease_mapping_df, on='studyId', how='left')
        .withColumn('contrast', f.coalesce('contrast', 'Phenotype'))
        .drop('Phenotype')
        # QC and filter for studies that have disease mapping
        .filter(f.col('diseaseFromSourceMappedId').isNotNull())  # ty:ignore[missing-argument]
        .distinct()
    )

    genes_df = (
        genes_df
        # derive studyId from filename
        .withColumn('input_file', f.input_file_name())
        .withColumn('studyId', f.regexp_replace(f.regexp_extract('input_file', r'([^/]+)\.csv\.gz$', 1), r'%20', ' '))
        .filter(
            f.col('Hit Class').isNotNull()  # ty:ignore[missing-argument]
            & ((f.col('Hit Class') == 'Positive Hit') | (f.col('Hit Class') == 'Negative Hit'))
        )
        .select(
            f.col('Gene').alias('targetFromSourceId'),
            f.col('P Value').alias('resourceScore'),
            f
            .when(f.col('Hit Class') == 'Positive Hit', f.lit('upper tail'))
            .when(f.col('Hit Class') == 'Negative Hit', f.lit('lower tail'))
            .alias('statisticalTestTail'),
            f.col('Phenotype').alias('log2FoldChangeValue'),
            'studyId',
        )
        .distinct()
    )
    evidence_df = studies_df.join(genes_df, on='studyId', how='inner').withColumn(
        'diseaseFromSourceMappedId', f.explode(f.split(f.col('diseaseFromSourceMappedId'), ', '))
    )

    logger.debug(f'writing output data to: {destination}')
    evidence_df.write.parquet(destination)


def parsing_experiment(description: str) -> dict:
    r"""Parse experimental details from descriptoin.

    Args:
        description (str): string representation of the experiment description.

    Returns:
        dic[str, str]: parsed description

    Examples:
        >>> import json
        >>> s = '''## iAstrocyte-LAMP1-Veh-CRISPRi
        ...
        ... CRISPRi FACS screen for cell-surface LAMP1 (marker of lysosome
        ... exocytosis) in human iPSC-derived astrocytes (iAstrocytes) treated with
        ... vehicle control.
        ...
        ... ## Experiment
        ...
        ... To screen against the druggable genome or genes involved in vesicular
        ... trafficking, we used the H1 and H4 sub-libraries from Horlbeck et al.
        ... The H1 and H4 sublibraries were packaged into lentivirus as previously
        ... described. For each experimental replicate, ~10 million iAstrocytes were
        ... plated..., transduced, treated and sorted as described in the study.
        ...
        ... ## Analysis
        ...
        ... We analyzed the data from the pooled CRISPRi screens using the
        ... MAGeCK-iNC bioinformatic pipeline previously described.
        ... '''
        >>> print(json.dumps(parsing_experiment(s),indent=2))
        {
          "title": "iAstrocyte-LAMP1-Veh-CRISPRi",
          "experiment": "To screen against the druggable genome or genes involved in vesicular",
          "analysis": "We analyzed the data from the pooled CRISPRi screens using the"
        }
    """
    # Clean
    repl_patterns = [(r'\*+', ''), (r'\r', ''), (r'\t', ''), (r'\n+', '\n')]
    description = reduce(lambda s, p: re.sub(p[0], p[1], s), repl_patterns, description or '')
    description = description.encode('ascii', 'ignore').decode('ascii')

    # Split
    lines = re.split(r'\n+', description.strip()) if description else ['']
    title = re.sub(r'#+\s+', '', lines[0]) if lines and '#' in lines[0] else (lines[0] if lines else None)

    experiment = None
    analysis = None
    try:
        for i in range(len(lines)):
            if lines[i] == '## Experiment':
                experiment = lines[i + 1]
            if lines[i] == '## Analysis':
                analysis = lines[i + 1]
    except Exception:
        experiment = None
        analysis = None

    return {'title': title, 'experiment': experiment, 'analysis': analysis}
