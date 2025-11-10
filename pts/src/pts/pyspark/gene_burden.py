from functools import partial, reduce
from typing import Any

import pandas as pd
import pyspark.sql.functions as f
import pyspark.sql.types as t
from loguru import logger
from pyspark.sql.dataframe import DataFrame

from pts.pyspark.common.session import Session

CURATION_SCHEMA = t.StructType([
    t.StructField('projectId', t.StringType(), True),
    t.StructField('targetFromSource', t.StringType(), True),
    t.StructField('targetFromSourceId', t.StringType(), True),
    t.StructField('diseaseFromSource', t.StringType(), True),
    t.StructField('diseaseFromSourceMappedId', t.StringType(), True),
    t.StructField('resourceScore', t.DoubleType(), True),
    t.StructField('pValueMantissa', t.DoubleType(), True),
    t.StructField('pValueExponent', t.IntegerType(), True),
    t.StructField('oddsRatio', t.DoubleType(), True),
    t.StructField('ConfidenceIntervalLower', t.DoubleType(), True),
    t.StructField('ConfidenceIntervalUpper', t.DoubleType(), True),
    t.StructField('beta', t.DoubleType(), True),
    t.StructField('sex', t.StringType(), True),
    t.StructField('ancestry', t.StringType(), True),
    t.StructField('ancestryId', t.StringType(), True),
    t.StructField('cohortId', t.StringType(), True),
    t.StructField('studySampleSize', t.IntegerType(), True),
    t.StructField('studyCases', t.IntegerType(), True),
    t.StructField('studyCasesWithQualifyingVariants', t.IntegerType(), True),
    t.StructField('allelicRequirements', t.StringType(), True),
    t.StructField('studyId', t.StringType(), True),
    t.StructField('statisticalMethod', t.StringType(), True),
    t.StructField('statisticalMethodOverview', t.StringType(), True),
    t.StructField('literature', t.StringType(), True),
    t.StructField('url', t.StringType(), True),
])


def gene_burden(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    spark = Session(app_name='gene_burden', properties=properties)

    logger.info(f'load data from {source}')
    disease_mappings_df = spark.load_data(source['disease_mappings'], format='csv', header=True, sep='\t')
    az_binary_df = spark.load_data(source['az_binary'])
    az_quantitative_df = spark.load_data(source['az_quantitative'])
    az_genes_df = spark.load_data(source['az_genes'], format='csv', header=False, schema='gene STRING, link STRING')
    az_phenotypes_df = spark.load_data(
        source['az_phenotypes'], format='csv', header=False, schema='diseaseFromSource STRING, url STRING'
    )
    finngen_manifest_df = spark.load_data(source['finngen_phenotypes'], format='json')
    finngen_df = spark.load_data(source['finngen'], format='csv', header=True, sep='\t')
    finngen_version = settings.get('finngen_release')
    genebass_df = spark.load_data(source['genebass'])
    cvdi_associations_df = pd.read_excel(
        source['cvdi'],
        sheet_name='ST6',
        skiprows=1,
        header=[0, 1, 2],
        skipfooter=1,
    )[['phenotype', 'Gene ID Ensembl', 'Gene', 'ALL ancestry']]
    cvdi_p_value_cutoff_df = pd.read_excel(
        source['cvdi'],
        sheet_name='ST3',
        skiprows=1,
        header=[0, 1],
        skipfooter=1,
    )
    burden_curation = spark.load_data(
        source['curated_studies'], header=True, sep='\t', format='csv', schema=CURATION_SCHEMA
    )

    disease_mappings_df = disease_mappings_df.select(
        f.col('PROPERTY_VALUE').alias('diseaseFromSource'),
        f.element_at(f.split(f.col('SEMANTIC_TAG'), '/'), -1).alias('diseaseFromSourceMappedId'),
    )
    burden_evidence_sets = [
        process_az_gene_burden(az_binary_df, az_quantitative_df, az_genes_df, az_phenotypes_df, disease_mappings_df),
        process_gene_burden_curation(burden_curation),
        process_genebass_gene_burden(genebass_df, disease_mappings_df),
        process_finngen_gene_burden(finngen_df, finngen_manifest_df, disease_mappings_df, finngen_version),
        process_cvdi_gene_burden(spark, cvdi_associations_df, cvdi_p_value_cutoff_df),
    ]
    union_by_diff_schema = partial(DataFrame.unionByName, allowMissingColumns=True)
    evd_df = reduce(union_by_diff_schema, burden_evidence_sets).distinct()
    evd_df.write.parquet(destination, mode='overwrite')


def process_cvdi_gene_burden(
    spark: Session,
    cvdi_associations_df: pd.DataFrame,
    cvdi_p_value_cutoff_df: pd.DataFrame,
) -> DataFrame:
    """This module extracts and processes target/disease evidence from the raw Broad CVDI Human Disease Portal.

    We use:
    - Table 6 as the main reference for significant target/disease associations. We filter out the ancestry specific
        associations because the study doesn't report any ancestry specific genes.
    - Table 3 contains the P cutoff for each of the methods. In the publication,
        they define statistical significance based on FDR < 0.01.
    """
    cvdi_method_desc = {
        'LOF + missense0.8 (MAF<0.1%)': (
            'Mixed-effects test carried out with LOF and predicted-deleterious missense variants '
            '(missense score > 0.8) with a MAF smaller than 0.1%.'
        ),
        'LOF + missense0.5 (MAF<0.001%)': (
            'Mixed-effects test carried out with LOF and predicted-deleterious missense variants '
            '(missense score > 0.5) with a MAF smaller than 0.001%.'
        ),
        'LOF (MAF<0.1%)': 'Mixed-effects test carried out with LOF variants with a MAF smaller than 0.1%.',
        'LOF + missense0.5 (MAF<0.1%)': (
            'Mixed-effects test carried out with LOF and predicted-deleterious missense variants '
            '(missense score > 0.5) with a MAF smaller than 0.1%.'
        ),
        'Cauchy': 'Combined test after combining mask-specific using Cauchy distribution.',
    }
    cvdi_pub = '39210047'

    def _process_cvdi_associations(cvdi_associations_df: pd.DataFrame) -> pd.DataFrame:
        """Parse Table 6 multiindex dataframe.

        Every column represents a different method. We slice the df to parse associations for each method, then merge.
        """

        def _slice_dataframe(df: pd.DataFrame, method: str) -> pd.DataFrame:
            """Slice a dataframe to extract the columns corresponding to a specific method.

            Args:
                df: DataFrame with columns indexed by method
                method: Method to extract

            Returns:
                df: DataFrame with columns corresponding to the specified method and a flatten structure of columns
            """
            df = df.xs(method, level=1, axis=1)
            df.columns = df.columns.get_level_values(1)
            return df.assign(method_name=method)

        # Get the list of statistical models parsed in the column hierarchy
        statistical_models = list({
            index_level_1 for (index_level_1, _) in cvdi_associations_df['ALL ancestry'].columns
        })
        # Append index columns to each slice

        index_cols = ['phenotype', 'Gene ID Ensembl', 'Gene']
        index_dataframe = cvdi_associations_df[index_cols]
        index_dataframe.columns = index_dataframe.columns.get_level_values(0)
        return pd.concat([
            pd.concat([index_dataframe, _slice_dataframe(cvdi_associations_df, model)], axis=1)
            for model in statistical_models
        ])

    def _process_cvdi_pvalues(cvdi_p_value_cutoff_df: pd.DataFrame) -> pd.DataFrame:
        """Parse Table 3 multiindex dataframe.

        Every column represents a different method
        """
        p_cutoff_table = cvdi_p_value_cutoff_df.drop(['AoU', 'UKB', 'META (no correction)', 'MGB'], axis=1)
        p_cutoff_table.columns = p_cutoff_table.columns.get_level_values(1)
        # Forward fill the 'Significance cutoff' values to fill the empty cells
        p_cutoff_table['Significance cutoff'] = p_cutoff_table['Significance cutoff'].ffill()
        return p_cutoff_table[p_cutoff_table['Significance cutoff'] == 'FDR1%'].filter(['Mask', 'P cutoff'])

    # Flatten MultiIndex columns before merging to avoid pandas merge errors
    cvdi_associations_df = _process_cvdi_associations(cvdi_associations_df)
    cvdi_p_value_cutoff_df = _process_cvdi_pvalues(cvdi_p_value_cutoff_df)

    associations_df = (
        cvdi_associations_df.merge(cvdi_p_value_cutoff_df, left_on='method_name', right_on='Mask')
        .drop('Mask', axis=1)
        .drop_duplicates()
        # Dropping rows with no odds ratio or invalid values:
        .astype({'OR [95%CI]': str})
        .dropna(subset=['OR [95%CI]'])
        # Also filter out rows where Gene ID Ensembl contains non-Ensembl values
        .query("`Gene ID Ensembl` != 'Gene ID Ensembl' and `Gene ID Ensembl`.notna()")
    )

    return (
        spark.spark.createDataFrame(
            associations_df,
        )
        .withColumn(
            'resourceScore',
            f.when(f.col('method_name') == f.lit('Cauchy'), f.col('Cauchy P-value')).otherwise(f.col('Meta P-value')),
        )
        # Filter out non significant associations (thresholds vary depending on the mask)
        .filter(f.col('resourceScore') <= f.col('P cutoff'))
        .withColumn('statisticalMethodOverview', f.col('method_name'))
        .replace(to_replace=cvdi_method_desc, subset=['statisticalMethodOverview'])
        .select(
            f.lit('gene_burden').alias('datasourceId'),
            f.lit('genetic_association').alias('datatypeId'),
            f.lit('CVDI Human Disease Portal').alias('projectId'),
            f.lit('UK Biobank 450k/All of Us/MGB').alias('cohortId'),
            f.translate('phenotype', '_', ' ').alias('diseaseFromSource'),
            f.col('Gene ID Ensembl').alias('targetFromSourceId'),
            f.when(f.col('cMAC') == 0, f.lit(None))
            .otherwise(f.col('cMAC'))
            .try_cast('int')
            .alias('studyCasesWithQualifyingVariants'),
            f.lit(748879).alias('studySampleSize'),
            f.col('resourceScore'),
            f.regexp_extract(f.col('OR [95%CI]'), r'(\d+\.\d+)', 1).try_cast('double').alias('oddsRatio'),
            f.regexp_extract(f.col('OR [95%CI]'), r'\[(\d+\.\d+)', 1)
            .try_cast('double')
            .alias('oddsRatioConfidenceIntervalLower'),
            f.regexp_extract(f.col('OR [95%CI]'), r'; (\d+\.\d+)\]', 1)
            .try_cast('double')
            .alias('oddsRatioConfidenceIntervalUpper'),
            f.col('method_name').alias('statisticalMethod'),
            f.col('statisticalMethodOverview'),
            f.array(
                f.struct(
                    f.lit('Broad CVDI Human Disease Portal').alias('niceName'),
                    f.concat(
                        f.lit(
                            'https://hugeamp.org:8000/research.html?ancestry=mixed&cohort=UKB_450k_AoU_250k_MGB_53k_META_overlapcorrected&file=600Traits.csv&gene='
                        ),
                        f.col('Gene'),
                        f.lit('&pageid=600_traits_app'),
                    ).alias('url'),
                )
            ).alias('urls'),
            f.array(f.lit(cvdi_pub)).alias('literature'),
            (f.log10(f.col('resourceScore')).try_cast('int') - f.lit(1)).alias('pValueExponent'),
            f.round(
                f.col('resourceScore') / f.pow(f.lit(10), (f.log10(f.col('resourceScore')).try_cast('int') - f.lit(1))),
                3,
            ).alias('pValueMantissa'),
        )
        .distinct()
    )


def apply_bonferroni_correction(n_tests: int) -> float:
    """Multiple test correction based on the number of tests.

    Args:
        n_tests (int): Number of hypotheses testes assuming they are independent

    Returns:
        float: new statistical significance level
    """
    return 0.05 / n_tests


def process_finngen_gene_burden(
    finngen_df: DataFrame, finngen_manifest_df: DataFrame, disease_mappings_df: DataFrame, finngen_release: str
) -> DataFrame:
    """Process Finngen's loss of function burden results."""
    finngen_pub = '36653562'

    finngen_df = (
        finngen_df
        # Bring description of Finngen's endpoint from manifest
        .join(finngen_manifest_df.selectExpr('phenocode as PHENO', 'phenostring as diseaseFromSource'), 'PHENO', 'left')
        .join(
            disease_mappings_df,
            on='diseaseFromSource',
            how='left',
        )
        .select(
            f.lit('gene_burden').alias('datasourceId'),
            f.lit('finnish').alias('ancestry'),
            f.lit('HANCESTRO_0321').alias('ancestryId'),
            f.col('BETA').try_cast('double').alias('beta'),
            (f.col('BETA') - f.col('SE')).try_cast('double').alias('betaConfidenceIntervalLower'),
            (f.col('BETA') + f.col('SE')).try_cast('double').alias('betaConfidenceIntervalUpper'),
            f.lit('FinnGen R12').alias('cohortId'),
            f.lit('genetic_association').alias('datatypeId'),
            f.col('diseaseFromSource'),
            f.col('PHENO').alias('diseaseFromSourceId'),
            f.col('diseaseFromSourceMappedId'),
            f.lit('FinnGen').alias('projectId'),
            f.array(f.lit(finngen_pub)).alias('literature'),
            (10 ** -f.col('LOG10P').try_cast('double')).alias('resourceScore'),
            (f.log10(10 ** -f.col('LOG10P').try_cast('double')).try_cast('int') - f.lit(1)).alias('pValueExponent'),
            f.round(
                (10 ** -f.col('LOG10P').try_cast('double'))
                / f.pow(f.lit(10), (f.log10(10 ** -f.col('LOG10P').try_cast('double')).try_cast('int') - f.lit(1))),
                3,
            ).alias('pValueMantissa'),
            f.lit(finngen_release).alias('releaseVersion'),
            f.lit('LoF burden').alias('statisticalMethod'),
            f.lit('Burden test carried out with LoF variants with MAF smaller than 1%.').alias(
                'statisticalMethodOverview'
            ),
            f.lit(500348).alias('studySampleSize'),
            f.split(f.col('ID'), r'\.')[0].alias('targetFromSourceId'),
        )
    )
    gene_count = finngen_df.select('targetFromSourceId').distinct().count()
    statistical_significance = apply_bonferroni_correction(gene_count)
    return finngen_df.filter(f.col('resourceScore') <= statistical_significance).distinct()


def process_gene_burden_curation(burden_curation_df: DataFrame) -> DataFrame:
    """Process manual gene burden evidence."""
    return burden_curation_df.select(
        f.lit('gene_burden').alias('datasourceId'),
        f.lit('genetic_association').alias('datatypeId'),
        'projectId',
        'targetFromSourceId',
        'diseaseFromSource',
        'diseaseFromSourceMappedId',
        'resourceScore',
        'pValueMantissa',
        'pValueExponent',
        'oddsRatio',
        f.when(f.col('oddsRatio').isNotNull(), f.col('ConfidenceIntervalLower')).alias(
            'oddsRatioConfidenceIntervalLower'
        ),
        f.when(f.col('oddsRatio').isNotNull(), f.col('ConfidenceIntervalUpper')).alias(
            'oddsRatioConfidenceIntervalUpper'
        ),
        'beta',
        f.when(f.col('beta').isNotNull(), f.col('ConfidenceIntervalLower')).alias('betaConfidenceIntervalLower'),
        f.when(f.col('beta').isNotNull(), f.col('ConfidenceIntervalUpper')).alias('betaConfidenceIntervalUpper'),
        f.split(f.col('sex'), ', ').alias('sex'),
        'ancestry',
        'ancestryId',
        'cohortId',
        'studySampleSize',
        'studyCases',
        'studyCasesWithQualifyingVariants',
        f.when(f.col('allelicRequirements').isNotNull(), f.array(f.col('allelicRequirements'))).alias(
            'allelicRequirements'
        ),
        'statisticalMethod',
        'statisticalMethodOverview',
        f.array(f.col('literature')).alias('literature'),
    ).distinct()


def process_az_gene_burden(
    az_binary_df: DataFrame,
    az_quantitative_df: DataFrame,
    az_genes_links_df: DataFrame,
    az_phenotypes_links_df: DataFrame,
    disease_mappings_df: DataFrame,
) -> DataFrame:
    """Process AZ gene burden data matching the original implementation."""

    def _get_az_release_version(gene_links: DataFrame) -> str:
        """Extract the release version from the gene links file."""
        return (
            gene_links.select(
                f.regexp_extract(f.col('link'), r'https://azphewas.com/geneView/([^/]+)/', 1).alias('extracted_hash')
            )
            .limit(1)
            .collect()[0]['extracted_hash']
        )

    az_method_desc = {
        'ptv': 'Burden test carried out with PTVs with a MAF smaller than 0.1%.',
        'ptv5pcnt': 'Burden test carried out with PTVs with a MAF smaller than 5%.',
        'UR': 'Burden test carried out with ultra rare damaging variants (MAF ≈ 0%).',
        'URmtr': 'Burden test carried out with MTR-informed ultra rare damaging variants (MAF ≈ 0%).',
        'raredmg': 'Burden test carried out with rare missense variants with a MAF smaller than 0.005%.',
        'raredmgmtr': (
            'Burden test carried out with MTR-informed rare missense variants with a MAF smaller than 0.005%.'
        ),
        'flexdmg': 'Burden test carried out with damaging variants with a MAF smaller than 0.01%.',
        'flexnonsyn': 'Burden test carried out with non synonymous variants with a MAF smaller than 0.01%.',
        'flexnonsynmtr': (
            'Burden test carried out with MTR-informed non synonymous variants with a MAF smaller than 0.01%.'
        ),
        'ptvraredmg': 'Burden test carried out with PTV or rare missense variants.',
        'rec': 'Burden test carried out with non-synonymous recessive variants with a MAF smaller than 1%.',
        'syn': 'Burden test carried out with synonymous variants.',
    }
    # Load and combine binary and quantitative data - following original logic exactly
    az_phewas_df = (
        az_binary_df
        # Renaming of some columns to match schemas of both binary and quantitative evidence
        .withColumnRenamed('BinOddsRatioLCI', 'LCI')
        .withColumnRenamed('BinOddsRatioUCI', 'UCI')
        .withColumnRenamed('BinNcases', 'nCases')
        .withColumnRenamed('BinQVcases', 'nCasesQV')
        .withColumnRenamed('BinNcontrols', 'nControls')
        # Combine binary and quantitative evidence into one dataframe
        .unionByName(
            az_quantitative_df.withColumn('nCases', f.col('nSamples')).withColumnRenamed('YesQV', 'nCasesQV'),
            allowMissingColumns=True,
        )
        .withColumn('pValue', f.col('pValue').cast('double'))
        .filter(f.col('pValue') <= 1e-7)
        .distinct()
        .repartition(20)
        .persist()
    )

    # WARNING: There are some associations with a p-value of 0.0 in the AstraZeneca PheWAS Portal.
    # This is a bug we still have to ellucidate and it might be due to a float overflow.
    # These evidence need to be manually corrected in order not to lose them and for them to pass validation
    # As an interim solution, their p value will equal to the minimum in the evidence set.
    logger.warning(f'There are {az_phewas_df.filter(f.col("pValue") == 0.0).count()} evidence with a p-value of 0.0.')
    minimum_pvalue = az_phewas_df.filter(f.col('pValue') > 0.0).agg({'pValue': 'min'}).collect()[0]['min(pValue)']
    az_phewas_df = az_phewas_df.withColumn(
        'pValue',
        f.when(f.col('pValue') == 0.0, f.lit(minimum_pvalue)).otherwise(f.col('pValue')),
    )

    # Transform data according to original logic
    return (
        az_phewas_df.withColumn('datasourceId', f.lit('gene_burden'))
        .withColumn('datatypeId', f.lit('genetic_association'))
        .withColumn('literature', f.array(f.lit('34375979')))
        .withColumn('projectId', f.lit('AstraZeneca PheWAS Portal'))
        .withColumn('cohortId', f.lit('UK Biobank 470k'))
        .withColumnRenamed('Gene', 'targetFromSourceId')
        .withColumnRenamed('Phenotype', 'diseaseFromSource')
        .join(
            disease_mappings_df,
            on='diseaseFromSource',
            how='left',
        )
        .withColumn('resourceScore', f.col('pValue'))
        .withColumn('pValueExponent', f.log10(f.col('pValue')).cast('int') - f.lit(1))
        .withColumn(
            'pValueMantissa',
            f.round(f.col('pValue') / f.pow(f.lit(10), f.col('pValueExponent')), 3),
        )
        .withColumn(
            'beta',
            f.when(f.col('Type') == 'Quantitative', f.col('beta')).cast('float'),
        )
        .withColumn(
            'betaConfidenceIntervalLower',
            f.when(f.col('Type') == 'Quantitative', f.col('LCI')).cast('float'),
        )
        .withColumn(
            'betaConfidenceIntervalUpper',
            f.when(f.col('Type') == 'Quantitative', f.col('UCI')).cast('float'),
        )
        .withColumn(
            'oddsRatio',
            f.when(f.col('Type') == 'Binary', f.col('binOddsRatio')).cast('float'),
        )
        .withColumn(
            'oddsRatioConfidenceIntervalLower',
            f.when(f.col('Type') == 'Binary', f.col('LCI')).cast('float'),
        )
        .withColumn(
            'oddsRatioConfidenceIntervalUpper',
            f.when(f.col('Type') == 'Binary', f.col('UCI')).cast('float'),
        )
        .withColumn('ancestry', f.lit('EUR'))
        .withColumn('ancestryId', f.lit('HANCESTRO_0005'))
        .withColumn('studySampleSize', f.col('nSamples').cast('int'))
        .withColumn('studyCases', f.col('nCases').cast('int'))
        .withColumn('studyCasesWithQualifyingVariants', f.col('nCasesQV').cast('int'))
        .withColumnRenamed('CollapsingModel', 'statisticalMethod')
        .withColumn('statisticalMethodOverview', f.col('statisticalMethod'))
        .replace(to_replace=az_method_desc, subset=['statisticalMethodOverview'])
        .withColumn(
            'allelicRequirements',
            f.when(f.col('statisticalMethod') == 'rec', f.array(f.lit('recessive'))).otherwise(
                f.array(f.lit('dominant'))
            ),
        )
        .withColumn('releaseVersion', f.lit(_get_az_release_version(az_genes_links_df)))
        # Add urls to the phenotypes
        .join(az_phenotypes_links_df, on='diseaseFromSource', how='left')
        .withColumn(
            'urls',
            f.array(
                f.struct(
                    f.col('url').alias('url'),
                    f.lit('AstraZeneca PheWAS Portal').alias('niceName'),
                )
            ),
        )
        .select(
            'datasourceId',
            'datatypeId',
            'allelicRequirements',
            'targetFromSourceId',
            'diseaseFromSource',
            'diseaseFromSourceMappedId',
            'pValueMantissa',
            'pValueExponent',
            'beta',
            'betaConfidenceIntervalLower',
            'betaConfidenceIntervalUpper',
            'oddsRatio',
            'oddsRatioConfidenceIntervalLower',
            'oddsRatioConfidenceIntervalUpper',
            'resourceScore',
            'ancestry',
            'ancestryId',
            'literature',
            'projectId',
            'cohortId',
            'releaseVersion',
            'studySampleSize',
            'studyCases',
            'studyCasesWithQualifyingVariants',
            'statisticalMethod',
            'statisticalMethodOverview',
            'urls',
        )
        .distinct()
    )


def process_genebass_gene_burden(genebass_df: DataFrame, disease_mappings_df: DataFrame):
    """Parse Genebass's disease/target evidence.

    Args:
        genebass_df: DataFrame with Genebass's portal data
        disease_mappings_df: DataFrame with curated mapping between disease and EFO IDs

    Returns:
        evd_df: DataFrame with Genebass's data following the t/d evidence schema.
    """
    genebass_pub = '36778668'

    # WARNING: There are some associations with a p-value of 0.0 in Genebass.
    # This is a bug we still have to ellucidate and it might be due to a float overflow.
    # These evidence need to be manually corrected in order not to lose them and for them to pass validation
    # As an interim solution, their p value will equal to the minimum in the evidence set.
    logger.warning(
        f'There are {genebass_df.filter(f.col("Pvalue_Burden") == 0.0).count()} evidence with a p-value of 0.0.'
    )
    minimum_pvalue = (
        genebass_df.filter(f.col('Pvalue_Burden') > 0.0)
        .agg({'Pvalue_Burden': 'min'})
        .collect()[0]['min(Pvalue_Burden)']
    )
    genebass_df = genebass_df.withColumn(
        'Pvalue_Burden',
        f.when(f.col('Pvalue_Burden') == 0.0, f.lit(minimum_pvalue)).otherwise(f.col('Pvalue_Burden')),
    )

    return (
        genebass_df.filter(f.col('Pvalue_Burden') <= 6.7e-7)
        .filter(f.col('trait_type') != 'categorical')
        .select(
            'gene_id',
            'annotation',
            'n_cases',
            'n_controls',
            'trait_type',
            'phenocode',
            'description',
            'Pvalue_Burden',
            'BETA_Burden',
            'SE_Burden',
        )
        .distinct()
        .withColumnRenamed('description', 'diseaseFromSource')
        .join(
            disease_mappings_df,
            on='diseaseFromSource',
            how='left',
        )
        .select(
            f.lit('gene_burden').alias('datasourceId'),
            f.lit('genetic_association').alias('datatypeId'),
            f.col('gene_id').alias('targetFromSourceId'),
            f.col('diseaseFromSource'),
            f.col('phenocode').alias('diseaseFromSourceId'),
            f.col('diseaseFromSourceMappedId'),
            f.round(
                f.col('Pvalue_Burden') / f.pow(f.lit(10), (f.log10(f.col('Pvalue_Burden')).try_cast('int') - f.lit(1))),
                3,
            ).alias('pValueMantissa'),
            (f.log10(f.col('Pvalue_Burden')).try_cast('int') - f.lit(1)).alias('pValueExponent'),
            f.col('BETA_Burden').alias('beta'),
            (f.col('BETA_Burden') - f.col('SE_Burden')).alias('betaConfidenceIntervalLower'),
            (f.col('BETA_Burden') + f.col('SE_Burden')).alias('betaConfidenceIntervalUpper'),
            f.col('Pvalue_Burden').alias('resourceScore'),
            f.lit('EUR').alias('ancestry'),
            f.lit('HANCESTRO_0009').alias('ancestryId'),
            f.lit('Genebass').alias('projectId'),
            f.lit('UK Biobank 450k').alias('cohortId'),
            (f.col('n_cases') + f.coalesce('n_controls', f.lit(0))).alias('studySampleSize'),
            f.col('n_cases').alias('studyCases'),
            f.col('annotation').alias('statisticalMethod'),
            f.when(f.col('annotation') == 'pLoF', f.lit('Burden test carried out with rare pLOF variants.'))
            .when(
                f.col('annotation') == 'missense|LC',
                f.lit(
                    'Burden test carried out with rare missense variants including low-confidence pLOF '
                    'and in-frame indels.'
                ),
            )
            .when(
                f.col('annotation') == 'synonymous',
                f.lit('Burden test carried out with rare synonymous variants.'),
            )
            .when(
                f.col('annotation') == 'pLoF|missense|LC',
                f.lit('Burden test carried out with pLOF or missense variants.'),
            )
            .otherwise(f.col('annotation'))
            .alias('statisticalMethodOverview'),
            f.array(f.lit(genebass_pub)).alias('literature'),
        )
        .distinct()
    )
