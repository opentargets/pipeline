from functools import partial, reduce
from typing import Any

import pandas as pd
import pyspark.sql.functions as f
import pyspark.sql.types as t
from loguru import logger
from pyspark.sql.dataframe import DataFrame

from pts.pyspark.common.ontology import add_efo_mapping
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


def _excel_sheet_to_spark(spark: Session, path: str, sheet: str) -> DataFrame:
    """Read an Excel sheet into Spark.

    Every cell is read as a string and empty cells become null, so all typing happens downstream
    """
    pdf = pd.read_excel(path, sheet_name=sheet, dtype=str).where(lambda x: x.notnull(), None)
    schema = t.StructType([t.StructField(column, t.StringType(), True) for column in pdf.columns])
    return spark.spark.createDataFrame(pdf, schema=schema)


def gene_burden(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    spark = Session(app_name='gene_burden', properties=properties)

    ontoma_disease_label_lut = source.pop('ontoma_disease_label_lut')
    ontoma_disease_id_lut = source.pop('ontoma_disease_id_lut', None)

    logger.info(f'load data from {source}')
    az_binary_df = spark.load_data(source['az_binary'])
    az_quantitative_df = spark.load_data(source['az_quantitative'])
    az_genes_df = spark.load_data(source['az_genes'], format='csv', header=False, schema='gene STRING, link STRING')
    az_phenotypes_df = spark.load_data(
        source['az_phenotypes'], format='csv', header=False, schema='diseaseFromSource STRING, url STRING'
    )
    finngen_manifest_df = spark.load_data(source['finngen_phenotypes'], format='json')
    finngen_df = spark.load_data(source['finngen'], format='csv', header=True, sep='\t')
    finngen_version = settings['finngen_release']
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
    gnh_gene_based_df = _excel_sheet_to_spark(spark, source['genes_and_health'], 'ST9')
    gnh_meta_df = _excel_sheet_to_spark(spark, source['genes_and_health'], 'ST13')
    gnh_recessive_df = _excel_sheet_to_spark(spark, source['genes_and_health'], 'ST15')
    brava_s4_df = _excel_sheet_to_spark(spark, source['brava'], 'Table S4')
    brava_s5_df = _excel_sheet_to_spark(spark, source['brava'], 'Table S5')
    brava_s6_df = _excel_sheet_to_spark(spark, source['brava'], 'Table S6')
    brava_s7_df = _excel_sheet_to_spark(spark, source['brava'], 'Table S7')
    brava_s12_df = _excel_sheet_to_spark(spark, source['brava'], 'Table S12')
    brava_s13_df = _excel_sheet_to_spark(spark, source['brava'], 'Table S13')
    brava_s14_df = _excel_sheet_to_spark(spark, source['brava'], 'Table S14')
    brava_s15_df = _excel_sheet_to_spark(spark, source['brava'], 'Table S15')
    burden_curation = spark.load_data(
        source['curated_studies'], header=True, sep='\t', format='csv', schema=CURATION_SCHEMA
    )

    burden_evidence_sets = [
        process_az_gene_burden(az_binary_df, az_quantitative_df, az_genes_df, az_phenotypes_df),
        process_gene_burden_curation(burden_curation),
        process_genebass_gene_burden(genebass_df),
        process_finngen_gene_burden(finngen_df, finngen_manifest_df, finngen_version),
        process_cvdi_gene_burden(spark, cvdi_associations_df, cvdi_p_value_cutoff_df),
        process_genes_and_health_gene_burden(st9_df=gnh_gene_based_df, st13_df=gnh_meta_df, st15_df=gnh_recessive_df),
        process_brava_gene_burden(
            s4_df=brava_s4_df,
            s5_df=brava_s5_df,
            s6_df=brava_s6_df,
            s7_df=brava_s7_df,
            s12_df=brava_s12_df,
            s13_df=brava_s13_df,
            s14_df=brava_s14_df,
            s15_df=brava_s15_df,
        ),
    ]
    union_by_diff_schema = partial(DataFrame.unionByName, allowMissingColumns=True)
    evd_df = reduce(union_by_diff_schema, burden_evidence_sets).distinct()

    mapped_evd_df = add_efo_mapping(
        spark=spark.spark,
        evidence_df=evd_df,
        disease_label_lut_path=ontoma_disease_label_lut,
        disease_id_lut_path=ontoma_disease_id_lut,
    )

    if 'curatedDiseaseFromSourceMappedId' in mapped_evd_df.columns:
        mapped_evd_df = mapped_evd_df.withColumn(
            'diseaseFromSourceMappedId',
            f.coalesce('diseaseFromSourceMappedId', 'curatedDiseaseFromSourceMappedId'),
        ).drop('curatedDiseaseFromSourceMappedId')

    mapped_evd_df.write.parquet(destination, mode='overwrite')


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
        cvdi_associations_df
        .merge(cvdi_p_value_cutoff_df, left_on='method_name', right_on='Mask')
        .drop('Mask', axis=1)
        .drop_duplicates()
        # Dropping rows with no odds ratio or invalid values:
        .astype({'OR [95%CI]': str})
        .dropna(subset=['OR [95%CI]'])
        # Also filter out rows where Gene ID Ensembl contains non-Ensembl values
        .query("`Gene ID Ensembl` != 'Gene ID Ensembl' and `Gene ID Ensembl`.notna()")
    )

    return (
        spark.spark
        .createDataFrame(
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
            f.lit(748879).alias('studySampleSize'),
            f.col('resourceScore'),
            f.col('cMAC').cast(t.IntegerType()).alias('studyCasesWithQualifyingVariants'),
            f.regexp_extract(f.col('OR [95%CI]'), r'(\d+\.\d+)', 1).cast('double').alias('oddsRatio'),
            f
            .regexp_extract(f.col('OR [95%CI]'), r'\[(\d+\.\d+)', 1)
            .cast('double')
            .alias('oddsRatioConfidenceIntervalLower'),
            f
            .regexp_extract(f.col('OR [95%CI]'), r'; (\d+\.\d+)\]', 1)
            .cast('double')
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
            (f.log10(f.col('resourceScore')).cast('int') - f.lit(1)).alias('pValueExponent'),
            f.round(
                f.col('resourceScore') / f.pow(f.lit(10), (f.log10(f.col('resourceScore')).cast('int') - f.lit(1))),
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


GNH_VARIANT_MASK_DESC = {
    # As defined in ST21.
    'A': 'high-confidence LoF variants',
    'B': 'deleterious missense variants',
    'C': 'missense variants',
    'D': 'synonymous variants',
}
GNH_TEST_DESC = {
    # regenie test to collapsing analysis type (additive ExWAS only, ST9).
    'ADD': 'Burden',
    'ADD-SKAT': 'SKAT',
    'ADD-SKATO': 'SKAT-O',
}
GNH_MAF_DESC = {
    '0.01': 'with a MAF smaller than 1%',
    '0.001': 'with a MAF smaller than 0.1%',
    '0.0001': 'with a MAF smaller than 0.01%',
    'singleton': 'restricted to singletons',
}


def _map_col(mapping: dict[str, str]) -> Any:
    """Build a Spark map literal from a python dict for column-value lookups."""
    return f.create_map([f.lit(x) for pair in mapping.items() for x in pair])


def _gnh_is_quantitative(qt_col: Any) -> Any:
    """Flag quantitative traits. QT is read inconsistently across sheets (float 1.0 or boolean True).

    Coalesced to a non-null boolean so its negation (used for binary traits) never evaluates to null.
    """
    return f.coalesce(f.lower(qt_col.cast('string')).isin('true', '1', '1.0'), f.lit(False))


def _gnh_method_overview(collapsing: Any, variant_mask_col: Any, freq_col: Any, suffix: str) -> Any:
    """Compose 'collapsing test carried out with <variants> <MAF clause><suffix>'."""
    return f.concat(
        collapsing,
        f.lit(' test carried out with '),
        _map_col(GNH_VARIANT_MASK_DESC)[variant_mask_col],
        f.lit(' '),
        _map_col(GNH_MAF_DESC)[f.lower(freq_col.cast('string'))],
        f.lit(suffix),
    )


def _process_gnh_additive(st9_df: DataFrame) -> DataFrame:
    """Normalise ST9 additive ExWAS gene-based tests (Genes & Health only, South Asian)."""
    return st9_df.select(
        f.col('Gene ID').alias('targetFromSourceId'),
        f.col('Phenotype').alias('diseaseFromSource'),
        f.col('EFO ID').alias('diseaseFromSourceId'),
        # Rename curated disease ID to avoid column name conflict with EFO mapping
        f.col('EFO ID').alias('curatedDiseaseFromSourceMappedId'),
        f.pow(f.lit(10), -f.col('LOG10P').cast('double')).alias('pValue'),
        f.col('BETA').cast('double').alias('effect'),
        f.col('SE').cast('double').alias('standardError'),
        _gnh_is_quantitative(f.col('QT')).alias('isQuantitative'),
        f.col('N').cast('int').alias('studySampleSize'),
        f.lit(None).cast('int').alias('studyCases'),
        f.lit(None).cast('int').alias('studyCasesWithQualifyingVariants'),
        f.lit('Pakistani and Bangladeshi').alias('ancestry'),
        f.lit('HANCESTRO_0006').alias('ancestryId'),
        f.lit('dominant').alias('allelicRequirements'),
        f.lit('Genes & Health').alias('cohortId'),
        f.concat_ws('_', f.col('TEST'), f.col('Mask')).alias('statisticalMethod'),
        _gnh_method_overview(_map_col(GNH_TEST_DESC)[f.col('TEST')], f.col('Variant Mask'), f.col('Freq'), '.').alias(
            'statisticalMethodOverview'
        ),
    )


def _process_gnh_meta(st13_df: DataFrame) -> DataFrame:
    """Normalise ST13 meta-analysis gene-based tests (Genes & Health + UK Biobank, burden only).

    Ancestry is left null as this is a mixed-ancestry meta-analysis. Some LOG10P underflow to inf
    (p==0); those are corrected with the minimum-p replacement in the shared formatter.
    """
    return st13_df.select(
        f.col('Gene ID').alias('targetFromSourceId'),
        f.col('Phenotype').alias('diseaseFromSource'),
        f.col('EFO ID').alias('diseaseFromSourceId'),
        # Rename curated disease ID to avoid column name conflict with EFO mapping
        f.col('EFO ID').alias('curatedDiseaseFromSourceMappedId'),
        f.pow(f.lit(10), -f.col('LOG10P').cast('double')).alias('pValue'),
        f.col('BETA').cast('double').alias('effect'),
        f.col('SE').cast('double').alias('standardError'),
        _gnh_is_quantitative(f.col('QT')).alias('isQuantitative'),
        (f.col('N UKB').cast('int') + f.col('N G&H').cast('int')).alias('studySampleSize'),
        f.lit(None).cast('int').alias('studyCases'),
        f.lit(None).cast('int').alias('studyCasesWithQualifyingVariants'),
        f.lit(None).cast('string').alias('ancestry'),
        f.lit(None).cast('string').alias('ancestryId'),
        f.lit('dominant').alias('allelicRequirements'),
        f.lit('Genes & Health + UK Biobank').alias('cohortId'),
        f.concat(f.lit('META.'), f.col('Mask')).alias('statisticalMethod'),
        _gnh_method_overview(
            f.lit('Burden'),
            f.col('Variant Mask'),
            f.col('Freq'),
            ' meta-analysed between Genes & Health and UK Biobank.',
        ).alias('statisticalMethodOverview'),
    )


def _process_gnh_recessive(st15_df: DataFrame) -> DataFrame:
    """Normalise ST15 recessive burden tests, keeping only significant biallelic pLoF/pDM associations.

    Drops the synonymous control mask and the suggestive tier. No standard error is reported, so no
    confidence intervals can be derived. Case counts read as 'na' for quantitative traits cast to null.
    """
    return st15_df.filter(
        (f.col('Significant REC P-value') == 'Significant') & (f.col('Variant consequence') == 'pLoF_pDM')
    ).select(
        f.col('Gene ID').alias('targetFromSourceId'),
        f.col('Phenotype').alias('diseaseFromSource'),
        f.col('EFO ID').alias('diseaseFromSourceId'),
        # Rename curated disease ID to avoid column name conflict with EFO mapping
        f.col('EFO ID').alias('curatedDiseaseFromSourceMappedId'),
        f.col('Recessive p-value').cast('double').alias('pValue'),
        f.col('Recessive log(OR)').cast('double').alias('effect'),
        f.lit(None).cast('double').alias('standardError'),
        _gnh_is_quantitative(f.col('QT')).alias('isQuantitative'),
        f.col('Number for ExWAS').cast('int').alias('studySampleSize'),
        f.col('Number of Cases').cast('int').alias('studyCases'),
        f.col('Biallelic Carriers').cast('int').alias('studyCasesWithQualifyingVariants'),
        f.lit('Pakistani and Bangladeshi').alias('ancestry'),
        f.lit('HANCESTRO_0006').alias('ancestryId'),
        f.lit('recessive').alias('allelicRequirements'),
        f.lit('Genes & Health').alias('cohortId'),
        f.lit('REC.pLoF_pDM').alias('statisticalMethod'),
        f.lit('Recessive burden test carried out with biallelic pLoF and deleterious missense genotypes.').alias(
            'statisticalMethodOverview'
        ),
    )


def process_genes_and_health_gene_burden(
    st9_df: DataFrame,
    st13_df: DataFrame,
    st15_df: DataFrame,
) -> DataFrame:
    """Process gene-based burden evidence from the Genes & Health study (PMID 41896352).

    Combines the three gene-based analyses reported in the supplementary tables (additive ExWAS ST9,
    meta-analysis with UK Biobank ST13, and recessive burden ST15) into the gene burden evidence schema.

    Gene IDs and EFO IDs are already provided by the study. The study-provided EFO ID is carried in
    ``curatedDiseaseFromSourceMappedId`` (mirroring the gene burden curation) so it is not overwritten by the
    OnToma ``diseaseFromSourceMappedId`` column; the pipeline coalesces both downstream.
    """
    gh_pub = '41896352'

    gh_df = reduce(
        DataFrame.unionByName,
        [_process_gnh_additive(st9_df), _process_gnh_meta(st13_df), _process_gnh_recessive(st15_df)],
    )

    # WARNING: some meta-analysis p-values underflow to 0.0 (inf LOG10P). Substitute the minimum non-zero
    # p-value so they pass validation instead of being dropped.
    gh_df = _substitute_zero_pvalues(gh_df, 'pValue', 'Genes & Health')

    # Local column expressions reused below (p-value exponent, and the effect ± standard error interval).
    p_exponent = f.log10(f.col('pValue')).cast('int') - f.lit(1)
    quantitative = f.col('isQuantitative')
    ci_lower = f.col('effect') - f.col('standardError')
    ci_upper = f.col('effect') + f.col('standardError')

    return gh_df.select(
        f.lit('gene_burden').alias('datasourceId'),
        f.lit('genetic_association').alias('datatypeId'),
        f.lit('Genes & Health').alias('projectId'),
        f.array(f.lit(gh_pub)).alias('literature'),
        'targetFromSourceId',
        'diseaseFromSource',
        'diseaseFromSourceId',
        'curatedDiseaseFromSourceMappedId',
        f.col('pValue').alias('resourceScore'),
        p_exponent.alias('pValueExponent'),
        f.round(f.col('pValue') / f.pow(f.lit(10), p_exponent), 3).alias('pValueMantissa'),
        # Quantitative traits report a beta; binary traits report a log(OR) -> odds ratio = exp(log(OR)).
        f.when(quantitative, f.col('effect')).cast('double').alias('beta'),
        f.when(quantitative, ci_lower).cast('double').alias('betaConfidenceIntervalLower'),
        f.when(quantitative, ci_upper).cast('double').alias('betaConfidenceIntervalUpper'),
        f.when(~quantitative, f.exp(f.col('effect'))).cast('double').alias('oddsRatio'),
        f.when(~quantitative, f.exp(ci_lower)).cast('double').alias('oddsRatioConfidenceIntervalLower'),
        f.when(~quantitative, f.exp(ci_upper)).cast('double').alias('oddsRatioConfidenceIntervalUpper'),
        'ancestry',
        'ancestryId',
        f.array(f.col('allelicRequirements')).alias('allelicRequirements'),
        'cohortId',
        'studySampleSize',
        'studyCases',
        'studyCasesWithQualifyingVariants',
        'statisticalMethod',
        'statisticalMethodOverview',
    ).distinct()


def process_finngen_gene_burden(
    finngen_df: DataFrame, finngen_manifest_df: DataFrame, finngen_release: str
) -> DataFrame:
    """Process Finngen's loss of function burden results."""
    finngen_pub = '36653562'

    finngen_df = (
        finngen_df
        # Bring description of Finngen's endpoint from manifest
        .join(
            finngen_manifest_df.selectExpr('phenocode as PHENO', 'phenostring as diseaseFromSource'), 'PHENO', 'left'
        ).select(
            f.lit('gene_burden').alias('datasourceId'),
            f.lit('finnish').alias('ancestry'),
            f.lit('HANCESTRO_0321').alias('ancestryId'),
            f.col('BETA').cast('double').alias('beta'),
            (f.col('BETA') - f.col('SE')).cast('double').alias('betaConfidenceIntervalLower'),
            (f.col('BETA') + f.col('SE')).cast('double').alias('betaConfidenceIntervalUpper'),
            f.lit('FinnGen R12').alias('cohortId'),
            f.lit('genetic_association').alias('datatypeId'),
            f.col('diseaseFromSource'),
            f.col('PHENO').alias('diseaseFromSourceId'),
            f.lit('FinnGen').alias('projectId'),
            f.array(f.lit(finngen_pub)).alias('literature'),
            (10 ** -f.col('LOG10P').cast('double')).alias('resourceScore'),
            (f.log10(10 ** -f.col('LOG10P').cast('double')).cast('int') - f.lit(1)).alias('pValueExponent'),
            f.round(
                (10 ** -f.col('LOG10P').cast('double'))
                / f.pow(f.lit(10), (f.log10(10 ** -f.col('LOG10P').cast('double')).cast('int') - f.lit(1))),
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
        # Rename curated disease ID to avoid column name conflict with EFO mapping
        f.col('diseaseFromSourceMappedId').alias('curatedDiseaseFromSourceMappedId'),
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
) -> DataFrame:
    """Process AZ gene burden data matching the original implementation."""

    def _get_az_release_version(gene_links: DataFrame) -> str:
        """Extract the release version from the gene links file."""
        return (
            gene_links
            .select(
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
    az_phewas_df = _substitute_zero_pvalues(az_phewas_df, 'pValue', 'AZ')

    # Transform data according to original logic
    return (
        az_phewas_df
        .withColumn('datasourceId', f.lit('gene_burden'))
        .withColumn('datatypeId', f.lit('genetic_association'))
        .withColumn('literature', f.array(f.lit('34375979')))
        .withColumn('projectId', f.lit('AstraZeneca PheWAS Portal'))
        .withColumn('cohortId', f.lit('UK Biobank 470k'))
        .withColumnRenamed('Gene', 'targetFromSourceId')
        .withColumnRenamed('Phenotype', 'diseaseFromSource')
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


def process_genebass_gene_burden(genebass_df: DataFrame):
    """Parse Genebass's disease/target evidence.

    Args:
        genebass_df: DataFrame with Genebass's portal data

    Returns:
        evd_df: DataFrame with Genebass's data following the t/d evidence schema.
    """
    genebass_pub = '36778668'

    # WARNING: There are some associations with a p-value of 0.0 in Genebass.
    # This is a bug we still have to ellucidate and it might be due to a float overflow.
    # These evidence need to be manually corrected in order not to lose them and for them to pass validation
    # As an interim solution, their p value will equal to the minimum in the evidence set.
    genebass_df = _substitute_zero_pvalues(genebass_df, 'Pvalue_Burden', 'Genebass')

    return (
        genebass_df
        .filter(f.col('Pvalue_Burden') <= 6.7e-7)
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
        .select(
            f.lit('gene_burden').alias('datasourceId'),
            f.lit('genetic_association').alias('datatypeId'),
            f.col('gene_id').alias('targetFromSourceId'),
            f.col('diseaseFromSource'),
            f.col('phenocode').alias('diseaseFromSourceId'),
            f.round(
                f.col('Pvalue_Burden') / f.pow(f.lit(10), (f.log10(f.col('Pvalue_Burden')).cast('int') - f.lit(1))),
                3,
            ).alias('pValueMantissa'),
            (f.log10(f.col('Pvalue_Burden')).cast('int') - f.lit(1)).alias('pValueExponent'),
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
            f
            .when(f.col('annotation') == 'pLoF', f.lit('Burden test carried out with rare pLOF variants.'))
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


def _substitute_zero_pvalues(df: DataFrame, pvalue_col: str, label: str) -> DataFrame:
    """Substitute p-values that underflow to 0.0 with the minimum non-zero p-value observed."""
    zero_p = df.filter(f.col(pvalue_col) == 0.0).count()
    if not zero_p:
        return df
    logger.warning(f'There are {zero_p} {label} evidence with a p-value of 0.0.')
    minimum_pvalue = df.filter(f.col(pvalue_col) > 0.0).agg({pvalue_col: 'min'}).collect()[0][f'min({pvalue_col})']
    return df.withColumn(
        pvalue_col, f.when(f.col(pvalue_col) == 0.0, f.lit(minimum_pvalue)).otherwise(f.col(pvalue_col))
    )


def process_brava_gene_burden(
    s4_df: DataFrame,
    s5_df: DataFrame,
    s6_df: DataFrame,
    s7_df: DataFrame,
    s12_df: DataFrame,
    s13_df: DataFrame,
    s14_df: DataFrame,
    s15_df: DataFrame,
) -> DataFrame:
    """Process gene burden evidence from the BRaVa consortium publication (PMID 42238450).

    Two types of results are processsed into the gene burden evidence schema:
    - Granular mask/MAF/ancestry Burden/SKAT/SKAT-O results (S14 quantitative + S15 binary)
    - Cauchy combination results aggregated across masks/MAFs/test class (S12 quantitative + S13 binary)
    """
    brava_pub = '42238450'

    # mask abbrevation for statisticalMethod
    mask_abbrev = {
        'pLoF': 'pLoF',
        'damaging_missense_or_protein_altering': 'DM/PA',
        'pLoF;damaging_missense_or_protein_altering': 'pLOF;DM/PA',
    }
    # mask description for statisticalMethodOverview
    mask_desc = {
        'pLoF': 'pLOF variants',
        'damaging_missense_or_protein_altering': 'damaging missense or protein altering variants',
        'pLoF;damaging_missense_or_protein_altering': 'pLoF or damaging missense/protein-altering variants',
    }
    # max MAF description for statisticalMethodOverview
    maf_desc = {
        '0.001': 'with a MAF smaller than 0.1%',
        '0.0001': 'with a MAF smaller than 0.01%',
    }
    ancestry_label = {
        'EUR': 'European',
        'AFR': 'African',
        'EAS': 'East Asian',
        'AMR': 'Admixed American',
        'SAS': 'Central and South Asian',
        'ALL': 'N/A',
        'non_EUR': 'non-European',
    }
    ancestry_id = {
        'EUR': 'HANCESTRO_0005',
        'AFR': 'HANCESTRO_0010',
        'EAS': 'HANCESTRO_0009',
        'AMR': 'HANCESTRO_0014',
        # 'Central and South Asian' requires mapping to two ids, so it is left null here
    }

    def _brava_sample_size_lookup(
        s4_df: DataFrame, s5_df: DataFrame, s6_df: DataFrame, s7_df: DataFrame
    ) -> DataFrame:
        """Build a (Phenotype ID, Ancestry Group) -> (studySampleSize, studyCases) lookup for BRaVa evidence.

        - 'ALL' rows come from the pooled totals (S6/S7).
        - Ancestry-specific rows are aggregated from the per-biobank tables (S4/S5), summed across biobanks.
        - 'non_EUR' isn't reported directly, so it's derived as the 'ALL' total minus the 'EUR' total per phenotype.
        """
        all_binary = s6_df.select(
            'Phenotype ID',
            f.lit('ALL').alias('Ancestry Group'),
            (f.col('N cases').cast('int') + f.col('N controls').cast('int')).alias('studySampleSize'),
            f.col('N cases').cast('int').alias('studyCases'),
        )
        all_quantitative = s7_df.select(
            'Phenotype ID',
            f.lit('ALL').alias('Ancestry Group'),
            f.col('N').cast('int').alias('studySampleSize'),
            f.lit(None).cast('int').alias('studyCases'),
        )
        ancestry_binary = (
            s4_df.distinct()
            .groupBy('Phenotype ID', f.col('Ancestry').alias('Ancestry Group'))
            .agg(
                (f.sum(f.col('N cases').cast('int')) + f.sum(f.col('N controls').cast('int')))
                .cast('int')
                .alias('studySampleSize'),
                f.sum(f.col('N cases').cast('int')).cast('int').alias('studyCases'),
            )
        )
        ancestry_quantitative = (
            s5_df.distinct()
            .groupBy('Phenotype ID', f.col('Ancestry').alias('Ancestry Group'))
            .agg(f.sum(f.col('N').cast('int')).cast('int').alias('studySampleSize'))
            .withColumn('studyCases', f.lit(None).cast('int'))
        )
        def _derive_non_eur(all_df: DataFrame, ancestry_df: DataFrame) -> DataFrame:
            """Derive 'non_EUR' totals as the 'ALL' total minus the 'EUR' total, per phenotype."""
            return (
                all_df
                .select(
                    'Phenotype ID', f.col('studySampleSize').alias('all_n'), f.col('studyCases').alias('all_cases')
                )
                .join(
                    ancestry_df
                    .filter(f.col('Ancestry Group') == 'EUR')
                    .select(
                        'Phenotype ID',
                        f.col('studySampleSize').alias('eur_n'),
                        f.col('studyCases').alias('eur_cases'),
                    ),
                    'Phenotype ID',
                )
                .select(
                    'Phenotype ID',
                    f.lit('non_EUR').alias('Ancestry Group'),
                    (f.col('all_n') - f.col('eur_n')).alias('studySampleSize'),
                    (f.col('all_cases') - f.col('eur_cases')).alias('studyCases'),
                )
            )

        non_eur_binary = _derive_non_eur(all_binary, ancestry_binary)
        non_eur_quantitative = _derive_non_eur(all_quantitative, ancestry_quantitative)
        return reduce(
            DataFrame.unionByName,
            [
                all_binary,
                all_quantitative,
                ancestry_binary,
                ancestry_quantitative,
                non_eur_binary,
                non_eur_quantitative,
            ],
        ).distinct()

    def _process_brava_granular(df: DataFrame) -> DataFrame:
        """Normalise granular BRaVa results into the gene burden evidence schema.

        Processes Tables S14 (quantitative) or S15 (binary).

        Each row is a gene/phenotype/ancestry/mask/MAF/class.

        Only Burden-class rows carry an effect size (BETA Burden/SE Burden); SKAT/SKAT-O rows, and a handful of
        Burden rows with a missing beta in the source, are kept as p-value-only evidence.
        """
        df = (
            df
            .withColumn('Pvalue', f.col('Pvalue').cast('double'))
            .withColumn('BETA Burden', f.col('BETA Burden').cast('double'))
            .withColumn('SE Burden', f.col('SE Burden').cast('double'))
        )
        df = _substitute_zero_pvalues(df, 'Pvalue', 'BRaVa')

        ancestry_group = f.col('meta analyzed')

        return df.select(
            f.col('Gene ID'),
            f.col('Description'),
            f.col('Phenotype ID'),
            ancestry_group.alias('Ancestry Group'),
            f.col('Pvalue').alias('resourceScore'),
            f.col('BETA Burden').alias('beta'),
            (f.col('BETA Burden') - f.col('SE Burden')).alias('betaConfidenceIntervalLower'),
            (f.col('BETA Burden') + f.col('SE Burden')).alias('betaConfidenceIntervalUpper'),
            # ancestryId and statisticalMethod are among the fields used for constructing the dedup id
            # as ancestryId is null for ALL/non_EUR/SAS (see ancestry_id dict above), ancestry_group
            # has to be added to statisticalMethod, otherwise rows will be mistaken as duplicate rows and dropped
            f.concat_ws(
                '.', f.col('class'), _map_col(mask_abbrev)[f.col('Mask')], f.col('max MAF'), ancestry_group
            ).alias('statisticalMethod'),
            f.concat(
                f.col('class'),
                f.lit(' test carried out with '),
                _map_col(mask_desc)[f.col('Mask')],
                f.lit(' '),
                _map_col(maf_desc)[f.col('max MAF')],
                f.lit(', meta-analyzed across BRaVa biobanks.'),
            ).alias('statisticalMethodOverview'),
        )

    def _process_brava_cauchy(df: DataFrame) -> DataFrame:
        """Normalise cauchy combination BRaVa results into the gene burden evidence schema.

        Processes Tables S12 (quantitative) or S15 (binary).

        Each row is Cauchy-aggregrated per gene/phenotype across masks/MAFs/test class.

        P-value-only evidence (no effect size reported for the Cauchy combination test).
        """
        pvalue_col = 'Min P-value of significant Cauchy associations'
        df = df.withColumn(pvalue_col, f.col(pvalue_col).cast('double'))
        df = _substitute_zero_pvalues(df, pvalue_col, 'BRaVa Cauchy')

        ancestry_group = f.col('Cauchy combination with minimum P-value')

        return df.select(
            f.col('Gene ID'),
            f.col('Description'),
            f.col('Phenotype ID'),
            ancestry_group.alias('Ancestry Group'),
            f.col(pvalue_col).alias('resourceScore'),
            # ancestryId and statisticalMethod are among the fields used for constructing the dedup id
            # as ancestryId is null for ALL/non_EUR/SAS (see ancestry_id dict above), ancestry_group
            # has to be added to statisticalMethod, otherwise rows will be mistaken as duplicate rows and dropped
            f.concat(f.lit('Cauchy.'), ancestry_group).alias('statisticalMethod'),
            f.lit(
                'Cauchy combination test aggregating Burden and SKAT/SKAT-O results across variant masks '
                'and MAF cutoffs.'
            ).alias('statisticalMethodOverview'),
        )

    sample_size_lookup = _brava_sample_size_lookup(s4_df, s5_df, s6_df, s7_df)
    union_by_diff_schema = partial(DataFrame.unionByName, allowMissingColumns=True)

    p_exponent = f.log10(f.col('resourceScore')).cast('int') - f.lit(1)

    brava_df = reduce(
        union_by_diff_schema,
        [
            _process_brava_granular(s14_df),
            _process_brava_granular(s15_df),
            _process_brava_cauchy(s12_df),
            _process_brava_cauchy(s13_df),
        ],
    ).select(
        f.lit('gene_burden').alias('datasourceId'),
        f.lit('genetic_association').alias('datatypeId'),
        f.lit('BRaVa Consortium').alias('projectId'),
        f.lit('BRaVa Consortium').alias('cohortId'),
        f.array(f.lit(brava_pub)).alias('literature'),
        f.col('Gene ID').alias('targetFromSourceId'),
        f.col('Description').alias('diseaseFromSource'),
        f.col('Phenotype ID'),
        f.col('Ancestry Group'),
        _map_col(ancestry_label)[f.col('Ancestry Group')].alias('ancestry'),
        _map_col(ancestry_id)[f.col('Ancestry Group')].alias('ancestryId'),
        'resourceScore',
        p_exponent.alias('pValueExponent'),
        f.round(f.col('resourceScore') / f.pow(f.lit(10), p_exponent), 3).alias('pValueMantissa'),
        'beta',
        'betaConfidenceIntervalLower',
        'betaConfidenceIntervalUpper',
        'statisticalMethod',
        'statisticalMethodOverview',
    )

    return (
        brava_df
        .join(sample_size_lookup, ['Phenotype ID', 'Ancestry Group'], 'left')
        .drop('Phenotype ID', 'Ancestry Group')
        .distinct()
    )
