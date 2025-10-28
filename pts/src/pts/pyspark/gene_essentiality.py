"""Parser to the gene-essentiality dataset."""

from typing import Any

from loguru import logger
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f

from pts.pyspark.common.session import Session


def gene_essentiality(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Loads and processes inputs to generate the Gene Essentiality annotation."""
    spark = Session(app_name='chemical_probes', properties=properties)
    keep_only_essentials = settings.get('keep_only_essentials', True)

    logger.debug(f'loading data from: {source}')
    models_df = spark.load_data(source['models'], format='csv', sep=',', header=True)
    essential_genes_df = spark.load_data(source['essential_genes'], format='csv', sep=',', header=True)
    gene_effect_df = spark.load_data(source['gene_effects'], format='csv', sep=',', header=True)
    expression_df = spark.load_data(source['gene_expression'], format='csv', sep=',', header=True)
    hotspots = spark.load_data(source['mutation_hotspots'], format='csv', sep=',', header=True)
    damaging_mutation = spark.load_data(source['mutation_damaging'], format='csv', sep=',', header=True)
    tissue_mapping = spark.load_data(source['depmap_tissue_mapping'], format='csv', sep=',', header=True)

    depmap_essentials = DepMapEssentiality(
        models_df=models_df,
        essential_genes_df=essential_genes_df,
        gene_effect_df=gene_effect_df,
        expression_df=expression_df,
        hotspots=hotspots,
        damaging_mutation=damaging_mutation,
        tissue_mapping=tissue_mapping,
        keep_only_essentials=keep_only_essentials,
    ).transform()
    logger.info('aggregating into gene essentiality data model')
    aggregated_df = depmap_essentials.aggregate()
    logger.info(f'writing output data to {destination}.')
    aggregated_df.write.parquet(destination, mode='overwrite')


class DepMapEssentiality:
    """Parser for DepMap gene essentiality dataset."""

    def __init__(
        self,
        models_df: DataFrame,
        essential_genes_df: DataFrame,
        gene_effect_df: DataFrame,
        expression_df: DataFrame,
        hotspots: DataFrame,
        damaging_mutation: DataFrame,
        tissue_mapping: DataFrame,
        keep_only_essentials: bool = True,
    ) -> None:
        self.models_df = models_df
        self.essetial_genes_df = essential_genes_df
        self.gene_effect_df = gene_effect_df
        self.expression_df = expression_df
        self.hotspots = hotspots
        self.damaging_mutation = damaging_mutation
        self.tissue_mapping = tissue_mapping
        self.keep_only_essentials = keep_only_essentials

        self.essentials = None

    def transform(self):
        """Transform raw data and join before aggregation."""
        models_df_trans = self._prepare_models()
        essential_genes_df_trans = self._prepare_essential_genes()
        expression_df_trans = self._melt(self.expression_df, 'expression')
        hotspots_trans = self._melt(self.hotspots, 'hotspotMutation').filter(f.col('hotspotMutation') > 0)
        damaging_mutation_trans = self._melt(self.damaging_mutation, 'damagingMutation').filter(
            f.col('damagingMutation') > 0
        )
        # Get a list of effect for each gene/cell line pair:
        gene_effect_df_trans = self._melt(self.gene_effect_df, 'geneEffect')

        # Join all the data together:
        self.essentials = (
            # The melted effect table:
            gene_effect_df_trans
            # Narrowing down the gene effect table to genes that are in the essential gene list:
            .join(essential_genes_df_trans, on='targetSymbol', how='left')
            .repartition('depmapId')
            # Joining with model information:
            .join(models_df_trans, on='depmapId', how='left')
            # Joining expression data:
            .join(expression_df_trans, on=['targetSymbol', 'depmapId'], how='left')
            # Joining damaging mutation data:
            .join(damaging_mutation_trans, on=['targetSymbol', 'depmapId'], how='left')
            # Joining hotspot mutation data:
            .join(hotspots_trans, on=['targetSymbol', 'depmapId'], how='left')
            # Joining tissue mapping:
            .join(self.tissue_mapping, on='tissueFromSource', how='left')
            # Format data:
            .select(
                'targetSymbol',
                'depmapId',
                'cellLineName',
                f.col('modelId').alias('diseaseCellLineId'),
                'diseaseFromSource',
                'tissueId',
                f.coalesce(f.col('tissueName'), f.lit('other')).alias('tissueName'),
                f.when(f.col('damagingMutation').isNotNull(), 'damaging')
                .when(f.col('hotspotMutation').isNotNull(), 'hotspot')
                .alias('mutation'),
                'geneEffect',
                'expression',
                f.coalesce(f.col('isEssential'), f.lit(False)).alias('isEssential'),
            )
            # Dropping rows with missing gene effect.
            # This can happen when there's no data for a gene in a given cell line:
            .filter(f.col('geneEffect').isNotNull())
            .persist()
        )

        # If only essential genes are needed, drop all other:
        if self.keep_only_essentials:
            self.essentials = self.essentials.filter(f.col('isEssential')).persist()

        return self

    def aggregate(self) -> DataFrame:
        """Returning aggregated view on essentiality.

        Returns:
            DataFrame: data grouped by gene then by tissue.

        Schema:
            root
            |-- targetSymbol: string (nullable = true)
            |-- isEssential: boolean (nullable = true)
            |-- depMapEssentiality: array (nullable = false)
            |    |-- element: struct (containsNull = false)
            |    |    |-- tissueId: string (nullable = true)
            |    |    |-- tissueName: string (nullable = true)
            |    |    |-- screens: array (nullable = false)
            |    |    |    |-- element: struct (containsNull = false)
            |    |    |    |    |-- depmapId: string (nullable = true)
            |    |    |    |    |-- cellLineName: string (nullable = true)
            |    |    |    |    |-- diseaseFromSource: string (nullable = true)
            |    |    |    |    |-- diseaseCellLineId: string (nullable = true)
            |    |    |    |    |-- mutation: string (nullable = true)
            |    |    |    |    |-- geneEffect: float (nullable = true)
            |    |    |    |    |-- expression: float (nullable = true)
        """
        # Return grouped essentiality data:
        return (
            # Aggregating data by gene:
            self.essentials.groupBy(
                'targetSymbol',
                'isEssential',
                'tissueId',
                'tissueName',
            )
            # Aggregating data further by tissue:
            .agg(
                f.collect_set(
                    f.struct(
                        f.col('depmapId').alias('depmapId'),
                        f.col('cellLineName').alias('cellLineName'),
                        f.col('diseaseFromSource').alias('diseaseFromSource'),
                        f.col('diseaseCellLineId').alias('diseaseCellLineId'),
                        f.col('mutation').alias('mutation'),
                        f.col('geneEffect').alias('geneEffect'),
                        f.col('expression').alias('expression'),
                    )
                ).alias('screens')
            )
            .groupBy('targetSymbol', 'isEssential')
            .agg(
                f.collect_set(
                    f.struct(
                        f.col('tissueId').alias('tissueId'),
                        f.col('tissueName').alias('tissueName'),
                        f.col('screens').alias('screens'),
                    )
                ).alias('depMapEssentiality')
            )
            .persist()
        )

    def get_stats(self) -> None:
        """Print statistics on the essentiality dataset."""
        logger.info(f'Number of entries: {self.essentials.count()}')
        logger.info(f'Number of essential genes: {self.essentials.select("targetSymbol").distinct().count()}')
        logger.info(f'Number of unique diseases: {self.essentials.select("diseaseFromSource").distinct().count()}')

    def _melt(self, df: DataFrame, value_name: str) -> DataFrame:
        """Melt a wide dataframe into a long format.

        Damaging mutation and hotspot mutation tables are provided in a wide format, where each column represents a gene
        and each row is a cell line.
        The values indicate the number of mutations in the gene for the cell line.

        Args:
            df (Dataframe): Dataframe in wide format
            value_name (str): Name of the value column.

        Returns:
            DataFrame: Melted dataframe. Columns: depmapId, targetSymbol, value_name (e.g. geneEffect, expression,
            hotspotMutation, damagingMutation)
        """
        df = (
            df
            # Some files don't have label for the first column:
            .withColumnRenamed('_c0', 'depmapId')
            # Some files have the wrong label:
            .withColumnRenamed('ModelID', 'depmapId')
        )

        # Extracting cell lines:
        genes = df.columns[1:]

        # Generate unpivot expression:
        unpivot_expression = (
            f"""stack({len(genes)}, {', '.join([f"'{c}', `{c}`" for c in genes])}) as (gene_label, {value_name})"""
        )

        # Transform dataset:
        return (
            df.select('depmapId', f.expr(unpivot_expression))
            .select(
                'depmapId',
                self._extract_gene_symbol(f.col('gene_label')),
                f.col(value_name).cast('float'),
            )
            .repartition('depmapId')
        )

    @staticmethod
    def _extract_gene_symbol(gene_col: Column) -> Column:
        """Extract gene symbol from a string, where space separates gene symbol with other components.

        Data example:
            Essentials
            AAMP (14)
            AARS1 (16)

        Args:
            gene_col (Column): Column containing gene symbols.

        Returns:
            Column: gene symbol.
        """
        return f.split(gene_col, ' ').getItem(0).alias('targetSymbol')

    def _prepare_models(self) -> DataFrame:
        """Prepare model data.

        Returns:
            DataFrame: columns: depmapId, cellLineName, modelId, tissueFromSource, diseaseFromSource
        """
        return self.models_df.select(
            f.col('ModelID').alias('depmapId'),
            # If cell line name is provided, it's picked:
            f.when(f.col('CellLineName').isNotNull(), f.col('CellLineName'))
            # When not cell line name, but Cancer Cell Line Enciclopedia name is provided, that's picked:
            .when(f.col('CCLEName').isNotNull(), f.col('CCLEName'))
            # If none of these sources are available, the cell line name is generated from the disease name:
            .otherwise(f.concat(f.col('OncotreePrimaryDisease'), f.lit(' cells')))
            .alias('cellLineName'),
            f.col('SangerModelID').alias('modelId'),
            f.lower(f.col('OncotreeLineage')).alias('tissueFromSource'),
            f.col('OncotreePrimaryDisease').alias('diseaseFromSource'),
        )

    def _prepare_essential_genes(self) -> DataFrame:
        """Prepare essential gene list.

        This method reads the essential gene list table and returns a dataframe with gene symbols and essentiality flag:

        Daata example:
            Essentials
            AAMP (14)
            AARS1 (16)

        Returns:
            DataFrame: columns: targetSymbol, isEssential
        """
        return self.essetial_genes_df.select(
            self._extract_gene_symbol(f.col('Essentials')),
            f.lit(True).alias('isEssential'),
        )
