"""Evidence parser for the Cancer Biomarkers database."""

import re
from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, StringType

from pts.pyspark.common.session import Session

ALTERATIONTYPE2FUNCTIONCSQ: dict = {
    # TODO: Map BIA
    'MUT': 'SO_0001777',  # somatic_variant
    'CNA': 'SO_0001563',  # copy_number_change
    'FUS': 'SO_0001882',  # feature_fusion
    'EXPR': None,
    'BIA': None,
}

DRUGRESPONSE2EFO: dict = {
    'Responsive': 'GO_0042493',  # response to drug
    'Not Responsive': 'EFO_0020002',  # lack of efficacy
    'Resistant': 'EFO_0020001',  # drug resistance
    'Increased Toxicity': 'EFO_0020003',  # drug toxicity
    'Increased Toxicity (Myelosupression)': 'EFO_0007053',  # myelosuppression
    'Increased Toxicity (Ototoxicity)': 'EFO_0006951',  # ototoxicity
    'Increased Toxicity (Hyperbilirubinemia)': 'HP_0002904',  # Hyperbilirubinemia
    'Increased Toxicity (Haemolytic Anemia)': 'EFO_0005558',  # hemolytic anemia
}

GENENAMESOVERRIDE: dict = {
    # Correct gene names to use their approved symbol
    'C15orf55': 'NUTM1',
    'MLL': 'KMT2A',
    'MLL2': 'KMT2D',
}


def cancer_biomarkers(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Loads and processes inputs to generate the Cancer Biomarkers evidence strings."""
    spark = Session(app_name='chemical_probes', properties=properties)

    logger.debug(f'loading data from: {source}')
    biomarkers_df = spark.load_data(path=source['associations'], format='csv', header=True, sep='\t')
    source_df = spark.load_data(path=source['source'], format='json')
    disease_df = spark.load_data(path=source['disease'], format='json')
    drugs_df = spark.load_data(path=source['drugs'], format='parquet').select(
        f.col('id').alias('drugId'), f.col('name').alias('drug')
    )

    # Process inputs to generate evidence strings
    generator = CancerBiomarkersEvidenceGenerator()
    evidence = generator.process_biomarkers(biomarkers_df, source_df, disease_df, drugs_df)

    # Write evidence strings
    # destination is a single path string, not a dictionary
    logger.info(f'writing output data to {destination}.')
    evidence.write.parquet(destination, mode='overwrite')


class CancerBiomarkersEvidenceGenerator:
    def __init__(self):
        self.evidence = None

        self.get_variant_id_udf = f.udf(CancerBiomarkersEvidenceGenerator.get_variant_id, StringType())
        self.zip_alterations_with_type_udf = f.udf(
            CancerBiomarkersEvidenceGenerator.zip_alterations_with_type, ArrayType(ArrayType(StringType()))
        )

    def process_biomarkers(
        self, biomarkers_df: DataFrame, source_df: DataFrame, disease_df: DataFrame, drugs_df: DataFrame
    ) -> DataFrame:
        """The diverse steps to prepare and enrich the input table."""
        biomarkers_enriched = (
            biomarkers_df
            .select(
                'Biomarker',
                'IndividualMutation',
                f.array_distinct(f.split(f.col('Alteration'), ';')).alias('alterations'),
                f.array_distinct(f.split(f.col('Gene'), ';')).alias('gene'),
                f.split(f.col('AlterationType'), ';').alias('alteration_types'),
                f.array_distinct(f.split(f.col('PrimaryTumorTypeFullName'), ';')).alias('tumor_type_full_name'),
                f.array_distinct(f.split(f.col('Drug'), ';|,')).alias('drug'),
                'DrugFullName',
                'Association',
                'gDNA',
                f.array_distinct(f.split(f.col('EvidenceLevel'), ',')).alias('confidence'),
                f.array_distinct(f.split(f.col('Source'), ';')).alias('source'),
            )
            .withColumn('confidence', f.explode(f.col('confidence')))
            .withColumn('tumor_type_full_name', f.explode(f.col('tumor_type_full_name')))
            .withColumn('tumor_type', f.translate(f.col('tumor_type_full_name'), ' -', ''))
            .withColumn('drug', f.explode(f.col('drug')))
            .withColumn('drug', f.translate(f.col('drug'), '[]', ''))
            .withColumn('gene', f.explode(f.col('gene')))
            .replace(to_replace=GENENAMESOVERRIDE, subset=['gene'])
            .withColumn('gene', f.upper(f.col('gene')))
            # At this stage alterations and alteration_types are both arrays
            # Disambiguation when the biomarker consists of multiple alterations is needed
            # This is solved by:
            # 1. Zipping both fields - tmp consists of a list of alteration/type tuples
            # 2. tmp is exploded - tmp consists of the alteration/type tuple
            # 3. alteration & alteration_type columns are overwritten with the elements in the tuple
            .withColumn('tmp', self.zip_alterations_with_type_udf(f.col('alterations'), f.col('alteration_types')))
            .withColumn('tmp', f.explode(f.col('tmp')))
            .withColumn('alteration_type', f.element_at(f.col('tmp'), 2))
            .withColumn(
                'alteration',
                f.when(~f.col('IndividualMutation').isNull(), f.col('IndividualMutation')).otherwise(  # ty:ignore[missing-argument]
                    f.element_at(f.col('tmp'), 1)
                ),
            )
            .drop('tmp')
            # Clean special cases on the alteration string
            .withColumn(
                'alteration',
                f
                .when(
                    f.col('alteration') == 'NRAS:.12.,.13.,.59.,.61.,.117.,.146.',
                    f.col('Biomarker'),  # 'NRAS (12,13,59,61,117,146)'
                )
                .when(
                    # Cleans strings like 'ARAF:.'
                    f.col('alteration').contains(':.'),  # ty:ignore[missing-argument, invalid-argument-type]
                    f.translate(f.col('alteration'), ':.', ''),
                )
                .when(
                    # Fusion genes are described with '__'
                    # biomarker is a cleaner representation when there's one alteration
                    (f.col('alteration').contains('__')) & (~f.col('Biomarker').contains('+')),  # ty:ignore[missing-argument, invalid-argument-type]
                    f.col('Biomarker'),
                )
                .otherwise(f.col('alteration')),
            )
            # Split source into literature and urls
            # literature contains PMIDs
            # urls are enriched from the source table if not a CT
            .withColumn('source', f.explode(f.col('source')))
            .withColumn('source', f.trim(f.regexp_extract(f.col('source'), r'(PMID:\d+)|([\w ]+)', 0).alias('source')))
            .join(source_df.select(f.col('label').alias('niceName'), 'source', 'url'), on='source', how='left')
            .withColumn(
                'literature',
                f.when(f.col('source').startswith('PMID'), f.regexp_extract(f.col('source'), r'(PMID:)(\d+)', 2)),  # ty:ignore[missing-argument, invalid-argument-type]
            )
            .withColumn(
                'urls',
                f.when(
                    f.col('source').startswith('NCT'),  # ty:ignore[missing-argument, invalid-argument-type]
                    f.struct(
                        f.lit('Clinical Trials').alias('niceName'),
                        f.concat(f.lit('https://clinicaltrials.gov/ct2/show/'), f.col('source')).alias('url'),
                    ),
                ).when(
                    (~f.col('source').startswith('PMID')) | (~f.col('source').startswith('NCIT')),  # ty:ignore[missing-argument, invalid-argument-type]
                    f.struct(f.col('niceName'), f.col('url')),
                ),
            )
            # The previous conditional clause creates a struct regardless of
            # whether any condition is met. The empty struct is replaced with null
            .withColumn('urls', f.when(~f.col('urls.niceName').isNull(), f.col('urls')))  # ty:ignore[missing-argument]
            # Enrich data
            .withColumn('functionalConsequenceId', f.col('alteration_type'))
            .replace(to_replace=ALTERATIONTYPE2FUNCTIONCSQ, subset=['functionalConsequenceId'])
            .replace(to_replace=DRUGRESPONSE2EFO, subset=['Association'])
            .join(
                disease_df.select(
                    f.regexp_replace(f.col('name'), '_', '').alias('tumor_type'),
                    f.regexp_extract(f.col('url'), r'[^/]+$', 0).alias('diseaseFromSourceMappedId'),
                ),
                on='tumor_type',
                how='left',
            )
            .withColumn('drug', f.upper(f.col('drug')))
            .withColumn(
                # drug class is coalesced when the precise name of the medicine is not provided
                'drug',
                f.when(f.col('drug').isNull() | (f.length(f.col('drug')) == 0), f.col('DrugFullName')).otherwise(  # ty:ignore[missing-argument]
                    f.col('drug')
                ),
            )
            .join(drugs_df, on='drug', how='left')
            .withColumn('drug', f.initcap(f.col('drug')))
            # Translate variantId
            .withColumn('variantId', f.when(~f.col('gDNA').isNull(), self.get_variant_id_udf(f.col('gDNA'))))  # ty:ignore[missing-argument]
            # Assign a GO ID when a gene expression data is reported
            .withColumn(
                'geneExpressionId',
                f
                .when((f.col('alteration_type') == 'EXPR') & (f.col('alteration').contains('over')), 'GO_0010628')  # ty:ignore[missing-argument, invalid-argument-type]
                .when((f.col('alteration_type') == 'EXPR') & (f.col('alteration').contains('under')), 'GO_0010629')  # ty:ignore[missing-argument, invalid-argument-type]
                .when((f.col('alteration_type') == 'EXPR') & (f.col('alteration').contains('norm')), 'GO_0010467'),  # ty:ignore[missing-argument, invalid-argument-type]
            )
            # Create geneticVariation struct
            .withColumn(
                'geneticVariation',
                f.when(
                    f.col('alteration_type') != 'EXPR',
                    f.struct(
                        f.col('functionalConsequenceId'),
                        f.col('variantId').alias('id'),
                        f.col('alteration').alias('name'),
                    ),
                ),
            )
            # Create geneExpression struct
            .withColumn(
                'geneExpression',
                f.when(
                    f.col('alteration_type') == 'EXPR',
                    f.struct(f.col('geneExpressionId').alias('id'), f.col('alteration').alias('name')),
                ),
            )
        )

        pre_evidence = (
            biomarkers_enriched
            .withColumn('datasourceId', f.lit('cancer_biomarkers'))
            .withColumn('datatypeId', f.lit('affected_pathway'))
            .withColumnRenamed('tumor_type_full_name', 'diseaseFromSource')
            .withColumnRenamed('drug', 'drugFromSource')
            # diseaseFromSourceMappedId, drugId populated above
            .withColumnRenamed('Association', 'drugResponse')
            # confidence, literature and urls populated above
            .withColumnRenamed('gene', 'targetFromSourceId')
            .withColumnRenamed('Biomarker', 'biomarkerName')
            # geneticVariation, geneExpression populated above
            .drop(
                'tumor_type',
                'source',
                'alteration',
                'alteration_type',
                'IndividualMutation',
                'geneExpressionId',
                'gDNA',
                'functionalConsequenceId',
                'variantId',
                'DrugFullName',
                'niceName',
                'url',
            )
        )

        # Group evidence
        self.evidence = (
            pre_evidence
            .groupBy(
                'datasourceId',
                'datatypeId',
                'drugFromSource',
                'drugId',
                'drugResponse',
                'targetFromSourceId',
                'diseaseFromSource',
                'diseaseFromSourceMappedId',
                'confidence',
                'biomarkerName',
            )
            .agg(
                f.collect_set('literature').alias('literature'),
                f.collect_set('urls').alias('urls'),
                f.collect_set('geneticVariation').alias('geneticVariation'),
                f.collect_set('geneExpression').alias('geneExpression'),
            )
            # Replace empty lists with null values
            .withColumn(
                'literature', f.when(f.size(f.col('literature')) == 0, f.lit(None)).otherwise(f.col('literature'))
            )
            .withColumn('urls', f.when(f.size(f.col('urls')) == 0, f.lit(None)).otherwise(f.col('urls')))
            .withColumn(
                'geneticVariation',
                f.when(f.size(f.col('geneticVariation')) == 0, f.lit(None)).otherwise(f.col('geneticVariation')),
            )
            .withColumn(
                'geneExpression',
                f.when(f.size(f.col('geneExpression')) == 0, f.lit(None)).otherwise(f.col('geneExpression')),
            )
            # Collect geneticVariation info into biomarkers struct
            .withColumn('biomarkers', f.struct('geneExpression', 'geneticVariation'))
            .drop('geneticVariation', 'geneExpression')
            .distinct()
        )

        return self.evidence

    @staticmethod
    def get_variant_id(gdna: str) -> str | None:
        """Converts the genomic coordinates to the CHROM_POS_REF_ALT notation.

        Ex.: 'chr14:g.105243048G_T' --> '14_105243048_G_T'
        """
        translate_dct = {'chr': '', ':g.': '_', '>': '_', 'del': '_', 'ins': '_'}
        try:
            for k, v in translate_dct.items():
                gdna = gdna.replace(k, v)
            _, head, tail = re.split(r'^(.*?_\d+)', gdna)
            if bool(re.search(r'\d+', tail)):
                tail = re.split(r'^(_\d+_)', tail)[-1]
            return head + '_' + tail
        except AttributeError:
            return

    @staticmethod
    def zip_alterations_with_type(alterations, alteration_type):
        """Zips in a tuple the combination of the alteration w/ its correspondent type.

        This function helps disambiguate cases when multiple alterations are reported.
        By expanding the array of alteration types it accounts for the cases when
        several alterations are reported but only one type is given.

        Ex.:
        alterations = ['MET:Y1230C', 'Y1235D']
        alteration_type = ['MUT']
        --> [('MET:Y1230C', 'MUT'), ('Y1235D', 'MUT')].
        """
        alteration_types = alteration_type * len(alterations)
        return list(zip(alterations, alteration_types, strict=False))
