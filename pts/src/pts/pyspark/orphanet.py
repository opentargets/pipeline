"""Evidence parser for Orphanet's gene-disease associations."""

from itertools import chain
from typing import Any

import defusedxml.ElementTree as eT
from loguru import logger
from pyspark.sql import DataFrame, Row
from pyspark.sql.functions import array_distinct, col, create_map, lit, split

from pts.pyspark.common.ontology import add_efo_mapping
from pts.pyspark.common.session import Session

# The rest of the types are assigned to -> germline for allele origins
EXCLUDED_ASSOCIATIONTYPES = [
    'Major susceptibility factor in',
    'Part of a fusion gene in',
    'Candidate gene tested in',
    'Role in the phenotype of',
    'Biomarker tested in',
    'Disease-causing somatic mutation(s) in',
]

# Assigning variantFunctionalConsequenceId:
CONSEQUENCE_MAP = {
    'Disease-causing germline mutation(s) (loss of function) in': 'SO_0002054',
    'Disease-causing germline mutation(s) in': None,
    'Modifying germline mutation in': None,
    'Disease-causing germline mutation(s) (gain of function) in': 'SO_0002053',
}


def orphanet(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    spark = Session(app_name='orphanet', properties=properties)

    logger.info(f'parse XML from {source["evidence"]} into a list of dictionaries')
    orphanet_disorders = parse_orphanet_xml(source['evidence'])
    orphanet_df = spark.spark.createDataFrame(Row(**x) for x in orphanet_disorders)

    logger.info('process evidence strings')
    evidence_df = process_orphanet(orphanet_df)

    logger.info('add EFO mappings')
    evidence_df = add_efo_mapping(
        spark=spark.spark,
        evidence_df=evidence_df,
        disease_label_lut_path=source['ontoma_disease_label_lut'],
        disease_id_lut_path=source['ontoma_disease_id_lut'],
    )

    logger.info(f'write evidence strings to {destination}')
    evidence_df.write.mode('overwrite').parquet(destination)


def parse_orphanet_xml(orphanet_file: str) -> list[dict]:
    """Function to parse Orphanet xml dump and return the parsed data as a list of dictionaries."""
    # defusedxml.ElementTree.parse() returns the root element directly, not an ElementTree object
    root = eT.parse(orphanet_file)

    # Type checking to ensure we have the correct element
    if not hasattr(root, 'find'):
        raise ValueError('Failed to parse XML file - invalid root element')

    # Checking if the basic nodes are in the xml structure:
    logger.info(f'There are {root.find("DisorderList").get("count")} disease in the Orphanet xml file.')
    orphanet_disorders = []
    for disorder in root.find('DisorderList').findall('Disorder'):
        # Extracting disease information:
        parsed_disorder = {
            'diseaseFromSource': disorder.find('Name').text,
            'diseaseFromSourceId': 'Orphanet_' + disorder.find('OrphaCode').text,
            'type': disorder.find('DisorderType/Name').text,
        }

        # One disease might be mapped to multiple genes:
        for association in disorder.find('DisorderGeneAssociationList'):
            # For each mapped gene, an evidence is created:
            evidence = parsed_disorder.copy()

            # Not all gene/disease association is backed up by publication:
            try:
                evidence['literature'] = [
                    pmid.replace('[PMID]', '').rstrip()
                    for pmid in association.find('SourceOfValidation').text.split('_')
                    if '[PMID]' in pmid
                ]
            except AttributeError:
                evidence['literature'] = None

            evidence['associationType'] = association.find('DisorderGeneAssociationType/Name').text
            evidence['confidence'] = association.find('DisorderGeneAssociationStatus/Name').text

            # Parse gene name and id - going for Ensembl gene id only:
            gene = association.find('Gene')
            evidence['targetFromSource'] = gene.find('Name').text

            # Extracting ensembl gene id from cross references:
            try:
                ensembl_gene_id = [
                    xref.find('Reference').text
                    for xref in gene.find('ExternalReferenceList')
                    if 'ENSG' in xref.find('Reference').text
                ]
                evidence['targetFromSourceId'] = ensembl_gene_id[0] if len(ensembl_gene_id) > 0 else None
            except TypeError:
                evidence['targetFromSourceId'] = None

            # Collect evidence:
            orphanet_disorders.append(evidence)
    return orphanet_disorders


def process_orphanet(orphanet_df: DataFrame) -> DataFrame:
    """The JSON Schema format is applied to the df."""
    # Map association type to sequence ontology ID:
    so_mapping_expr = create_map([lit(x) for x in chain(*CONSEQUENCE_MAP.items())])

    return (
        orphanet_df.filter(~col('associationType').isin(EXCLUDED_ASSOCIATIONTYPES))
        .filter(~col('targetFromSourceId').isNull())
        .withColumn(
            'variantFunctionalConsequenceId',
            so_mapping_expr.getItem(col('associationType')),
        )
        .select(
            lit('orphanet').alias('datasourceId'),
            lit('genetic_association').alias('datatypeId'),
            split(lit('germline,somatic'), ',').alias('alleleOrigins'),
            'confidence',
            'diseaseFromSource',
            'diseaseFromSourceId',
            array_distinct(col('literature')).alias('literature'),
            'targetFromSource',
            'targetFromSourceId',
            'variantFunctionalConsequenceId',
        )
    )
