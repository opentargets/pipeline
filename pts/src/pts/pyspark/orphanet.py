"""Evidence parser for Orphanet's gene-disease associations."""

import xml.etree.ElementTree as ET
from itertools import chain
from typing import Any

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
    orphanet_disorders = parse_orphanet_xml(spark.spark.read.text(source['evidence'], wholetext=True).collect()[0][0])
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


def _text(el: ET.Element | None, path: str) -> str | None:
    if el is None:
        return None
    found = el.find(path)
    if found is None:
        return None
    return found.text


def _req(el: ET.Element | None, path: str) -> ET.Element:
    if el is None:
        raise ValueError(f'missing required node: {path}')
    found = el.find(path)
    if found is None:
        raise ValueError(f'missing required node: {path}')
    return found


def parse_orphanet_xml(xml_string: str) -> list[dict]:
    """Function to parse Orphanet xml dump and return the parsed data as a list of dictionaries."""
    # Insecure parsing is fine, we trust Orphanet and the worst that can happen
    # is we freeze a machine in GCP.
    try:
        root: ET.Element[str] = ET.fromstring(xml_string)  # noqa: S314
    except ET.ParseError as e:
        raise ValueError('failed to parse xml file') from e

    # Checking if the basic nodes are in the xml structure:
    disorder_list = _req(root, 'DisorderList')
    logger.info(f'There are {disorder_list.get("count")} disease in the Orphanet xml file.')
    orphanet_disorders = []
    for disorder in disorder_list.findall('Disorder'):
        # Extracting disease information:
        orphanet_disorder_id = _text(disorder, 'OrphaCode')
        if orphanet_disorder_id is None:
            logger.warning(f'skipping orphanet disorder without id: {ET.tostring(disorder)}')
            continue
        orphanet_disorder_id = f'Orphanet_{orphanet_disorder_id}'
        parsed_disorder: dict[str, object] = {
            'diseaseFromSource': _text(disorder, 'Name'),
            'diseaseFromSourceId': 'Orphanet_' + orphanet_disorder_id,
            'type': _text(disorder, 'DisorderType/Name'),
        }

        # One disease might be mapped to multiple genes:
        for association in _req(disorder, 'DisorderGeneAssociationList'):
            # For each mapped gene, an evidence is created:
            evidence = parsed_disorder.copy()

            # Not all gene/disease associations are backed by publications:
            source_of_validation = _text(association, 'SourceOfValidation')
            if source_of_validation is None:
                evidence['literature'] = None
            else:
                evidence['literature'] = [
                    pmid.replace('[PMID]', '').rstrip() for pmid in source_of_validation.split('_') if '[PMID]' in pmid
                ]

            disorder_gene_association_type = _text(association, 'DisorderGeneAssociationType/Name')
            disorder_gene_association_status = _text(association, 'DisorderGeneAssociationStatus/Name')

            evidence['associationType'] = disorder_gene_association_type
            evidence['confidence'] = disorder_gene_association_status

            # Parse gene name and id - going for Ensembl gene id only:
            gene = association.find('Gene')
            gene_name = _text(gene, 'Name')
            evidence['targetFromSource'] = gene_name

            # Extracting ensembl gene id from cross references:
            ensembl = []
            xrefs = gene.find('ExternalReferenceList') if gene is not None else None
            for xref in xrefs or []:
                r = xref.find('Reference')
                if r is not None and r.text and 'ENSG' in r.text:
                    ensembl.append(r.text)
            evidence['targetFromSourceId'] = ensembl[0] if ensembl else None

            # Collect evidence:
            orphanet_disorders.append(evidence)
    return orphanet_disorders


def process_orphanet(orphanet_df: DataFrame) -> DataFrame:
    """The JSON Schema format is applied to the df."""
    # Map association type to sequence ontology ID:
    so_mapping_expr = create_map([lit(x) for x in chain(*CONSEQUENCE_MAP.items())])

    return (
        orphanet_df
        .filter(~col('associationType').isin(EXCLUDED_ASSOCIATIONTYPES))
        .filter(~col('targetFromSourceId').isNull())
        .withColumn(
            'variantFunctionalConsequenceId',
            so_mapping_expr[col('associationType')],
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
