"""Evidence from Orphanet's gene-disease associations.

Polars port of the `pyspark orphanet` parser together with its `evidence_postprocess` step.
Diseases are mapped with the OnToma lookup tables through `add_efo_mapping`, by label and by id.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.orphanet import ORPHANET_SCHEMA
from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.expressions import confidence_score, direction_when_disease_present
from pts.transformers.evidence.utils.luts import build_luts
from pts.transformers.evidence.utils.ontology import add_efo_mapping
from pts.transformers.evidence.utils.run import write_evidence

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

ORPHANET = EvidenceDatasource(
    datasource_id='orphanet',
    unique_fields=['targetId', 'targetFromSourceId', 'diseaseId', 'datasourceId', 'diseaseFromSourceId'],
    score=confidence_score({'Assessed': 1.0, 'Not yet assessed': 0.5}),
)
DIRECTION_ON_TRAIT = direction_when_disease_present('risk')
DIRECTION_ON_TARGET = (
    pl
    .when(pl.col('variantFunctionalConsequenceId') == 'SO_0002053')
    .then(pl.lit('GoF'))
    .when(pl.col('variantFunctionalConsequenceId') == 'SO_0002054')
    .then(pl.lit('LoF'))
    .otherwise(None)
)


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
            'diseaseFromSourceId': orphanet_disorder_id,
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
            for xref in xrefs if xrefs is not None else []:
                r = xref.find('Reference')
                if r is not None and r.text and 'ENSG' in r.text:
                    ensembl.append(r.text)
            evidence['targetFromSourceId'] = ensembl[0] if ensembl else None

            # Collect evidence:
            orphanet_disorders.append(evidence)
    return orphanet_disorders


def parse_orphanet(path: str) -> pl.LazyFrame:
    """Raw disorder-gene associations of the Orphanet XML dump, before disease mapping."""
    text, _ = StorageHandle(path).read()
    rows = parse_orphanet_xml(text.decode('utf-8'))
    return pl.DataFrame(rows, schema=ORPHANET_SCHEMA).lazy()


def process_orphanet(raw: pl.LazyFrame) -> pl.LazyFrame:
    """Drop the excluded association types and rows without an Ensembl gene, and shape the evidence columns."""
    return (
        raw
        .filter(pl.col('associationType').is_not_null() & ~pl.col('associationType').is_in(EXCLUDED_ASSOCIATIONTYPES))
        .filter(pl.col('targetFromSourceId').is_not_null())
        .select(
            datasourceId=pl.lit('orphanet'),
            datatypeId=pl.lit('genetic_association'),
            alleleOrigins=pl.lit(['germline', 'somatic'], dtype=pl.List(pl.String)),
            confidence=pl.col('confidence'),
            diseaseFromSource=pl.col('diseaseFromSource'),
            diseaseFromSourceId=pl.col('diseaseFromSourceId'),
            literature=pl.col('literature').list.unique(maintain_order=True),
            targetFromSource=pl.col('targetFromSource'),
            targetFromSourceId=pl.col('targetFromSourceId'),
            variantFunctionalConsequenceId=pl
            .col('associationType')
            .replace_strict(CONSEQUENCE_MAP, default=None, return_dtype=pl.String),
        )
    )


def orphanet(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process Orphanet evidence.

    Args:
        source: `evidence` XML file, `ontoma_disease_label_lut`, `ontoma_disease_id_lut`, `disease`,
            `target` and `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    logger.info(f'parse orphanet XML from {source["evidence"]}')
    raw = add_efo_mapping(
        process_orphanet(parse_orphanet(source['evidence'])),
        disease_label_lut_path=source['ontoma_disease_label_lut'],
        disease_id_lut_path=source['ontoma_disease_id_lut'],
    ).collect()
    luts = build_luts(raw.lazy(), disease=source['disease'], target=source['target'], literature=source['literature'])
    evidence = (
        ORPHANET
        .postprocess(raw.lazy(), luts)
        .assign_direction_on_trait(DIRECTION_ON_TRAIT)
        .assign_direction_on_target(DIRECTION_ON_TARGET, luts.target)
        .hash_long_variant_identifiers()
    )
    write_evidence(evidence, destination)
