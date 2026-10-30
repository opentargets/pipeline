"""Target index dataset generation.

Assembles the target index from the Ensembl core database dump and the HGNC, UniProt, ChEMBL,
Gene Ontology, HPA, gnomAD, COSMIC hallmarks, NCBI and Reactome inputs. The per-source frames are
built in `pts.transformers.target`.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from otter.config.model import Config

from pts.postgres import read_dump_tables
from pts.transformers.target import chembl_class, constraint, go, hallmarks, hgnc, hpa, ncbi, reactome, uniprot
from pts.transformers.target.assemble import assemble, promote_hgnc_swissprot
from pts.transformers.target.ensembl import build_ensembl
from pts.transformers.utils.dataset import write_dataset
from pts.transformers.utils.ensembl import CoreDump


def target(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any] | None,
    config: Config,
) -> None:
    """Generate the target index dataset.

    Args:
        source: paths keyed by `ensembl` (core dump directory), `chembl`, `uniprot`, `uniprot_ssl`,
            `hpa`, `hpa_sl`, `gnomad`, `hgnc`, `gene_ontology_human`, `gene_ontology_rna`,
            `gene_ontology_rna_lookup`, `gene_ontology_eco_lookup`, `hallmarks`, `ncbi`,
            `reactome_pathways` and `reactome_etl`.
        destination: path of the target dataset.
        settings: unused.
        config: for `work_path`, where the ChEMBL restore puts its scratch.
    """
    logger.info(f'restoring the protein classification tables from {source["chembl"]}')
    # scratch_root: the restore needs gigabytes, and `work_path` is the work disk.
    tables = read_dump_tables(
        source['chembl'], chembl_class.TABLES, schema_name=chembl_class.SCHEMA_NAME, scratch_root=config.work_path
    )
    protein_class_df = chembl_class.build_protein_classification(**tables)

    logger.info(f'building ensembl genes from {source["ensembl"]}')
    ensembl = build_ensembl(CoreDump(source['ensembl']))

    logger.info('building hgnc')
    hgnc_df = hgnc.build_hgnc(hgnc.read_hgnc(source['hgnc']))

    logger.info('building uniprot')
    uniprot_entries = uniprot.read_uniprot(source['uniprot'])
    uniprot_df = uniprot.build_uniprot(uniprot_entries, uniprot.read_uniprot_ssl(source['uniprot_ssl']))

    # before GO, so GO annotations reach the genes through the promoted accessions too
    ensembl = promote_hgnc_swissprot(ensembl, hgnc_df, uniprot_entries)

    logger.info('building gene ontology')
    go_df = go.build_go(
        go.read_tsv(source['gene_ontology_human'], comment_prefix='!'),
        go.read_tsv(source['gene_ontology_rna'], comment_prefix='!'),
        go.read_tsv(source['gene_ontology_rna_lookup']),
        go.read_tsv(source['gene_ontology_eco_lookup'], comment_prefix='!'),
        ensembl,
    )

    logger.info('building hallmarks, ncbi, hpa')
    hallmarks_df = hallmarks.build_hallmarks(hallmarks.read_hallmarks(source['hallmarks']))
    ncbi_df = ncbi.build_ncbi(ncbi.read_ncbi(source['ncbi']))
    hpa_df = hpa.build_hpa(hpa.read_hpa(source['hpa']), hpa.read_hpa_ssl(source['hpa_sl']))

    logger.info('building constraint and reactome')
    constraint_df = constraint.build_constraint(constraint.read_constraint(source['gnomad']))
    reactome_df = reactome.build_reactome(
        reactome.read_reactome_pathways(source['reactome_pathways']),
        reactome.read_reactome_etl(source['reactome_etl']),
    )

    result = assemble(
        ensembl,
        hgnc_df,
        go_df,
        hallmarks_df,
        uniprot_df,
        protein_class_df,
        hpa_df,
        constraint_df,
        ncbi_df,
        reactome_df,
    )
    logger.info(f'writing {result.height} targets to {destination}')
    write_dataset(result, destination)
