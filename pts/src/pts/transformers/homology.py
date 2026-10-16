"""Homology dataset generation.

Resolves Ensembl Compara homologies into one row per target and homologue. Homologue gene names
come from each species' Ensembl core database, human symbols and the target universe from
output/target.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.ensembl import CoreDump, gene_names
from pts.schemas.homology import homology_schema, orthologs_schema
from pts.transformers.utils.dataset import scan_dataset, write_dataset
from pts.transformers.utils.spark_casts import cast_double

DEFAULT_HGNC_ORTHOLOG_SPECIES = [
    '9606-human',
    '9598-chimpanzee',
    '9544-macaque',
    '10090-mouse',
    '10116-rat',
    '9986-rabbit',
    '10141-guineapig',
    '9615-dog',
    '9823-pig',
    '8364-frog',
    '7955-zebrafish',
    '7227-fly',
    '6239-worm',
]
HUMAN = 'homo_sapiens'
PARALOG_TYPES = ['other_paralog', 'within_species_paralog']

#: The Compara TSV columns the step reads.
COMPARA_COLUMNS = [
    'gene_stable_id',
    'species',
    'identity',
    'homology_type',
    'homology_gene_stable_id',
    'homology_species',
    'homology_identity',
    'is_high_confidence',
]


def read_species_dictionary(path: str) -> pl.DataFrame:
    """Read `species_EnsemblVertebrates.txt`, whose data rows end in a tab the header lacks."""
    return (
        scan_dataset(path, format='tsv', infer_schema=False, truncate_ragged_lines=True)
        .select('#name', 'species', 'taxonomy_id')
        .collect()
    )


def read_homologies(path: str) -> pl.DataFrame:
    """Read the Compara homology TSVs, keeping the rows with a human gene on either side."""
    return (
        scan_dataset(path, format='tsv', infer_schema=False)
        .select(COMPARA_COLUMNS)
        .filter((pl.col('species') == HUMAN) | (pl.col('homology_species') == HUMAN))
        .collect()
    )


def read_gene_names(path: str, species: list[str]) -> pl.DataFrame:
    """Gene id and name of every gene of every species, from `<path>/<species>/` core dumps."""
    return pl.concat([gene_names(CoreDump(f'{path}/{name}')) for name in species])


def build_homology(
    species_dictionary: pl.DataFrame,
    homologies: pl.DataFrame,
    names: pl.DataFrame,
    target_species: list[str],
) -> pl.DataFrame:
    """Build the homologue pairs of every human gene.

    Args:
        species_dictionary: `#name`, `species` and `taxonomy_id` of every Ensembl species.
        homologies: Compara homology rows, with at least `COMPARA_COLUMNS`, all strings.
        names: `id` and `name` of every gene of every species.
        target_species: whitelisted species as `<taxonomy id>-<name>`, in priority order.

    Returns:
        DataFrame in `orthologs_schema`.
    """
    tax_ids = [s.split('-')[0] for s in target_species]
    priority = pl.DataFrame(
        {'speciesId': tax_ids, 'priority': list(range(len(tax_ids)))},
        schema={'speciesId': pl.String, 'priority': pl.Int32},
    )
    species = species_dictionary.select(
        name='#name',
        speciesName='species',
        taxonomy_id='taxonomy_id',
    ).filter(pl.col('taxonomy_id').is_in(tax_ids))

    trimmed = pl.col('name').str.strip_chars(' ')
    symbols = names.select(
        homology_gene_stable_id='id',
        targetGeneSymbol=pl.when(trimmed != '').then(trimmed).otherwise(pl.col('id')),
    )

    compara = homologies.select(COMPARA_COLUMNS)
    query_human = compara.filter(pl.col('species') == HUMAN)
    swapped = compara.filter(
        ((pl.col('species') == HUMAN) & pl.col('homology_type').is_in(PARALOG_TYPES))
        | ((pl.col('species') != HUMAN) & (pl.col('homology_species') == HUMAN))
    ).select(
        gene_stable_id='homology_gene_stable_id',
        species='homology_species',
        identity='homology_identity',
        homology_type='homology_type',
        homology_gene_stable_id='gene_stable_id',
        homology_species='species',
        homology_identity='identity',
        is_high_confidence='is_high_confidence',
    )

    return (
        pl
        .concat([query_human, swapped])
        .join(species, left_on='homology_species', right_on='speciesName', how='inner', maintain_order='left')
        .join(symbols, on='homology_gene_stable_id', how='left', maintain_order='left')
        .select(
            id='gene_stable_id',
            speciesId='taxonomy_id',
            speciesName='name',
            homologyType='homology_type',
            targetGeneId='homology_gene_stable_id',
            isHighConfidence=pl.when(pl.col('is_high_confidence') != 'NULL').then(pl.col('is_high_confidence')),
            targetGeneSymbol='targetGeneSymbol',
            queryPercentageIdentity=cast_double('identity'),
            targetPercentageIdentity=cast_double('homology_identity'),
        )
        .join(priority, on='speciesId', how='left', maintain_order='left')
        .cast(orthologs_schema)
    )


def resolve_homology(orthologs: pl.DataFrame, target: pl.DataFrame) -> pl.DataFrame:
    """Resolve homologue symbols and keep the pairs whose human gene is in output/target.

    A homologue that is itself a target takes its `approvedSymbol`; otherwise its own name is
    kept, and a homologue without one takes the query gene's `approvedSymbol`.

    Args:
        orthologs: pairs from `build_homology`.
        target: `id` and `approvedSymbol` of every target.

    Returns:
        DataFrame in `homology_schema`, sorted by `targetId`, `targetGeneId`, `homologyType` and
        `speciesId`.
    """
    symbols = target.select('id', 'approvedSymbol')
    paralogs = symbols.select(targetGeneId='id', paralogGeneSymbol='approvedSymbol')
    return (
        orthologs
        .join(symbols, on='id', how='inner', maintain_order='left')
        .join(paralogs, on='targetGeneId', how='left', maintain_order='left')
        .with_columns(targetGeneSymbol=pl.coalesce('paralogGeneSymbol', 'targetGeneSymbol', 'approvedSymbol'))
        .rename({'id': 'targetId'})
        .select(homology_schema.names())
        .sort('targetId', 'targetGeneId', 'homologyType', 'speciesId', maintain_order=True)
    )


def homology(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build the homology dataset and write it out.

    Args:
        source: Mapping with keys `species_dictionary` (`species_EnsemblVertebrates.txt`),
            `homologies` (directory of Compara homology TSVs), `ensembl` (directory of one core
            dump per species) and `target` (output/target).
        destination: Output path for the homology dataset.
        settings: `ensembl_species`, the core dump directories under `ensembl`; optionally
            `hgncOrthologSpecies`.
        config: Unused.
    """
    settings = settings or {}
    target_species: list[str] = settings.get('hgncOrthologSpecies', DEFAULT_HGNC_ORTHOLOG_SPECIES)

    logger.info(f'loading data from {source}')
    species_dictionary = read_species_dictionary(source['species_dictionary'])
    homologies = read_homologies(source['homologies'])
    names = read_gene_names(source['ensembl'], settings['ensembl_species'])
    target = scan_dataset(source['target']).select('id', 'approvedSymbol').collect()

    logger.info('building homologue pairs')
    orthologs = build_homology(species_dictionary, homologies, names, target_species)

    logger.info('resolving homologue symbols against output/target')
    result = resolve_homology(orthologs, target)

    logger.info(f'writing output data to {destination}')
    write_dataset(result, destination)
