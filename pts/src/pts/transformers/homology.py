import gc
import json
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.homology import homology_schema, orthologs_schema
from pts.transformers.utils.dataset import write_dataset
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


class FilteredJSONDecoder(json.JSONDecoder):
    """JSON Decoder that filter keys in the JSON object.

    This decoder calls a hook on each JSON object that filters out keys not in
    allowed_keys set. It also processes the root_key, so we can get to the data.

    This saves us a lot of memory and time, and serves as a workaround for the
    bug in polars that causes it to crash when loading large JSON objects:

    https://github.com/pola-rs/polars/issues/17677
    """

    def __init__(
        self,
        root_key='genes',
        allowed_keys=None,
        *args,
        **kwargs,
    ) -> None:
        self.root_key = root_key
        self.allowed_keys = allowed_keys or {'id', 'name'}
        super().__init__(*args, **kwargs, object_hook=self.filter_keys)

    def filter_keys(self, obj: dict) -> dict:
        """Modified filter that keeps only certain roots."""
        if self.root_key in obj:
            return {self.root_key: obj[self.root_key]}
        return {k: v for k, v in obj.items() if k in self.allowed_keys}


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
    source: str,
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    s = StorageHandle(source)
    f = s.open()

    data = json.loads(f.read(), cls=FilteredJSONDecoder)
    df = pl.from_dicts(data['genes'])
    # free up memory
    del data
    gc.collect()

    # write the result locally
    write_dataset(df, destination)
    logger.info('transformation complete')
