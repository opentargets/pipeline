"""The variant step: the variant index, from fastVEP and the joins on the mirrored Ensembl data.

The variant rows are collected from the evidence and credible sets, then annotated by
`chromosome.annotate_chromosomes`: fastVEP on each chromosome, then, on chunks of each
chromosome's rows in parallel, joins on the mirrored Ensembl data (transcript fields, colocated
variants, GERP, AlphaMissense, LOFTEE and SIFT) fill in the variants and consequences frames that
`index_rows` turns into index rows. GPN-Star scores, supplementary variant annotations (gnomAD, LoF
curation) merged by variant, and amino-acid level predictions (FoldX) through each variant's first
transcript with an amino-acid change are merged into each chunk. The inputs these share are prepared
once, before the chromosomes run, the remote ones read into local scratch, and each chunk is written
as one part of the output once every chunk has succeeded.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.utils.dataset import check_destinations_absent
from pts.transformers.variant.chromosome import Pools, annotate_chromosomes
from pts.transformers.variant.inputs import COPY_BYTES, chromosome_rows, prepared_chromosomes, shared_inputs
from pts.transformers.variant.prepared import ChunkSettings
from pts.transformers.variant.settings import VariantDestination, VariantSettings, VariantSource


def variant(
    source: dict[str, Any],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and write the variant index.

    Args:
        source: the `VariantSource` paths.
        destination: the `VariantDestination`, the index's `output`.
        settings: the `VariantSettings`.
        config: for `work_path`, where the step's scratch goes.
    """
    sources = VariantSource.model_validate(source)
    destinations = VariantDestination.model_validate(destination)
    options = VariantSettings.model_validate(settings)
    check_destinations_absent([destinations.output])
    if sources.gpn_star is not None and options.gpn_star is None:
        raise ValueError('a gpn_star source needs gpn_star cut-offs in the settings')
    with TemporaryDirectory(dir=config.work_path) as directory:
        scratch = Path(directory)
        rows, sizes = chromosome_rows(sources, scratch / 'rows')
        logger.info('Preparing the inputs every chromosome reads')
        shared = shared_inputs(sources, options, rows, scratch)
        chunk_settings = ChunkSettings(
            chunk_rows=options.chunk_rows, batch_size=options.batch_size, hash_threshold=options.hash_threshold
        )
        prepared = prepared_chromosomes(rows, shared, chunk_settings, scratch / 'chromosomes')
        logger.info('Annotating {} chromosomes', len(prepared))
        parts = annotate_chromosomes(
            prepared, sizes, shared.reference, shared.preparation, _pools(options), scratch / 'chromosomes' / 'parts'
        )
        logger.info('Publishing {} parts to {}', len(parts), destinations.output)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda part: _publish(part, destinations.output), parts))
    logger.info('Wrote variant index to {}', destinations.output)


def _pools(options: VariantSettings) -> Pools:
    """The step's process pools, as the settings size them."""
    return Pools(
        fastvep=options.parallel_chromosomes,
        preparations=options.parallel_preparations,
        chunks=options.parallel_chunks,
        chromosome_threads=options.chromosome_threads,
        chunk_threads=options.chunk_threads,
        preparation_niceness=options.preparation_niceness,
        chunk_niceness=options.chunk_niceness,
    )


def _publish(part: Path, destination: str) -> None:
    """Move a part into the output, a local directory or an object store prefix.

    The parts are written by the chunk processes, so the step copies them rather than writing one
    frame through `write_dataset`.
    """
    target = f'{destination}/{part.name}'
    if '://' not in destination:
        Path(destination).mkdir(parents=True, exist_ok=True)
        shutil.move(part, target)
        return
    with part.open('rb') as source, StorageHandle(target).open('wb') as out:
        shutil.copyfileobj(source, out, COPY_BYTES)
    part.unlink()
