"""Target view dataset generation.

The target dataset with its homologues, tractability, safety liabilities, chemical probes and
transcripts nested back in, in the shape target had before they were split out of it. A
compatibility view for consumers of that shape (Platform, API, croissant), not an input to other
steps. `tep` is not reproduced and `alternativeGenes` is always null.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.schemas.target_view import VIEW_COLUMNS
from pts.transformers.utils.dataset import scan_dataset, write_dataset

SOURCES = ['target', 'homology', 'target_tractability', 'target_safety_event', 'chemical_probes', 'transcript']


def homologues(homology: pl.DataFrame) -> pl.DataFrame:
    """The homologues of each target, by ascending priority.

    Raises:
        ValueError: if a target with more than one homologue has one without a priority.
    """
    counts = homology.group_by('targetId').agg(pl.len(), pl.col('priority').null_count().alias('nulls'))
    if counts.filter((pl.col('len') > 1) & (pl.col('nulls') > 0)).height:
        raise ValueError('a target with more than one homologue has a homologue without a priority')
    homologue = pl.struct(
        'speciesId',
        'speciesName',
        'homologyType',
        'targetGeneId',
        'isHighConfidence',
        'targetGeneSymbol',
        'queryPercentageIdentity',
        'targetPercentageIdentity',
        'priority',
    )
    return homology.group_by('targetId').agg(homologues=homologue.sort_by(pl.col('priority'), homologue))


def tractability(target_tractability: pl.DataFrame) -> pl.DataFrame:
    """The tractability assessments of each target."""
    assessment = pl.struct('modality', pl.col('category').alias('id'), 'value')
    return target_tractability.group_by('targetId').agg(tractability=assessment.sort())


def safety_liabilities(target_safety_event: pl.DataFrame) -> pl.DataFrame:
    """The safety events of each target."""
    liability = pl.struct('event', 'eventId', 'effects', 'biosamples', 'datasource', 'literature', 'url', 'studies')
    return target_safety_event.group_by('targetId').agg(safetyLiabilities=liability.sort())


def chemical_probes(probes: pl.DataFrame) -> pl.DataFrame:
    """The chemical probes of each target."""
    probe = pl.struct(
        'targetFromSourceId',
        pl.col('drugFromSource').alias('id'),
        'drugFromSourceId',
        'drugId',
        'mechanismOfAction',
        'origin',
        'control',
        'isHighQuality',
        'probesDrugsScore',
        'scoreInCells',
        'scoreInOrganisms',
        'urls',
    )
    return probes.group_by('targetId').agg(chemicalProbes=probe.sort())


def canonical_exons(canonical: pl.DataFrame) -> pl.DataFrame:
    """The exon boundaries of each target's canonical transcripts, as strings: start, end, start, end, ...

    Exons are ordered by start then end, nulls first.
    """
    exons = (
        canonical
        .filter(pl.col('exons').list.len() > 0)
        .select('targetId', 'exons')
        .explode('exons')
        .select(
            'targetId',
            pl.col('exons').struct.field('start').cast(pl.String),
            pl.col('exons').struct.field('end').cast(pl.String),
            start_order=pl.col('exons').struct.field('start'),
            end_order=pl.col('exons').struct.field('end'),
        )
        .sort('targetId', 'start_order', 'end_order', nulls_last=False, maintain_order=True)
        .with_row_index('exon')
    )
    boundaries = pl.concat([
        exons.select('targetId', 'exon', side=pl.lit(0), boundary='start'),
        exons.select('targetId', 'exon', side=pl.lit(1), boundary='end'),
    ]).sort('exon', 'side')
    return boundaries.group_by('targetId', maintain_order=True).agg(canonicalExons='boundary')


def transcripts(transcript: pl.DataFrame) -> pl.DataFrame:
    """The transcripts of each target, its canonical transcript and the exons of that transcript.

    A target with more than one canonical transcript has one row per canonical transcript, and the
    exons of all of them.
    """
    swissprot, trembl = pl.col('uniprotSwissprotIds'), pl.col('uniprotTremblIds')
    entry = pl.struct(
        'transcriptId',
        'biotype',
        uniprotId=pl.coalesce(swissprot.list.first(), trembl.list.first()),
        isUniprotReviewed=pl.when(swissprot.is_not_null()).then(True).when(trembl.is_not_null()).then(False),
        translationId='proteinId',
        alphafoldId=pl.col('alphafoldIds').list.first(),
        uniprotIsoformId=pl.col('uniprotIsoformIds').list.first(),
        isEnsemblCanonical='isEnsemblCanonical',
    )
    per_target = transcript.group_by('targetId').agg(
        transcriptIds=pl.col('transcriptId').drop_nulls().sort(),
        transcripts=entry.sort(),
    )
    canonical = transcript.filter(pl.col('isEnsemblCanonical'))
    canonical_transcript = canonical.select(
        'targetId',
        canonicalTranscript=pl.struct(pl.col('transcriptId').alias('id'), 'chromosome', 'start', 'end', 'strand'),
    ).sort('targetId', 'canonicalTranscript')
    return per_target.join(canonical_transcript, on='targetId', how='left', maintain_order='left').join(
        canonical_exons(canonical), on='targetId', how='left', maintain_order='left'
    )


def build_target_view(
    target: pl.DataFrame,
    homology: pl.DataFrame,
    target_tractability: pl.DataFrame,
    target_safety_event: pl.DataFrame,
    probes: pl.DataFrame,
    transcript: pl.DataFrame,
) -> pl.DataFrame:
    """Every target row, in target's order, with the view columns after its own.

    Lists are sorted, apart from `homologues`, by priority first, and `canonicalExons`.
    """
    view = target
    for aggregate in [
        homologues(homology),
        tractability(target_tractability),
        safety_liabilities(target_safety_event),
        chemical_probes(probes),
        transcripts(transcript),
    ]:
        view = view.join(aggregate.rename({'targetId': 'id'}), on='id', how='left', maintain_order='left')
    return view.with_columns(alternativeGenes=pl.lit(None, dtype=VIEW_COLUMNS['alternativeGenes'])).select(
        *target.columns, *VIEW_COLUMNS
    )


def target_view(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build the target view and write it.

    Args:
        source: `target`, `homology`, `target_tractability`, `target_safety_event`,
            `chemical_probes` and `transcript` locations.
        destination: the target view directory.
        settings: unused.
        config: unused.
    """
    logger.info('load target view inputs')
    frames = pl.collect_all([scan_dataset(source[name]) for name in SOURCES])
    result = build_target_view(*frames)
    logger.info(f'{result.height} targets in the target view')
    write_dataset(result, destination)
