"""Target view dataset generation.

The target dataset with its homologues, tractability, safety liabilities, chemical probes and
transcripts nested back in, in the shape target had before they were split out of it, then the ids of
its credible sets and pharmacogenomics records: the served `target_view`. `tep` is not reproduced and
`alternativeGenes` is always null. Also writes the served `target_by_region_view` and `transcript_view`.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.views.target_by_region_view import TargetByRegionView
from ot_datamodels.views.target_view import TargetView
from ot_datamodels.views.transcript_view import TranscriptView, TranscriptViewRow
from otter.config.model import Config

from pts.schemas.dataset.conform import conform, without_enums
from pts.task_outputs import writes
from pts.transformers.utils.dataset import scan_dataset, write_dataset

SOURCES = ['target', 'homology', 'target_tractability', 'target_safety_event', 'chemical_probes', 'transcript']
#: the columns read from the sources the id lists come from
ID_SOURCES = {
    'study': ['studyId', 'geneId'],
    'credible_set': ['id', 'studyId'],
    'pharmacogenomics': ['pharmacogenomicsId', 'targetFromSourceId'],
}

VIEW_SCHEMA = to_polars_schema(TargetView)
TRANSCRIPT_FIELDS = list(TranscriptViewRow.model_fields)


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


def credible_set_ids(study: pl.DataFrame, credible_set: pl.DataFrame) -> pl.DataFrame:
    """The credible sets of the studies of each gene, distinct and sorted, by `geneId`."""
    return (
        study
        .filter(pl.col('geneId').is_not_null())
        .join(credible_set.filter(pl.col('id') != ''), on='studyId', how='inner')
        .unique(['geneId', 'id'])
        .group_by('geneId')
        .agg(credibleSetIds=pl.col('id').sort())
    )


def pharmacogenomics_ids(pharmacogenomics: pl.DataFrame) -> pl.DataFrame:
    """The ids of the pharmacogenomics rows naming each target, ascending, by `targetFromSourceId`."""
    return (
        pharmacogenomics
        .filter(pl.col('targetFromSourceId').is_not_null())
        .group_by('targetFromSourceId')
        .agg(pharmacogenomicsIds=pl.col('pharmacogenomicsId').sort())
    )


def build_target_view(
    target: pl.DataFrame,
    homology: pl.DataFrame,
    target_tractability: pl.DataFrame,
    target_safety_event: pl.DataFrame,
    probes: pl.DataFrame,
    transcript: pl.DataFrame,
    study: pl.DataFrame,
    credible_set: pl.DataFrame,
    pharmacogenomics: pl.DataFrame,
) -> pl.DataFrame:
    """Every target row, in target's order, with the view columns after its own, conformed to `TargetView`.

    Lists are sorted, apart from `homologues`, by priority first, and `canonicalExons`. `credibleSetIds` and
    `pharmacogenomicsIds` are `[]` for a target with none.
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
    for ids in [
        credible_set_ids(study, credible_set).rename({'geneId': 'id'}),
        pharmacogenomics_ids(pharmacogenomics).rename({'targetFromSourceId': 'id'}),
    ]:
        view = view.join(ids, on='id', how='left', maintain_order='left')
    view = view.with_columns(
        alternativeGenes=pl.lit(None, dtype=VIEW_SCHEMA['alternativeGenes']),
        credibleSetIds=pl.col('credibleSetIds').fill_null(pl.lit([], dtype=VIEW_SCHEMA['credibleSetIds'])),
        pharmacogenomicsIds=pl
        .col('pharmacogenomicsIds')
        .cast(VIEW_SCHEMA['pharmacogenomicsIds'])
        .fill_null(pl.lit([], dtype=VIEW_SCHEMA['pharmacogenomicsIds'])),
    )
    return conform(view.select(VIEW_SCHEMA.names()), TargetView)


def build_target_by_region(view: pl.DataFrame) -> pl.DataFrame:
    """One row per view row: its genomic location and id, by location then id, conformed to `TargetByRegionView`."""
    location = pl.col('genomicLocation')
    frame = view.rechunk().select(
        chromosome=location.struct.field('chromosome'),
        start=location.struct.field('start'),
        end=location.struct.field('end'),
        target=pl.struct('id'),
    )
    frame = frame.sort('chromosome', 'start', 'end', pl.col('target').struct.field('id'), maintain_order=True)
    return conform(frame, TargetByRegionView)


def build_transcript_view(transcript: pl.DataFrame) -> pl.DataFrame:
    """The transcripts of each target, rows by `targetId`, conformed to `TranscriptView`.

    Each in the transcript's column order, by `transcriptId`.
    """
    rows = transcript.rechunk().sort('targetId', 'transcriptId', maintain_order=True)
    frame = rows.group_by('targetId', maintain_order=True).agg(transcripts=pl.struct(TRANSCRIPT_FIELDS))
    return conform(frame, TranscriptView)


@writes(view=TargetView, by_region=TargetByRegionView, transcript=TranscriptView)
def target_view(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build the target view, the targets by region and the transcript view, and write them.

    Args:
        source: `target`, `homology`, `target_tractability`, `target_safety_event`,
            `chemical_probes`, `transcript`, `study`, `credible_set` and `pharmacogenomics` locations.
        destination: `view`, `by_region` and `transcript` directories.
        settings: unused.
        config: unused.
    """
    logger.info('load target view inputs')
    scans = [scan_dataset(source[name]) for name in SOURCES]
    scans += [scan_dataset(source[name]).select(columns) for name, columns in ID_SOURCES.items()]
    frames = [without_enums(frame) for frame in pl.collect_all(scans)]
    view = build_target_view(*frames)
    logger.info(f'{view.height} targets in the target view')
    write_dataset(view, str(destination['view']))
    write_dataset(build_target_by_region(view), str(destination['by_region']))
    transcript = build_transcript_view(frames[SOURCES.index('transcript')])
    logger.info(f'{transcript.height} targets in the transcript view')
    write_dataset(transcript, str(destination['transcript']))
