"""Credible set views: the served `credible_set_view`, `credible_set_locus_view` and the three credible-set read models.

Replaces `credible_sets.sql`, `study_credible_sets.sql`, `target_credible_sets.sql` and `variant_credible_sets.sql`.
A set-level frame (each credible set with its lead variant's alleles, its study's trait, condition, target,
biosample and diseases, and its top L2G gene) is built in memory. `credible_set_variant` is split into
`bucket_count` hash buckets of `credibleSetId` on local disk, and the buckets are built `concurrency` at a time:
each adds the lead variant's statistics and the set size to its sets and writes one part of every view.
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels import DatasetModel
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.views.credible_set_locus_view import LOCUS_FIELDS, CredibleSetLocusView
from ot_datamodels.views.credible_set_view import CredibleSetView
from ot_datamodels.views.study_credible_set_view import StudyCredibleSetView
from ot_datamodels.views.target_credible_set_view import TargetCredibleSetView
from ot_datamodels.views.variant_credible_set_view import VariantCredibleSetView
from otter.config.model import Config

from pts.schemas.dataset.conform import conform, without_enums
from pts.task_outputs import writes
from pts.transformers.utils.buckets import bucket_of, for_each_bucket, read_bucket, write_buckets
from pts.transformers.utils.dataset import check_destinations_absent, list_parts, scan_dataset

DEFAULT_BUCKET_COUNT = 16
DEFAULT_CONCURRENCY = 4
ROW_GROUP_SIZE = 100_000
#: The columns read from each source but the two credible set outputs, which are read whole.
COLUMNS = {
    'variant': ['variantId', 'referenceAllele', 'alternateAllele'],
    'study': ['studyId', 'traitFromSource', 'condition', 'geneId', 'biosampleFromSourceId', 'diseaseIds'],
    'l2g_prediction': ['credibleSetId', 'geneId', 'score'],
    'target': ['id', 'approvedSymbol'],
    'disease': ['id', 'name', 'therapeuticAreas'],
    'biosample': ['biosampleId', 'biosampleName'],
}
#: The lead variant's statistics: credible_set_variant column -> view column.
LEAD = {
    'pValueMantissa': 'leadPValueMantissa',
    'pValueExponent': 'leadPValueExponent',
    'reportedBeta': 'leadBeta',
    'reportedStandardError': 'leadStandardError',
    'effectAlleleFrequency': 'leadEffectAlleleFrequency',
    'sampleSize': 'leadSampleSize',
}
#: The read models' names for the lead statistics and the lead's chromosome.
READ_MODEL_NAMES = {
    'leadPValueMantissa': 'pValueMantissa',
    'leadPValueExponent': 'pValueExponent',
    'leadBeta': 'beta',
    'chromosome': 'leadChromosome',
}
#: Destination key -> the model of the view it holds.
VIEWS: dict[str, type[DatasetModel]] = {
    'view': CredibleSetView,
    'locus': CredibleSetLocusView,
    'study': StudyCredibleSetView,
    'target': TargetCredibleSetView,
    'variant': VariantCredibleSetView,
}
DISEASES = to_polars_schema(VariantCredibleSetView)['diseases']


def top_l2g(l2g_prediction: pl.DataFrame, target: pl.DataFrame) -> pl.DataFrame:
    """The top L2G gene of each credible set with predictions.

    The highest score, then the lowest `geneId`; with the score and the gene's approved symbol (null when the gene is
    not a target).
    """
    top = (
        l2g_prediction
        .sort(['credibleSetId', 'score', 'geneId'], descending=[False, True, False])
        .group_by('credibleSetId', maintain_order=True)
        .first()
    )
    return top.join(target, left_on='geneId', right_on='id', how='left', maintain_order='left').select(
        'credibleSetId',
        topL2GGeneId='geneId',
        topL2GApprovedSymbol='approvedSymbol',
        topL2GScore='score',
    )


def study_diseases(study: pl.DataFrame, disease: pl.DataFrame) -> pl.DataFrame:
    """The diseases of each study with a known one, and their therapeutic areas.

    `diseases`: (id, name) in `diseaseIds` order, repeats kept and unknown ids left out. `diseaseTherapeuticAreas`:
    the distinct (id, name) of their therapeutic areas that are diseases, by id then name.
    """
    names = disease.select('id', 'name')
    areas = (
        disease
        .select('id', area='therapeuticAreas')
        .explode('area')
        .join(names.rename({'id': 'area', 'name': 'areaName'}), on='area', how='inner')
    )
    listed = (
        study
        .select('studyId', id='diseaseIds')
        .with_columns(position=pl.int_ranges(pl.col('id').list.len()))
        .explode('id', 'position')
    )
    known = listed.join(names, on='id', how='inner').sort('studyId', 'position').rechunk()
    diseases = known.group_by('studyId', maintain_order=True).agg(diseases=pl.struct('id', 'name'))
    therapeutic = (
        known
        .select('studyId', 'id')
        .unique()
        .join(areas, on='id', how='inner')
        .select('studyId', id='area', name='areaName')
        .unique()
        .sort('studyId', 'id', 'name')
        .rechunk()
        .group_by('studyId', maintain_order=True)
        .agg(diseaseTherapeuticAreas=pl.struct('id', 'name'))
    )
    return diseases.join(therapeutic, on='studyId', how='left', maintain_order='left')


def set_frame(
    credible_set: pl.DataFrame,
    variant: pl.DataFrame,
    study: pl.DataFrame,
    l2g_prediction: pl.DataFrame,
    target: pl.DataFrame,
    disease: pl.DataFrame,
    biosample: pl.DataFrame,
) -> pl.DataFrame:
    """Every credible set with what its views need but the lead statistics and the set size.

    Unmatched values as the scripts' LEFT JOINs left them: alleles and `traitFromSource` `''`, diseases `[]`, the
    rest null.
    """
    # an output written with other Enum categories (2612-3's) is conformed to the views' Enums from strings
    sets = without_enums(credible_set).select(
        credibleSetId='id',
        studyId='studyId',
        studyType='studyType',
        chromosome='chromosome',
        leadVariantId='leadVariantId',
        leadPosition=pl.col('leadVariantId').str.split('_').list.get(1).cast(pl.UInt32),
        locusStart='locusStart',
        locusEnd='locusEnd',
        fineMappingMethod='fineMappingMethod',
        logBayesFactor='logBayesFactor',
        confidence='confidence',
        purityMeanR2='purityMeanR2',
        purityMinR2='purityMinR2',
        qualityControls='qualityControls',
        isTransQtl=pl.col('qualityControls').list.contains('Trans QTL').cast(pl.UInt8),
    )
    alleles = variant.filter(pl.col('variantId').is_in(sets['leadVariantId'].implode())).select(
        leadVariantId='variantId', leadReferenceAllele='referenceAllele', leadAlternateAllele='alternateAllele'
    )
    symbols = target.select('id', 'approvedSymbol')
    info = (
        study
        .select(
            'studyId',
            'traitFromSource',
            'condition',
            studyTargetId='geneId',
            biosampleId='biosampleFromSourceId',
        )
        .join(study_diseases(study, disease), on='studyId', how='left', maintain_order='left')
        .join(
            symbols.rename({'id': 'studyTargetId', 'approvedSymbol': 'studyTargetApprovedSymbol'}),
            on='studyTargetId',
            how='left',
            maintain_order='left',
        )
        .join(biosample.select('biosampleId', 'biosampleName'), on='biosampleId', how='left', maintain_order='left')
    )
    return (
        sets
        .join(alleles, on='leadVariantId', how='left', maintain_order='left')
        .join(info, on='studyId', how='left', maintain_order='left')
        .join(top_l2g(l2g_prediction, symbols), on='credibleSetId', how='left', maintain_order='left')
        .with_columns(
            pl.col('leadReferenceAllele', 'leadAlternateAllele', 'traitFromSource').fill_null(''),
            pl.col('diseases', 'diseaseTherapeuticAreas').fill_null(pl.lit([], dtype=DISEASES)),
        )
    )


def target_pairs(credible_set: pl.DataFrame, study: pl.DataFrame, target: pl.DataFrame) -> pl.DataFrame:
    """The distinct (target, credible set) pairs of the studies of each target (`study.geneId`), targets only."""
    return (
        study
        .filter(pl.col('geneId').is_in(target['id'].implode()))
        .select('studyId', targetId='geneId')
        .join(credible_set.select('studyId', credibleSetId='id'), on='studyId', how='inner')
        .select('targetId', 'credibleSetId')
        .unique()
    )


def locus_frame(rows: pl.DataFrame) -> pl.DataFrame:
    """One row per credible set of the `credible_set_variant` rows.

    Its variants by `posteriorProbability` descending, then `variantId`.
    """
    return (
        rows
        .rechunk()
        .sort(['credibleSetId', 'posteriorProbability', 'variantId'], descending=[False, True, False])
        .select('credibleSetId', row=pl.struct(LOCUS_FIELDS))
        .group_by('credibleSetId', maintain_order=True)
        .agg(locus=pl.col('row'))
    )


def bucket_views(rows: pl.DataFrame, sets: pl.DataFrame, pairs: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """The five views of some credible sets, conformed, by destination key.

    Args:
        rows: every `credible_set_variant` row of the sets.
        sets: the sets' `set_frame` rows.
        pairs: the sets' `target_pairs`.

    Raises:
        ValueError: if a set has no row for its lead variant.
    """
    size = rows.group_by('credibleSetId').agg(locusSize=pl.len())
    lead = rows.select(
        'credibleSetId', pl.col('variantId').alias('leadVariantId'), *(pl.col(c).alias(n) for c, n in LEAD.items())
    )
    full = (
        sets
        .join(lead, on=['credibleSetId', 'leadVariantId'], how='inner', maintain_order='left')
        .join(size, on='credibleSetId', how='inner', maintain_order='left')
        .sort('credibleSetId')
    )
    repeated = full.height - full['credibleSetId'].n_unique()
    if repeated:
        raise ValueError(f'duplicate credibleSetId in credible_set_view: {repeated} repeated lead rows')
    if full.height != sets.height:
        raise ValueError(f'{sets.height - full.height} credible sets have no credible_set_variant row for their lead')
    read_model = full.rename(READ_MODEL_NAMES)
    target = pairs.join(read_model, on='credibleSetId', how='inner').with_columns(pl.col('biosampleName').fill_null(''))
    variant = rows.select('credibleSetId', 'variantId', 'posteriorProbability').join(
        read_model, on='credibleSetId', how='inner', maintain_order='left'
    )
    frames = {
        'view': full,
        'locus': locus_frame(rows),
        'study': read_model.filter(pl.col('studyType') == 'gwas'),
        'target': target.sort('targetId', 'credibleSetId'),
        'variant': variant,
    }
    return {key: conform(frame.select(list(VIEWS[key].model_fields)), VIEWS[key]) for key, frame in frames.items()}


def credible_set_view_frames(
    credible_set: pl.DataFrame,
    credible_set_variant: pl.DataFrame,
    variant: pl.DataFrame,
    study: pl.DataFrame,
    l2g_prediction: pl.DataFrame,
    target: pl.DataFrame,
    disease: pl.DataFrame,
    biosample: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """The five views, unbucketed, by destination key."""
    sets = set_frame(credible_set, variant, study, l2g_prediction, target, disease, biosample)
    return bucket_views(credible_set_variant, sets, target_pairs(credible_set, study, target))


def _by_bucket(frame: pl.DataFrame, bucket_count: int) -> dict[int, pl.DataFrame]:
    parts = frame.with_columns(bucket_of('credibleSetId', bucket_count)).partition_by(
        'bucket', as_dict=True, include_key=False
    )
    return {int(bucket): rows for (bucket,), rows in parts.items()}


@writes(**VIEWS)
def credible_set_view(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build the five credible set views in buckets and write them, one part per bucket each.

    Args:
        source: `credible_set`, `credible_set_variant`, `variant`, `study`, `l2g_prediction`, `target`, `disease`
            and `biosample` locations.
        destination: `view`, `locus`, `study`, `target` and `variant` directories.
        settings: optional `bucket_count` and `concurrency`.
        config: for `work_path`, where the buckets go.

    Raises:
        ValueError: if anything already exists at a destination, or a set has no row for its lead variant.
    """
    settings = settings or {}
    bucket_count = int(settings.get('bucket_count', DEFAULT_BUCKET_COUNT))
    concurrency = int(settings.get('concurrency', DEFAULT_CONCURRENCY))
    paths = {key: str(destination[key]).rstrip('/') for key in VIEWS}
    check_destinations_absent(paths.values())
    credible_set = scan_dataset(source['credible_set']).collect()
    inputs = pl.collect_all([scan_dataset(source[name]).select(columns) for name, columns in COLUMNS.items()])
    frames = dict(zip(COLUMNS, inputs, strict=True))
    sets = set_frame(credible_set, *(frames[name] for name in COLUMNS))
    pairs = target_pairs(credible_set, frames['study'], frames['target'])
    logger.info(f'{sets.height} credible sets, {pairs.height} target pairs')
    del credible_set, inputs, frames
    sets_by_bucket, pairs_by_bucket = _by_bucket(sets, bucket_count), _by_bucket(pairs, bucket_count)
    empty_sets, empty_pairs = sets.clear(), pairs.clear()
    del sets, pairs
    parts = list_parts(source['credible_set_variant'], '*.parquet')
    work = config.work_path if config is not None else None
    if work is not None:
        Path(work).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=work, prefix='credible_set_view_') as tmp:
        directory = Path(tmp)
        total = write_buckets(parts, lambda f: f, 'credibleSetId', bucket_count, directory, concurrency)
        logger.info(f'{total} credible set variants from {len(parts)} parts in {bucket_count} buckets')
        for path in paths.values():
            if '://' not in path:
                os.makedirs(path)

        def build(bucket: int) -> dict[str, int]:
            rows = read_bucket(directory, bucket)
            bucket_sets = sets_by_bucket.get(bucket, empty_sets)
            if rows is None:
                if bucket_sets.height:
                    raise ValueError(f'{bucket_sets.height} credible sets have no credible_set_variant rows')
                return dict.fromkeys(VIEWS, 0)
            views = bucket_views(rows, bucket_sets, pairs_by_bucket.get(bucket, empty_pairs))
            for key, frame in views.items():
                frame.write_parquet(
                    f'{paths[key]}/part-{bucket:05d}.parquet',
                    compression='zstd',
                    statistics=True,
                    row_group_size=ROW_GROUP_SIZE,
                )
            counts = {key: frame.height for key, frame in views.items()}
            logger.info(f'credible set bucket {bucket + 1}/{bucket_count}: {counts}')
            return counts

        built = for_each_bucket(build, bucket_count, concurrency)
    totals = {key: sum(b[key] for b in built) for key in VIEWS}
    logger.info(f'credible set views: {totals}')
