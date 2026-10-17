"""Coding variant view.

Ported from the pyspark coding_variant step. Lists every amino acid changing consequence of a
variant that has evidence, with the diseases, therapeutic areas and datasources behind it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.coding_variant import datasource
from pts.transformers.utils.dataset import scan_dataset, write_dataset

HARMONIC_SUM_CONSTANT = sum(1 / ((i + 1) ** 2) for i in range(1000))

DATASOURCE_NICE_NAMES = {
    'eva': 'ClinVar',
    'eva_somatic': 'ClinVar-somatic',
    'uniprot_variants': 'UniProt Variants',
    'mol_qtl': 'molQTL',
    'gwas_credible_sets': 'GWAS credible sets',
}

EVIDENCE_SOURCES = ['evidence_eva', 'evidence_eva_somatic', 'evidence_uniprot_variant']

OUTPUT_COLUMNS = [
    'targetId',
    'variantId',
    'therapeuticAreas',
    'diseases',
    'datasources',
    'variantEffect',
    'variantFunctionalConsequenceIds',
    'uniprotAccessions',
    'referenceAminoAcid',
    'aminoAcidPosition',
    'alternateAminoAcid',
]

SORT_COLUMNS = [
    'variantId',
    'targetId',
    'aminoAcidPosition',
    'referenceAminoAcid',
    'alternateAminoAcid',
    'variantEffect',
    'variantFunctionalConsequenceIds',
    'uniprotAccessions',
]


def coding_variant(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Generate the view on coding variants with their functional context.

    Args:
        source: paths keyed `variant`, `target`, `disease`, `evidence_eva`, `evidence_eva_somatic`,
            `evidence_uniprot_variant`, `evidence_gwas_credible_set`, `credible_set` and
            `pharmacogenetics`.
        destination: output directory.
        settings: unused.
        config: unused.
    """
    variants = read_variants(source['variant'])
    frames = {name: scan_dataset(path) for name, path in source.items() if name != 'variant'}
    write_dataset(build_view(variants, frames), destination)
    logger.info(f'wrote coding variant view to {destination}')


def read_variants(path: str) -> pl.LazyFrame:
    """Read the amino acid changing consequences of the variant dataset, one part at a time.

    Args:
        path: the variant dataset, a file or a directory of parts.

    Returns:
        The `process_variants` rows of every part.
    """
    handle = StorageHandle(path)
    parts = sorted(handle.glob('*.parquet')) if handle.stat().is_dir else [path]
    return pl.concat([process_variants(scan_dataset(part)).collect() for part in parts]).lazy()


def build_view(variants: pl.LazyFrame, frames: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Build the coding variant view from its input datasets.

    Args:
        variants: the `process_variants` rows.
        frames: the other input datasets, keyed as the transformer's `source`.

    Returns:
        One row per amino acid changing consequence of a variant with evidence.
    """
    variant_ids = variants.select('variantId')
    diseases = process_diseases(frames['disease'])
    pooled = pl.concat(
        [
            process_evidence(diseases, [frames[name] for name in EVIDENCE_SOURCES]),
            process_gwas_associations(
                variant_ids, diseases, frames['evidence_gwas_credible_set'], frames['credible_set']
            ),
            process_pharmacogenetics(frames['pharmacogenetics']),
            process_qtls(frames['credible_set'], variant_ids),
        ],
        how='diagonal_relaxed',
    )
    empty_datasources = pl.lit([], dtype=pl.List(datasource))
    return (
        variants
        .join(aggregate_evidence(pooled), on='variantId', how='inner')
        .with_columns(
            pl.when(pl.col('hasEvidence')).then(pl.col('datasources')).otherwise(empty_datasources).alias('datasources')
        )
        .join(process_target_index(frames['target']), on='targetId', how='left')
        .with_columns(uniprot_accessions(pl.col('uniprotAccessions'), pl.col('refUniprotAccessions')))
        .select(OUTPUT_COLUMNS)
        .sort(SORT_COLUMNS, nulls_last=True, maintain_order=True)
    )


def aggregate_evidence(pooled: pl.LazyFrame) -> pl.LazyFrame:
    """Summarise the pooled evidence per variant.

    Args:
        pooled: evidence rows from every source.

    Returns:
        Per variant: sorted `therapeuticAreas`, `diseases` by harmonic sum, `datasources` and
        `hasEvidence`, false when every evidence scored zero.
    """
    diseases = (
        pooled
        .filter(pl.col('diseaseId').is_not_null())
        .group_by('variantId', 'diseaseId')
        .agg(harmonic_sum(pl.col('score')).alias('harmonicSum'))
        .group_by('variantId')
        .agg(
            pl
            .col('diseaseId')
            .sort_by('harmonicSum', 'diseaseId', descending=[True, False], nulls_last=True)
            .alias('diseases')
        )
    )
    datasources = (
        pooled
        .group_by('variantId', 'datasourceId')
        .agg(pl.col('evidenceId').drop_nulls().n_unique().cast(pl.Int32).alias('datasourceCount'))
        .group_by('variantId')
        .agg(
            pl
            .struct(
                'datasourceCount',
                'datasourceId',
                pl
                .col('datasourceId')
                .replace_strict(DATASOURCE_NICE_NAMES, default=pl.col('datasourceId'), return_dtype=pl.String)
                .alias('datasourceNiceName'),
            )
            .sort_by(pl.col('datasourceId'))
            .alias('datasources')
        )
    )
    return (
        pooled
        .group_by('variantId')
        .agg(
            pl.col('therapeuticAreas').list.explode(keep_nulls=False, empty_as_null=False).unique().sort(),
            pl.col('zeroEvidence').not_().any().alias('hasEvidence'),
        )
        .join(diseases, on='variantId', how='left')
        .join(datasources, on='variantId', how='left')
        .with_columns(pl.col('diseases').fill_null([]))
    )


def uniprot_accessions(variant: pl.Expr, target: pl.Expr) -> pl.Expr:
    """Intersect the consequence's accessions with the target's, sorted by length then value.

    Args:
        variant: accessions of the consequence.
        target: swissprot accessions of the target.

    Returns:
        The intersection when both are present, otherwise whichever is.
    """
    return (
        pl
        .when(variant.is_not_null() & target.is_not_null())
        .then(target.list.set_intersection(variant))
        .otherwise(pl.coalesce(variant, target))
        .list.eval(
            pl.struct(pl.element().str.len_chars().alias('length'), pl.element().alias('id')).sort().struct.field('id')
        )
        .alias('uniprotAccessions')
    )


def process_target_index(target: pl.LazyFrame) -> pl.LazyFrame:
    """Map each target to its swissprot accessions.

    Args:
        target: the target dataset.

    Returns:
        `targetId` and `refUniprotAccessions`; targets whose proteins hold no swissprot accession
        are dropped, targets without proteins are kept with null accessions.
    """
    accessions = pl.col('refUniprotAccessions')
    return target.select(
        pl.col('id').alias('targetId'),
        pl
        .col('proteinIds')
        .list.filter(pl.element().struct.field('source') == 'uniprot_swissprot')
        .list.eval(pl.element().struct.field('id'))
        .alias('refUniprotAccessions'),
    ).filter(accessions.list.len().fill_null(-1) != 0)


def _harmonic_terms(series: pl.Series) -> pl.Series:
    scores = series.struct.field('score').to_numpy()
    positions = series.struct.field('position').to_numpy().astype(np.float64)
    return pl.Series(scores / (positions * positions) / HARMONIC_SUM_CONSTANT, dtype=pl.Float64)


def harmonic_sum(scores: pl.Expr) -> pl.Expr:
    """Harmonic sum of the scores, `score / position**2 / HARMONIC_SUM_CONSTANT` in descending order.

    Args:
        scores: the scores to sum; nulls are ignored.

    Returns:
        The sum, or null when there is no score.
    """
    ordered = scores.drop_nulls().sort(descending=True)
    terms = pl.struct(
        ordered.alias('score'),
        pl.int_range(1, ordered.len() + 1).alias('position'),
    ).map_batches(_harmonic_terms, return_dtype=pl.Float64, is_elementwise=True)
    return terms.cum_sum().last()


def parse_amino_acid_change(aa: pl.Expr) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Extract the reference amino acid, position and alternate amino acid from the change string.

    Args:
        aa: the amino acid change, e.g. `A12V`.

    Returns:
        `referenceAminoAcid`, `aminoAcidPosition` and `alternateAminoAcid`.
    """
    empty = aa.str.slice(0, 0)
    ref_aa = pl.coalesce(aa.str.extract(r'^([-\*A-Za-z]+)', 1), empty).alias('referenceAminoAcid')
    alt_aa = pl.coalesce(
        aa.str.extract(r'([-\*A-Zaz+]+)(?:\r\n|[\n\r\x{85}\x{2028}\x{2029}])?$', 1), empty
    ).alias('alternateAminoAcid')
    position = aa.str.extract(r'([0-9]+)', 1).cast(pl.Int32, strict=False).alias('aminoAcidPosition')
    return (ref_aa, position, alt_aa)


def process_variants(variant: pl.LazyFrame) -> pl.LazyFrame:
    """Get every amino acid changing consequence of each variant.

    Args:
        variant: the variant dataset.

    Returns:
        One row per transcript consequence with an amino acid change.
    """
    consequence = pl.col('consequence')
    return (
        variant
        .select(
            'variantId',
            pl
            .col('variantEffect')
            .list.filter(pl.element().struct.field('method') == 'AlphaMissense')
            .list.first()
            .struct.field('score')
            .alias('variantEffect'),
            pl
            .col('transcriptConsequences')
            .list.filter(pl.element().struct.field('aminoAcidChange').is_not_null())
            .alias('consequence'),
        )
        .filter(consequence.list.len() > 0)
        .explode('consequence')
        .select(
            'variantId',
            'variantEffect',
            consequence.struct.field('targetId'),
            consequence.struct.field('variantFunctionalConsequenceIds'),
            consequence.struct.field('uniprotAccessions'),
            consequence.struct.field('aminoAcidChange'),
        )
        .filter(pl.col('aminoAcidChange').is_not_null())
        .select(
            'variantId',
            'variantEffect',
            'targetId',
            'variantFunctionalConsequenceIds',
            'uniprotAccessions',
            *parse_amino_acid_change(pl.col('aminoAcidChange')),
        )
    )


def zero_evidence() -> pl.Expr:
    """Whether the evidence scored zero."""
    return (pl.col('score') == 0).fill_null(False).alias('zeroEvidence')


def process_evidence(diseases: pl.LazyFrame, evidence: list[pl.LazyFrame]) -> pl.LazyFrame:
    """Get all evidence supported by variants.

    Args:
        diseases: `diseaseId` and `therapeuticAreas`.
        evidence: the variant evidence datasets.

    Returns:
        Evidence rows with their therapeutic areas; zero-score evidence loses its disease.
    """
    columns = ['datasourceId', 'variantId', 'score', 'id', 'diseaseId']
    return (
        pl
        .concat([frame.select(columns) for frame in evidence], how='vertical_relaxed')
        .filter(pl.col('variantId').is_not_null())
        .select(
            'datasourceId',
            'variantId',
            'score',
            pl.col('id').alias('evidenceId'),
            zero_evidence(),
            pl.when(pl.col('score') == 0).then(None).otherwise(pl.col('diseaseId')).alias('diseaseId'),
        )
        .join(diseases, on='diseaseId', how='left')
    )


def process_pharmacogenetics(pharmacogenetics: pl.LazyFrame) -> pl.LazyFrame:
    """Get all pharmacogenetics evidence.

    Args:
        pharmacogenetics: the pharmacogenetics dataset.

    Returns:
        One row per evidence, each with its own `evidenceId`.
    """
    return (
        pharmacogenetics
        .select('variantId', 'datasourceId')
        .filter(pl.col('variantId').is_not_null())
        .with_row_index('evidenceId')
        .select(
            'variantId',
            'datasourceId',
            pl.col('evidenceId').cast(pl.String),
            pl.lit(False).alias('zeroEvidence'),
        )
    )


def explode_loci(credible_set: pl.LazyFrame, variant_ids: pl.LazyFrame) -> pl.LazyFrame:
    """One row per credible set locus entry whose variant is in `variant_ids`.

    Args:
        credible_set: the credible set dataset.
        variant_ids: `variantId` of the coding variants.

    Returns:
        `studyLocusId`, `studyType` and `variantId`.
    """
    return (
        credible_set
        .select(
            'studyLocusId',
            'studyType',
            pl.col('locus').list.eval(pl.element().struct.field('variantId')).alias('variantId'),
        )
        .filter(pl.col('variantId').list.len() > 0)
        .explode('variantId')
        .join(variant_ids.unique(), on='variantId', how='semi')
    )


def process_qtls(credible_set: pl.LazyFrame, variant_ids: pl.LazyFrame) -> pl.LazyFrame:
    """Get all molecular QTL evidence on the coding variants.

    Args:
        credible_set: the credible set dataset.
        variant_ids: `variantId` of the coding variants.

    Returns:
        One row per QTL locus entry, each with its own `evidenceId`.
    """
    return (
        explode_loci(credible_set, variant_ids)
        .filter(pl.col('studyType') != 'gwas')
        .with_row_index('evidenceId')
        .select(
            'variantId',
            pl.lit('mol_qtl').alias('datasourceId'),
            pl.col('evidenceId').cast(pl.String),
            pl.lit(False).alias('zeroEvidence'),
        )
    )


def process_gwas_associations(
    variant_ids: pl.LazyFrame,
    diseases: pl.LazyFrame,
    gwas_evidence: pl.LazyFrame,
    credible_set: pl.LazyFrame,
) -> pl.LazyFrame:
    """Get the strongest GWAS evidence of each credible set on its coding variants.

    Args:
        variant_ids: `variantId` of the coding variants, one row per consequence.
        diseases: `diseaseId` and `therapeuticAreas`.
        gwas_evidence: the GWAS credible set evidence dataset.
        credible_set: the credible set dataset.

    Returns:
        One row per top evidence, locus variant and coding consequence of that variant.
    """
    score = pl.col('score')
    return (
        gwas_evidence
        .select('studyLocusId', 'diseaseId', 'score', 'datasourceId')
        .filter(score == score.max().over('studyLocusId', 'diseaseId'))
        .join(explode_loci(credible_set, variant_ids).select('studyLocusId', 'variantId'), on='studyLocusId')
        .select(
            'variantId',
            'diseaseId',
            'score',
            'datasourceId',
            pl.col('studyLocusId').alias('evidenceId'),
            zero_evidence(),
        )
        .join(variant_ids, on='variantId')
        .join(diseases, on='diseaseId', how='left')
    )


def process_diseases(disease: pl.LazyFrame) -> pl.LazyFrame:
    """Get the therapeutic areas of every disease.

    Args:
        disease: the disease dataset.

    Returns:
        `diseaseId` and `therapeuticAreas`.
    """
    return disease.select(pl.col('id').alias('diseaseId'), 'therapeuticAreas')
