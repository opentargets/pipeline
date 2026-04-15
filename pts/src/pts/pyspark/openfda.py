"""OpenFDA adverse drug reaction signal detection.

Ported from platform-etl-backend OpenFDA step. Processes FDA FAERS data
to identify statistically significant drug-reaction associations using a
Monte Carlo method.

Scala sources ported:
    - stage/OpenFdaCompute.scala      (orchestrator)
    - stage/PrePrepRawFdaData.scala   (pre-prep)
    - stage/PrepareAdverseEventData.scala
    - stage/PrepareDrugList.scala
    - stage/PrepareBlacklistData.scala
    - stage/EventsFiltering.scala
    - stage/PrepareSummaryStatistics.scala
    - stage/PrepareForMontecarlo.scala
    - stage/AttachMeddraData.scala
    - stage/MonteCarloSampling.scala
    - utils/MathUtils.scala           (critical-value UDF)
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import pandas as pd
import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame, Window
from pyspark.sql.types import DoubleType

from pts.pyspark.common.session import Session

# ---------------------------------------------------------------------------
# Monte Carlo UDF
# ---------------------------------------------------------------------------


def _calculate_critical_values(
    permutations: int,
    n_j: int,
    n_i: list[int],
    total: int,
    prob: float,
) -> float:
    """Compute the Monte Carlo critical value for a drug's reaction distribution.

    Ports MathUtils.calculateCriticalValues from Scala/Breeze.  Uses a seeded
    RNG (seed=0) to reproduce the Scala pipeline results.

    Args:
        permutations: Number of simulation iterations.
        n_j: Number of reports that mention this drug.
        n_i: Counts of reports per reaction (for this drug).
        total: Total number of distinct reports in the dataset.
        prob: Percentile threshold (e.g. 0.95).

    Returns:
        The ``prob``-th percentile of the per-permutation maximum LLR.
    """
    if n_j is None or total is None or not n_i or n_j <= 0 or total <= 0:
        return 0.0

    rng = random.Random(0)
    total_n = float(total)
    z = float(n_j)
    size = len(n_i)

    # Normalise probability vector
    p = [ni / total_n for ni in n_i]
    p_sum = sum(p)
    p = [pi / p_sum for pi in p]

    max_llrs: list[float] = []

    for _ in range(permutations):
        # rmultinom: sequential binomial sampling (Scala port)
        x = [0.0] * size
        remaining_n = n_j
        cumulative_p = 0.0

        for j in range(size):
            if remaining_n <= 0:
                break
            remaining_p = 1.0 - cumulative_p
            if remaining_p <= 0:
                break
            p_j = p[j] / remaining_p if remaining_p > 0 else 0.0
            p_j = min(max(p_j, 0.0), 1.0)
            # Binomial sample
            k = sum(1 for _ in range(remaining_n) if rng.random() < p_j)
            x[j] = float(k)
            remaining_n -= k
            cumulative_p += p[j]

        # Compute LLR for this permutation.
        # Each reaction contributes two independent terms; compute each only when valid.
        # When xi=0, term1 vanishes (0*log(0)=0) but term2 must still be included.
        perm_llrs: list[float] = []
        for j in range(size):
            xi = x[j]
            yi = float(n_i[j])
            c = z - xi
            t1 = xi * (math.log(xi) - math.log(yi)) if xi > 0 and yi > 0 else 0.0
            t2 = c * (math.log(c) - math.log(total_n - yi)) if c > 0 and total_n - yi > 0 else 0.0
            perm_llrs.append(t1 + t2)

        max_llr = max(perm_llrs) if perm_llrs else 0.0
        max_llr += -z * math.log(z) + z * math.log(total_n) if z > 0 and total_n > 0 else 0.0
        max_llrs.append(max_llr if not (math.isnan(max_llr) or math.isinf(max_llr)) else 0.0)

    if not max_llrs:
        return 0.0

    max_llrs_sorted = sorted(max_llrs)
    idx = min(int(prob * len(max_llrs_sorted)), len(max_llrs_sorted) - 1)
    return float(max_llrs_sorted[idx])


@f.pandas_udf(DoubleType())
def _critical_values_pandas_udf(
    permutations_s: pd.Series,
    n_j_s: pd.Series,
    n_i_s: pd.Series,
    total_s: pd.Series,
    prob_s: pd.Series,
) -> pd.Series:
    """Vectorised pandas UDF for Monte Carlo critical value computation.

    Processes rows in batches and uses numpy to vectorise the inner simulation:
    - multinomial sampling replaces the sequential binomial for-loop
    - LLR is computed with numpy broadcasting across all permutations at once
    """

    def _single(permutations: int, n_j, n_i, total, prob: float) -> float:
        if n_j is None or total is None or n_i is None or len(n_i) == 0 or n_j <= 0 or total <= 0:
            return 0.0

        rng = np.random.default_rng(0)
        total_n = float(total)
        z = float(n_j)
        n_i_arr = np.array(n_i, dtype=np.float64)

        p = n_i_arr / total_n
        p_sum = p.sum()
        if p_sum <= 0:
            return 0.0
        p /= p_sum

        # All permutation samples at once: shape (permutations, size)
        samples = rng.multinomial(int(n_j), p, size=permutations).astype(np.float64)

        yi = n_i_arr  # (size,)
        xi = samples  # (permutations, size)

        # Compute LLR as sum of two independent terms.
        # term1 = xi*log(xi/yi):  valid when xi>0 and yi>0
        # term2 = (z-xi)*log((z-xi)/(total_n-yi)):  valid when z-xi>0 and total_n-yi>0
        # When xi=0 term1 vanishes by convention (0*log(0)=0) and term2 must still be
        # computed correctly — the old code incorrectly set the whole inner sum to 0,
        # which caused the constant correction to dominate every permutation.
        t1_v = (xi > 0) & (yi > 0)
        t2_v = ((z - xi) > 0) & ((total_n - yi) > 0)
        term1 = np.where(t1_v, xi * (np.log(np.where(t1_v, xi, 1.0)) - np.log(yi)), 0.0)
        term2 = np.where(t2_v, (z - xi) * (np.log(np.where(t2_v, z - xi, 1.0)) - np.log(total_n - yi)), 0.0)

        max_llrs = (term1 + term2).max(axis=1)
        max_llrs += -z * np.log(z) + z * np.log(total_n)
        max_llrs = np.where(np.isfinite(max_llrs), max_llrs, 0.0)

        idx = min(int(prob * permutations), permutations - 1)
        return float(np.sort(max_llrs)[idx])

    return pd.Series([
        _single(perm, n_j, n_i, total, prob)
        for perm, n_j, n_i, total, prob in zip(permutations_s, n_j_s, n_i_s, total_s, prob_s, strict=True)
    ])


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------


def _prepare_drug_list(chembl_df: DataFrame) -> DataFrame:
    """Build (chembl_id, drug_name) lookup from ChEMBL drug molecule table.

    Ports PrepareDrugList.scala.

    Args:
        chembl_df: Drug molecule DataFrame with columns id, name, synonyms,
            tradeNames.

    Returns:
        Deduplicated DataFrame with columns chembl_id and drug_name (lowercase).
    """
    return (
        chembl_df
        .selectExpr(
            'id as chembl_id',
            'synonyms',
            'name as pref_name',
            'tradeNames as trade_names',
        )
        .withColumn(
            'drug_names',
            f.array_distinct(f.flatten(f.array(
                f.col('trade_names'),
                f.array(f.col('pref_name')),
                f.col('synonyms'),
            ))),
        )
        .withColumn('_drug_name', f.explode(f.col('drug_names')))
        .withColumn('drug_name', f.lower(f.col('_drug_name')))
        .select('chembl_id', 'drug_name')
        .distinct()
        .orderBy('drug_name')
    )


def _prepare_blacklist(df: DataFrame) -> DataFrame:
    """Normalise the blacklisted reaction names.

    Ports PrepareBlacklistData.scala.

    Args:
        df: DataFrame with a single column named ``reactions``.

    Returns:
        DataFrame with normalised (trimmed, lowercased) reaction names.
    """
    return (
        df
        .toDF('reactions')
        .withColumn('reactions', f.translate(f.trim(f.lower(f.col('reactions'))), '^', "\\'"))
        .orderBy(f.col('reactions').asc())
    )


def _pre_prep_raw_fda_data(fda_df: DataFrame) -> DataFrame:
    """Reduce raw FAERS records to essential columns.

    Ports PrePrepRawFdaData.scala.

    Args:
        fda_df: Raw FAERS parquet DataFrame.

    Returns:
        DataFrame with safetyreportid, serious, seriousnessdeath, receivedate,
        qualification and patient columns.
    """
    return fda_df.selectExpr(
        'safetyreportid',
        'serious',
        'seriousnessdeath',
        'receivedate',
        'primarysource.qualification as qualification',
        'patient',
    )


def _prepare_adverse_event_data(
    fda_pre_prepped: DataFrame,
    drug_list: DataFrame,
    blacklist: DataFrame,
) -> DataFrame:
    """Clean FAERS data, link to ChEMBL drugs, filter blacklisted reactions.

    Ports PrepareAdverseEventData.scala + EventsFiltering.scala.

    Args:
        fda_pre_prepped: Pre-prepped FAERS DataFrame (from _pre_prep_raw_fda_data)
            OR already-flattened DataFrame with individual reaction+drug columns.
        drug_list: ChEMBL (chembl_id, drug_name) lookup.
        blacklist: Normalised blacklist with ``reactions`` column.

    Returns:
        DataFrame with one row per (report, drug, reaction) triplet linked to a
        ChEMBL ID, with dead rows filtered out.
    """
    # If the DataFrame has a 'patient' struct (raw form), explode it.
    # Otherwise assume it is already in flat per-reaction/drug form.
    if 'patient' in fda_pre_prepped.columns:
        flat = (
            fda_pre_prepped
            .withColumn('reaction', f.explode(f.col('patient.reaction')))
            .withColumn('drug', f.explode(f.col('patient.drug')))
            .selectExpr(
                'safetyreportid',
                'serious',
                'receivedate',
                "ifnull(seriousnessdeath, '0') as seriousness_death",
                'qualification',
                "trim(translate(lower(reaction.reactionmeddrapt), '^', '\\'')) as reaction_reactionmeddrapt",
                "ifnull(lower(drug.medicinalproduct), '') as drug_medicinalproduct",
                'ifnull(drug.openfda.generic_name, array()) as drug_generic_name_list',
                'ifnull(drug.openfda.brand_name, array()) as drug_brand_name_list',
                'ifnull(drug.openfda.substance_name, array()) as drug_substance_name_list',
                'drug.drugcharacterization as drugcharacterization',
            )
        )
    else:
        # Already flat: rename seriousnessdeath if present
        flat = fda_pre_prepped.withColumnRenamed('seriousnessdeath', 'seriousness_death') \
            if 'seriousnessdeath' in fda_pre_prepped.columns else fda_pre_prepped

    filtered = (
        flat
        .where(
            f.col('qualification').isin('1', '2', '3')
            & (f.col('drugcharacterization') == '1')
        )
        .withColumn(
            'drug_names',
            f.array_distinct(f.concat(
                f.col('drug_brand_name_list'),
                f.array(f.col('drug_medicinalproduct')),
                f.col('drug_generic_name_list'),
                f.col('drug_substance_name_list'),
            )),
        )
        .withColumn('_drug_name', f.explode(f.col('drug_names')))
        .withColumn('drug_name', f.lower(f.col('_drug_name')))
        .drop('drug_generic_name_list', 'drug_substance_name_list', '_drug_name')
        .where(
            f.col('drug_name').isNotNull()
            & f.col('reaction_reactionmeddrapt').isNotNull()
            & f.col('safetyreportid').isNotNull()
            & (f.col('seriousness_death') == '0')
            & (f.length(f.col('drug_name')) > 0)
        )
    )

    # Anti-join blacklisted reactions
    not_blacklisted = filtered.join(
        blacklist,
        filtered['reaction_reactionmeddrapt'] == blacklist['reactions'],
        'left_anti',
    )

    # Inner join with ChEMBL drug list
    return not_blacklisted.join(drug_list, 'drug_name', 'inner')


def _prepare_summary_statistics(
    fda_data: DataFrame,
    target_col_id: str,
    target_stats_col_id: str,
) -> DataFrame:
    """Compute per-reaction, per-drug and per-pair report counts.

    Ports PrepareSummaryStatistics.scala.

    Args:
        fda_data: Cleaned (chembl_id, reaction, safetyreportid) DataFrame.
        target_col_id: Name of the drug dimension column (e.g. 'chembl_id').
        target_stats_col_id: Name for the drug-count column (e.g. 'chembl_id_stats').

    Returns:
        DataFrame with safetyreportid, reaction, target_col_id,
        uniq_report_ids_by_reaction, target_stats_col_id, uniq_report_ids.
    """
    report_id = f.col('safetyreportid')
    ae = f.col('reaction_reactionmeddrapt')

    w_reaction = Window.partitionBy(ae)
    w_target = Window.partitionBy(f.col(target_col_id))
    w_pair = Window.partitionBy(f.col(target_col_id), ae)

    return (
        fda_data
        .withColumn('uniq_report_ids_by_reaction', f.approx_count_distinct(report_id).over(w_reaction))
        .withColumn(target_stats_col_id, f.approx_count_distinct(report_id).over(w_target))
        .withColumn('uniq_report_ids', f.approx_count_distinct(report_id).over(w_pair))
        .select(
            'safetyreportid',
            'reaction_reactionmeddrapt',
            'uniq_report_ids_by_reaction',
            target_stats_col_id,
            'uniq_report_ids',
            target_col_id,
        )
    )


def _prepare_for_montecarlo(fda_stats: DataFrame, target_stats_col_id: str) -> DataFrame:
    """Build contingency tables and compute log-likelihood ratios.

    Ports PrepareForMontecarlo.scala.

    Args:
        fda_stats: DataFrame from _prepare_summary_statistics.
        target_stats_col_id: Name of the drug-count column.

    Returns:
        DataFrame with A, B, C, D, llr and the reaction/drug columns.
    """
    total_reports: int = fda_stats.select('safetyreportid').distinct().count()

    return (
        fda_stats
        .drop('safetyreportid')
        .withColumnRenamed('uniq_report_ids', 'A')
        .withColumn('C', f.col(target_stats_col_id) - f.col('A'))
        .withColumn('B', f.col('uniq_report_ids_by_reaction') - f.col('A'))
        .withColumn(
            'D',
            f.lit(total_reports)
            - f.col(target_stats_col_id)
            - f.col('uniq_report_ids_by_reaction')
            + f.col('A'),
        )
        .withColumn('aterm', f.col('A') * (f.log(f.col('A')) - f.log(f.col('A') + f.col('B'))))
        .withColumn('cterm', f.col('C') * (f.log(f.col('C')) - f.log(f.col('C') + f.col('D'))))
        .withColumn(
            'acterm',
            (f.col('A') + f.col('C'))
            * (f.log(f.col('A') + f.col('C')) - f.log(f.col('A') + f.col('B') + f.col('C') + f.col('D'))),
        )
        .withColumn('llr', f.col('aterm') + f.col('cterm') - f.col('acterm'))
        .distinct()
        .where(f.col('llr').isNotNull() & ~f.isnan(f.col('llr')))
    )


def _attach_meddra(
    fda_data: DataFrame,
    target_col_id: str,
    meddra_preferred: DataFrame,
    meddra_low_level: DataFrame,
) -> DataFrame:
    """Enrich reactions with MedDRA codes.

    Ports AttachMeddraData.scala.

    Args:
        fda_data: Montecarlo-ready DataFrame.
        target_col_id: Drug dimension column name.
        meddra_preferred: Raw preferred terms DataFrame (single '_c0' column).
        meddra_low_level: Raw low-level terms DataFrame (single '_c0' column).

    Returns:
        DataFrame with an added meddraCode column.
    """
    def _parse_meddra(df: DataFrame, cols: list[str]) -> DataFrame:
        parsed = (
            df
            .withColumn('_c0', f.regexp_replace(f.col('_c0'), r'\$+', ','))
            .withColumn('_c0', f.regexp_replace(f.col('_c0'), r'\$$', ''))
            .withColumn('_c0', f.split(f.col('_c0'), ','))
            .select([f.col('_c0').getItem(i).alias(col) for i, col in enumerate(cols)])
        )
        name_cols = [c for c in cols if 'name' in c]
        for c in name_cols:
            parsed = parsed.withColumn(c, f.lower(f.col(c)))
        return parsed

    pt_terms = _parse_meddra(meddra_preferred, ['pt_code', 'pt_name'])
    llt_terms = _parse_meddra(meddra_low_level, ['llt_code', 'llt_name'])

    with_preferred = fda_data.join(
        pt_terms,
        fda_data['reaction_reactionmeddrapt'] == pt_terms['pt_name'],
        'left_outer',
    )
    with_both = with_preferred.join(
        llt_terms,
        with_preferred['reaction_reactionmeddrapt'] == llt_terms['llt_name'],
        'left_outer',
    )
    return (
        with_both
        .withColumn('meddraCode', f.coalesce(f.col('pt_code'), f.col('llt_code')))
        .drop('pt_name', 'llt_name', 'pt_code', 'llt_code')
        .dropDuplicates([target_col_id, 'reaction_reactionmeddrapt'])
    )


def _run_montecarlo(
    fda_data: DataFrame,
    target_col_id: str,
    target_stats_col_id: str,
    percentile: float,
    permutations: int,
) -> DataFrame:
    """Apply Monte Carlo significance test and return significant drug-reaction pairs.

    Ports MonteCarloSampling.scala.

    Args:
        fda_data: DataFrame with meddraCode column added.
        target_col_id: Drug dimension column name.
        target_stats_col_id: Drug-count column name.
        percentile: Percentile for critical value threshold (e.g. 0.95).
        permutations: Number of Monte Carlo iterations.

    Returns:
        DataFrame with columns chembl_id, event, count, llr, critval, meddraCode.
    """
    crit_val = (
        fda_data
        .withColumn('uniq_reports_total', f.col('A') + f.col('B') + f.col('C') + f.col('D'))
        .withColumn('uniq_report_ids', f.col('A'))
        .groupBy(f.col(target_col_id))
        .agg(
            f.first('uniq_reports_total').alias('uniq_reports_total'),
            f.collect_list('uniq_report_ids').alias('uniq_reports_combined'),
            f.collect_list('uniq_report_ids_by_reaction').alias('n_i'),
            f.first(f.col(target_stats_col_id)).alias(target_stats_col_id),
        )
        .withColumn(
            'criticalValue',
            _critical_values_pandas_udf(
                f.lit(permutations),
                f.col(target_stats_col_id).cast('int'),
                f.col('n_i'),
                f.col('uniq_reports_total').cast('int'),
                f.lit(percentile),
            ),
        )
        .select(target_col_id, 'criticalValue')
    )

    return (
        fda_data
        .join(crit_val, target_col_id, 'inner')
        .where(
            (f.col('llr') > f.col('criticalValue'))
            & (f.col('criticalValue') > 0)
        )
        .selectExpr(
            f'{target_col_id}',
            'reaction_reactionmeddrapt as event',
            'A as count',
            'llr',
            'criticalValue as critval',
            'meddraCode',
        )
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def openfda(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Run the OpenFDA adverse drug reaction pipeline.

    Reads pre-extracted FAERS parquet files, links drug records to ChEMBL,
    computes log-likelihood ratios and applies a Monte Carlo significance
    filter to produce a dataset of significant drug-reaction signals.

    Args:
        source: Path mapping with keys:
            - drug_molecule: ChEMBL drug molecule parquet
            - fda_data: Pre-extracted FAERS parquet glob (intermediate/openfda/*)
            - blacklisted_events: TSV of blacklisted reaction terms
            - meddra_preferred_terms: (optional) MedDRA preferred terms .asc file
            - meddra_low_level_terms: (optional) MedDRA low-level terms .asc file
        destination: Path mapping with keys:
            - fda_unfiltered: Output path for all enriched pairs
            - fda_results: Output path for significant pairs only
        settings: Configuration values:
            - montecarlo_permutations: Number of MC iterations (default 100)
            - montecarlo_percentile: Percentile threshold (default 0.95)
        properties: Spark session properties.
    """
    spark = Session(app_name='openfda', properties=properties).spark

    permutations: int = int(settings.get('montecarlo_permutations', 100))
    percentile: float = float(settings.get('montecarlo_percentile', 0.95))

    target_col_id = 'chembl_id'
    target_stats_col_id = 'chembl_id_stats'

    logger.info('Loading input data')
    chembl_df = spark.read.parquet(source['drug_molecule'])
    fda_raw = spark.read.parquet(source['fda_data'])
    blacklist_raw = (
        spark.read
        .option('sep', '\t')
        .option('ignoreLeadingWhiteSpace', 'true')
        .option('ignoreTrailingWhiteSpace', 'true')
        .csv(source['blacklisted_events'])
    )

    logger.info('Preparing drug list')
    drug_list = _prepare_drug_list(chembl_df)

    logger.info('Preparing blacklist')
    blacklist = _prepare_blacklist(blacklist_raw)

    logger.info('Pre-prepping raw FAERS data')
    fda_pre_prepped = _pre_prep_raw_fda_data(fda_raw)

    logger.info('Preparing adverse event data')
    fda_cooked = _prepare_adverse_event_data(fda_pre_prepped, drug_list, blacklist)

    logger.info('Preparing summary statistics')
    fda_stats = _prepare_summary_statistics(fda_cooked, target_col_id, target_stats_col_id)

    logger.info('Preparing for Monte Carlo')
    fda_mc_ready = _prepare_for_montecarlo(fda_stats, target_stats_col_id)

    # Optional MedDRA enrichment
    if 'meddra_preferred_terms' in source and 'meddra_low_level_terms' in source:
        logger.info('Attaching MedDRA data')
        meddra_preferred = spark.read.csv(source['meddra_preferred_terms'])
        meddra_low_level = spark.read.csv(source['meddra_low_level_terms'])
        fda_with_meddra = _attach_meddra(fda_mc_ready, target_col_id, meddra_preferred, meddra_low_level)
    else:
        logger.info('No MedDRA data provided; adding empty meddraCode column')
        fda_with_meddra = fda_mc_ready.withColumn('meddraCode', f.lit(''))

    logger.info('Writing unfiltered output')
    fda_with_meddra.write.mode('overwrite').parquet(destination['fda_unfiltered'])

    logger.info('Running Monte Carlo sampling')
    mc_results = _run_montecarlo(
        fda_with_meddra, target_col_id, target_stats_col_id, percentile, permutations
    )

    logger.info('Writing significant results')
    mc_results.coalesce(1).write.mode('overwrite').parquet(destination['fda_results'])
