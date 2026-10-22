"""Target prioritisation features.

Computes per-target features for target prioritisation: membrane location and secretion, safety
events, tractability, genetic constraint, paralog and mouse ortholog identity, cancer driver
status, mouse knockout score, chemical probes, clinical stage and baseline expression specificity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.schemas.outputs.target_prioritisation import TargetPrioritisation
from pts.transformers.utils.dataset import scan_dataset, write_dataset

ONCOTSG = ['TSG', 'oncogene', 'Oncogene', 'oncogene,TSG', 'TSG,oncogene', 'fusion,oncogene', 'oncogene,fusion']
MEMBRANE_SOURCES = ['HPA_1', 'HPA_secreted', 'HPA_add_1', 'uniprot_1', 'uniprot_secreted', 'HPA_dif']
LIGAND_POCKET_CATEGORIES = ['High-Quality Ligand', 'High-Quality Pocket', 'Small Molecule Binder']
STAGE_SCORES = {
    'APPROVAL': 1.0,
    'PREAPPROVAL': 0.8,
    'PHASE_3': 0.7,
    'PHASE_2_3': 0.5,
    'PHASE_2': 0.2,
    'PHASE_1_2': 0.15,
    'PHASE_1': 0.1,
    'EARLY_PHASE_1': 0.05,
    'IND': 0.05,
    'PRECLINICAL': 0.01,
    'UNKNOWN': 0.01,
}
LOW_THRESHOLD = 0.6
CAST_TRIM = ''.join(chr(c) for c in range(0x21))
BASELINE_DATASOURCES = ['gtex', 'tabula_sapiens']
BASELINE_COLUMNS = [
    'targetId',
    'datasourceId',
    'tissueBiosampleId',
    'celltypeBiosampleId',
    'specificity_score',
    'distribution_score',
]

#: Output column -> internal column, in output order.
OUTPUT = {
    'targetId': 'targetid',
    'isInMembrane': 'Nr_mb',
    'isSecreted': 'Nr_secreted',
    'hasSafetyEvent': 'Nr_Event',
    'hasPocket': 'Nr_Pocket',
    'hasLigand': 'Nr_Ligand',
    'hasSmallMoleculeBinder': 'Nr_sMBinder',
    'geneticConstraint': 'cal_score',
    'paralogMaxIdentityPercentage': 'Nr_paralogs',
    'mouseOrthologMaxIdentityPercentage': 'Nr_ortholog',
    'isCancerDriverGene': 'Nr_CDG',
    'mouseKOScore': 'negScaledHarmonicSum',
    'hasHighQualityChemicalProbes': 'Nr_chprob',
    'maxClinicalStage': 'inClinicalTrials',
    'tissueSpecificity': 'Nr_specificity',
    'tissueDistribution': 'Nr_distribution',
    'celltypeSpecificity': 'Nr_ct_specificity',
    'celltypeDistribution': 'Nr_ct_distribution',
}


def int32(value: int) -> pl.Expr:
    """An Int32 literal."""
    return pl.lit(value, dtype=pl.Int32)


def cast_double(column: str) -> pl.Expr:
    """Cast a text column to double as spark's non-ANSI cast does, giving null on bad text."""
    value = pl.col(column).str.strip_chars(CAST_TRIM)
    number = value.str.replace(r'^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)[dDfF]$', '$1')
    signed_nan = value.str.contains(r'^[+-][nN][aA][nN]$') & ~value.str.contains(r'^[+-]NaN$')
    return pl.when(~signed_nan).then(number.cast(pl.Float64, strict=False)).alias(column)


def divide(numerator: pl.Expr, denominator: float) -> pl.Expr:
    """Divide by a constant through numpy, keeping nulls."""
    return numerator.cast(pl.Float64).map_batches(lambda s: np.divide(s, denominator), return_dtype=pl.Float64)


def explode(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Explode a list column, dropping rows whose list is null or empty."""
    return df.explode(column, empty_as_null=False, keep_nulls=False)


def explode_outer(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Explode a list column, keeping one null row for a null or empty list."""
    return df.explode(column, empty_as_null=True, keep_nulls=True)


def join_list(column: str) -> pl.Expr:
    """Join a list column with commas; null when the list is null or empty."""
    return pl.when(pl.col(column).list.len() > 0).then(pl.col(column).list.join(','))


def compute_target_prioritisation(
    targets: pl.DataFrame,
    tractability: pl.DataFrame,
    target_safety_event: pl.DataFrame,
    chemical_probes: pl.DataFrame,
    homology: pl.DataFrame,
    mouse_phenotypes: pl.DataFrame,
    molecule: pl.DataFrame,
    mechanism_of_action: pl.DataFrame,
    baseline_expression: pl.DataFrame | pl.LazyFrame,
    uniprot_slterms: pl.DataFrame,
    mouse_pheno_scores: pl.DataFrame,
) -> pl.DataFrame:
    """Compute all target prioritisation features.

    Args:
        targets: Target data.
        tractability: target_tractability data (targetId, modality, category, value).
        target_safety_event: target_safety_event data (one row per event record).
        chemical_probes: chemical_probes data (one row per probe/target).
        homology: homology data (one row per target/homologue pair).
        mouse_phenotypes: Mouse phenotype data.
        molecule: Drug molecule data.
        mechanism_of_action: Mechanism of action data.
        baseline_expression: Aggregated baseline expression with specificity/distribution scores per
            (target, biosample).
        uniprot_slterms: UniProt subcellular location terms.
        mouse_pheno_scores: Mouse phenotype scores, with a double `score`.

    Returns:
        One row per target with the 18 target prioritisation columns, sorted.
    """
    targets = targets.rechunk()
    parent_child_cousins = find_parent_child_cousins(uniprot_slterms.rechunk())
    queryset = targets.select(pl.col('id').alias('targetid'))

    result = (
        queryset
        .pipe(biotype_query, targets)
        .pipe(target_membrane_query, targets, parent_child_cousins)
        .pipe(ligand_pocket_query, tractability.rechunk())
        .pipe(safety_query, target_safety_event.rechunk())
        .pipe(constraint_query, targets)
        .pipe(paralogs_query, homology.rechunk())
        .pipe(orthologs_mouse_query, homology.rechunk())
        .pipe(driver_gene_query, targets)
        .pipe(mouse_model_query, mouse_phenotypes.rechunk(), mouse_pheno_scores.rechunk())
        .pipe(chemical_probes_query, chemical_probes.rechunk())
        .pipe(clin_trials_query, mechanism_of_action.rechunk(), molecule.rechunk())
        .pipe(baseline_query, baseline_expression)
    )
    return result.select(pl.col(internal).alias(name) for name, internal in OUTPUT.items()).sort(
        list(OUTPUT), nulls_last=True
    )


def find_parent_child_cousins(uniprot_df: pl.DataFrame) -> pl.DataFrame:
    """Build parent-child-cousins lookup from UniProt subcellular location terms.

    Args:
        uniprot_df: UniProt subcellular location terms.

    Returns:
        Frame with Name, SubcellID, Is_a, Child_SLterms, Contains_SLterms, concat and toSearch
        columns.
    """
    exploded_is_a = explode_outer(
        uniprot_df.select(
            pl.col('Name'),
            pl.col('Subcellular location ID').alias('SubcellID'),
            pl.col('Is part of'),
            pl.col('Is a').str.split(',').alias('Is_a_exploded'),
        ),
        'Is_a_exploded',
    )
    exploded_as_part_of = explode_outer(
        exploded_is_a.select(
            'Name',
            'SubcellID',
            'Is_a_exploded',
            pl.col('Is part of').str.split(',').alias('Is_part_exploded'),
        ),
        'Is_part_exploded',
    )
    first_df = exploded_as_part_of.select(
        'Name',
        'SubcellID',
        pl.col('Is_a_exploded').str.split(';').list.first().alias('Is_a_exploded_SL'),
        pl.col('Is_part_exploded').str.split(';').list.first().alias('Is_part_SL'),
    )

    parental_df = first_df.select('Name', 'SubcellID', pl.col('Is_a_exploded_SL').alias('Is_a')).unique(
        maintain_order=True
    )
    child_df = (
        first_df
        .select('SubcellID', 'Is_a_exploded_SL')
        .unique()
        .group_by('Is_a_exploded_SL')
        .agg(pl.col('SubcellID').drop_nulls().sort().alias('SubcellID_child'))
    )
    parent_child_df = parental_df.join(
        child_df, left_on='SubcellID', right_on='Is_a_exploded_SL', how='left', maintain_order='left'
    ).select('Name', 'SubcellID', 'Is_a', 'SubcellID_child')

    cousins_df = first_df.group_by('Is_part_SL').agg(
        pl.col('SubcellID').drop_nulls().sort().alias('SubcellID_are_part')
    )

    return (
        parent_child_df
        .join(cousins_df, left_on='SubcellID', right_on='Is_part_SL', how='left', maintain_order='left')
        .select(
            'Name',
            'SubcellID',
            'Is_a',
            pl.col('SubcellID_child').alias('Child_SLterms'),
            pl.col('SubcellID_are_part').alias('Contains_SLterms'),
            pl
            .concat_str(
                [pl.col('SubcellID'), join_list('SubcellID_child'), join_list('SubcellID_are_part')],
                separator=',',
                ignore_nulls=True,
            )
            .alias('concat'),
        )
        .with_columns(pl.col('concat').str.split(',').alias('toSearch'))
    )


def biotype_query(queryset: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Add biotype information."""
    pr_df = targets.select(
        pl.col('id').alias('targetid'),
        pl.col('biotype'),
        pl
        .when(pl.col('biotype') == 'protein_coding')
        .then(int32(1))
        .when(pl.col('biotype') == '')
        .then(pl.lit(None, dtype=pl.Int32))
        .otherwise(int32(0))
        .alias('Nr_biotype'),
    )
    return pr_df.join(queryset, on='targetid', how='left', maintain_order='left')


def search_terms(parent_child_cousins: pl.DataFrame, name: str) -> list[str]:
    """The toSearch terms of every row with the given name."""
    rows = explode(parent_child_cousins.filter(pl.col('Name') == name).select(pl.col('toSearch')), 'toSearch')
    return rows['toSearch'].drop_nulls().to_list()


def target_membrane_query(
    queryset: pl.DataFrame,
    targets: pl.DataFrame,
    parent_child_cousins: pl.DataFrame,
) -> pl.DataFrame:
    """Determine membrane and secreted status."""
    membrane_terms = search_terms(parent_child_cousins, 'Cell membrane')
    secreted_terms = search_terms(parent_child_cousins, 'Secreted')

    subcellular_locations = explode_outer(
        targets.select(pl.col('id').alias('targetid'), pl.col('subcellularLocations').alias('col')), 'col'
    ).unnest('col')

    location_info = subcellular_locations.select(
        'targetid',
        pl.when(pl.col('location').is_null()).then(pl.lit('noInfo')).otherwise(pl.lit('hasInfo')).alias('result'),
    ).unique(maintain_order=True)

    source = pl.col('source')
    in_membrane = pl.col('termSL').is_in(membrane_terms)
    in_secreted = pl.col('termSL').is_in(secreted_terms)
    membrane_grouped = (
        subcellular_locations
        .with_columns(
            pl
            .when((source == 'HPA_main') & in_membrane)
            .then(pl.lit('HPA_1'))
            .when((source == 'HPA_main') & ~in_membrane)
            .then(pl.lit('HPA_dif'))
            .when(source == 'HPA_extracellular_location')
            .then(pl.lit('HPA_secreted'))
            .when((source == 'HPA_additional') & in_membrane)
            .then(pl.lit('HPA_add_1'))
            .when((source == 'HPA_additional') & ~in_membrane)
            .then(pl.lit('HPA_dif'))
            .when((source == 'uniprot') & in_membrane)
            .then(pl.lit('uniprot_1'))
            .when((source == 'uniprot') & in_secreted)
            .then(pl.lit('uniprot_secreted'))
            .otherwise(pl.lit('Noinfo'))
            .alias('Count_mb')
        )
        .filter(pl.col('Count_mb').is_in(MEMBRANE_SOURCES))
        .select('targetid', 'Count_mb', 'source')
        .unique(['targetid', 'Count_mb'], keep='first', maintain_order=True)
        .group_by('targetid')
        .agg(
            pl.col('Count_mb').unique().sort().alias('mb'),
            pl.col('source').count().cast(pl.Int64).alias('counted'),
        )
    )

    def has(value: str) -> pl.Expr:
        return pl.col('mb').list.contains(value)

    protein_classification = membrane_grouped.with_columns(
        pl
        .when(has('HPA_1') | has('HPA_add_1'))
        .then(pl.lit('yes'))
        .when(has('HPA_dif'))
        .then(pl.lit('dif'))
        .otherwise(pl.lit('no'))
        .alias('HPA_membrane'),
        pl.when(has('HPA_secreted')).then(pl.lit('yes')).otherwise(pl.lit('no')).alias('HPA_secreted'),
        pl.when(has('uniprot_1')).then(pl.lit('yes')).otherwise(pl.lit('no')).alias('uniprot_membrane'),
        pl.when(has('uniprot_secreted')).then(pl.lit('yes')).otherwise(pl.lit('no')).alias('uniprot_secreted'),
    )

    hpa_mb, hpa_sec = pl.col('HPA_membrane'), pl.col('HPA_secreted')
    uni_mb, uni_sec = pl.col('uniprot_membrane'), pl.col('uniprot_secreted')
    membrane_with_loc = protein_classification.with_columns(
        pl
        .when((hpa_mb == 'yes') & (hpa_sec == 'no'))
        .then(pl.lit('inMembrane'))
        .when(((hpa_mb == 'no') | (hpa_mb == 'dif')) & (hpa_sec == 'yes'))
        .then(pl.lit('onlySecreted'))
        .when((hpa_mb == 'yes') & (hpa_sec == 'yes'))
        .then(pl.lit('secreted&inMembrane'))
        .when((hpa_mb == 'no') & (hpa_sec == 'no'))
        .then(
            pl
            .when((uni_mb == 'yes') & (uni_sec == 'no'))
            .then(pl.lit('inMembrane'))
            .when((uni_mb == 'no') & (uni_sec == 'yes'))
            .then(pl.lit('onlySecreted'))
            .when((uni_mb == 'yes') & (uni_sec == 'yes'))
            .then(pl.lit('secreted&inMembrane'))
        )
        .when(hpa_mb == 'dif')
        .then(pl.lit('noMembraneHPA'))
        .alias('loc')
    )

    joined = queryset.join(membrane_with_loc, on='targetid', how='left', maintain_order='left').join(
        location_info, on='targetid', how='left', maintain_order='left'
    )

    loc, result = pl.col('loc'), pl.col('result')
    return joined.with_columns(
        pl
        .when((loc == 'secreted&inMembrane') | (loc == 'inMembrane'))
        .then(int32(1))
        .when((loc != 'secreted&inMembrane') | (loc != 'inMembrane'))
        .then(int32(0))
        .when(loc.is_null() & (result == 'hasInfo'))
        .then(int32(0))
        .alias('Nr_mb'),
        pl
        .when((loc == 'secreted&inMembrane') | (loc == 'onlySecreted'))
        .then(int32(1))
        .when((loc == 'inMembrane') & (result == 'hasInfo'))
        .then(int32(0))
        .when(((loc != 'onlySecreted') | (loc != 'secreted&inMembrane')) & (result == 'hasInfo'))
        .then(int32(0))
        .when(result == 'noInfo')
        .then(pl.lit(None, dtype=pl.Int32))
        .otherwise(int32(0))
        .alias('Nr_secreted'),
    )


def ligand_pocket_query(queryset: pl.DataFrame, tractability: pl.DataFrame) -> pl.DataFrame:
    """Extract ligand, pocket, and small molecule binder status from tractability."""

    def pivot_sum(category: str) -> pl.Expr:
        presence = pl.col('presence').filter(pl.col('type') == category)
        return pl.when(presence.count() > 0).then(presence.cast(pl.Int64).sum()).alias(category)

    def flag(category: str) -> pl.Expr:
        return pl.when(pl.col(category) == 1).then(int32(1)).otherwise(int32(0))

    filtered_targets = (
        tractability
        .select(
            pl.col('targetId').alias('targetid'),
            pl.col('category').alias('type'),
            pl.col('value').cast(pl.Int32).alias('presence'),
        )
        .filter(pl.col('type').is_in(LIGAND_POCKET_CATEGORIES))
        .group_by('targetid')
        .agg(pivot_sum(c) for c in LIGAND_POCKET_CATEGORIES)
        .with_columns(
            flag('High-Quality Ligand').alias('Nr_Ligand'),
            flag('High-Quality Pocket').alias('Nr_Pocket'),
            flag('Small Molecule Binder').alias('Nr_sMBinder'),
        )
    )
    return queryset.join(filtered_targets, on='targetid', how='left', maintain_order='left')


def safety_query(queryset: pl.DataFrame, target_safety_event: pl.DataFrame) -> pl.DataFrame:
    """Extract safety event information."""
    agg_events = (
        target_safety_event
        .select(pl.col('targetId').alias('targetid'), pl.col('event'))
        .group_by('targetid')
        .agg(
            pl.col('event').count().cast(pl.Int64).alias('nEvents'),
            pl.col('event').drop_nulls().unique().sort().alias('events'),
        )
        .with_columns(pl.when(pl.col('nEvents') > 0).then(pl.lit('Yes')).alias('hasSafetyEvent'))
        .with_columns(pl.when(pl.col('hasSafetyEvent') == 'Yes').then(int32(-1)).alias('Nr_Event'))
    )
    return queryset.join(agg_events, on='targetid', how='left', maintain_order='left')


def constraint_query(queryset: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Compute genetic constraint score (LoF)."""
    constraints = (
        explode(targets.select(pl.col('id').alias('targetid'), pl.col('constraint').alias('col')), 'col')
        .unnest('col')
        .filter(pl.col('constraintType') == 'lof')
    )
    lo, hi = constraints.select(pl.col('upperRank').min().alias('lowerRank'), pl.col('upperRank').max()).row(0)
    if lo is None or hi is None:
        return queryset.with_columns(pl.lit(None, dtype=pl.Float64).alias('cal_score'))

    scaled = divide(pl.col('upperRank') - lo, float(hi - lo)) if hi != lo else pl.lit(None, dtype=pl.Float64)
    constraints = constraints.select('targetid', (2.0 * scaled - 1.0).alias('cal_score'), 'constraintType')
    return queryset.join(constraints, on='targetid', how='left', maintain_order='left')


def homology_type() -> pl.Expr:
    """The first `_`-separated token of the homology type."""
    return pl.col('homologyType').str.split('_').list.first()


def paralogs_query(queryset: pl.DataFrame, homology: pl.DataFrame) -> pl.DataFrame:
    """Compute paralog maximum identity percentage."""
    paralog = (
        homology
        .select(
            pl.col('targetId').alias('targetid'),
            homology_type()
            .str.replace_all('other', 'paralog_other', literal=True)
            .str.replace_all('within', 'paralog_intrasp', literal=True)
            .alias('homoType'),
            pl.col('queryPercentageIdentity'),
        )
        .filter(pl.col('homoType').str.contains('paralog', literal=True))
        .group_by('targetid')
        .agg(pl.col('queryPercentageIdentity').max().alias('max'))
        .with_columns(
            pl
            .when(pl.col('max') < 60)
            .then(pl.lit(0.0))
            .when(pl.col('max') >= 60)
            .then(-divide(pl.col('max') - 60.0, 40.0))
            .alias('Nr_paralogs')
        )
    )
    return queryset.join(paralog, on='targetid', how='left', maintain_order='left')


def orthologs_mouse_query(queryset: pl.DataFrame, homology: pl.DataFrame) -> pl.DataFrame:
    """Compute mouse ortholog maximum identity percentage."""
    orthologs = (
        homology
        .select(
            pl.col('targetId').alias('targetid'), 'homologyType', 'speciesName', 'queryPercentageIdentity'
        )
        .with_columns(homology_type().alias('homoType'))
        .filter(pl.col('homoType').str.contains('ortholog', literal=True) & (pl.col('speciesName') == 'Mouse'))
        .group_by('targetid')
        .agg(pl.col('queryPercentageIdentity').max().alias('max'))
        .with_columns(
            pl
            .when(pl.col('max') < 80)
            .then(pl.lit(0.0))
            .when(pl.col('max') >= 80)
            .then(divide(pl.col('max') - 80.0, 20.0))
            .alias('Nr_ortholog')
        )
    )
    return queryset.join(orthologs, on='targetid', how='left', maintain_order='left')


def driver_gene_query(queryset: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Identify cancer driver genes."""
    attributes = explode_outer(
        targets.select(
            pl.col('id').alias('targetid'), pl.col('hallmarks').struct.field('attributes').alias('col')
        ),
        'col',
    ).select('targetid', pl.col('col').struct.field('description'))
    onco_targets = (
        attributes
        .with_columns(
            pl.when(pl.col('description').is_in(ONCOTSG)).then(int32(1)).otherwise(int32(0)).alias('annotation')
        )
        .group_by('targetid')
        .agg(pl.col('annotation').max().alias('counts'))
        .with_columns(pl.when(pl.col('counts') != 0).then(int32(-1)).alias('Nr_CDG'))
    )
    return queryset.join(onco_targets, on='targetid', how='left', maintain_order='left')


def squared_rank(index: pl.Expr) -> pl.Expr:
    """Square of a 0-based index's 1-based rank, as a double."""
    rank = (index + 1).cast(pl.Float64)
    return rank * rank


def left_to_right_sum(terms: pl.Expr) -> pl.Expr:
    """Left-to-right sum of a double sequence; null when it is empty."""
    return terms.cum_sum().last()


def harmonic_sum(scores: pl.Expr) -> pl.Expr:
    """Sum of the non-null scores, sorted descending, each divided by its squared rank; 0.0 when there are none."""
    ranked = pl.element().drop_nulls().sort(descending=True)
    weighted = ranked / squared_rank(pl.int_range(ranked.len()))
    return scores.list.eval(left_to_right_sum(weighted)).list.first().fill_null(0.0)


def max_harmonic_sum(scores: pl.Expr) -> pl.Expr:
    """Harmonic sum of as many 1.0 scores as there are non-null scores; 0.0 when there are none."""
    weights = pl.lit(1.0) / squared_rank(pl.int_range(pl.element().drop_nulls().len()))
    return scores.list.eval(left_to_right_sum(weights)).list.first().fill_null(0.0)


def mouse_model_query(
    queryset: pl.DataFrame,
    mouse_df: pl.DataFrame,
    mouse_pheno_scores: pl.DataFrame,
) -> pl.DataFrame:
    """Compute mouse knockout score using harmonic sum."""
    pheno_scores = mouse_pheno_scores.select(
        pl.col('id').alias('idLabel'),
        pl.when(pl.col('score') == 0.0).then(pl.lit(0.0)).otherwise(pl.col('score')).cast(pl.Float64).alias('score'),
    )
    mouse_models = explode(
        mouse_df.select(
            pl.col('targetFromSourceId').alias('target_id_'), pl.col('modelPhenotypeClasses').alias('classes')
        ),
        'classes',
    ).with_columns(pl.col('classes').struct.field('id').alias('classId'))

    scores = (
        mouse_models
        .join(pheno_scores, left_on='classId', right_on='idLabel', how='left', maintain_order='left')
        .group_by('target_id_')
        .agg(pl.col('score').drop_nulls())
        .with_columns(
            harmonic_sum(pl.col('score')).alias('harmonicSum'),
            max_harmonic_sum(pl.col('score')).alias('maxHarmonicSum'),
        )
    )
    maximum = scores.select(pl.col('maxHarmonicSum').max()).item()
    scaled = divide(pl.col('harmonicSum'), maximum) if maximum is not None and maximum > 0 else pl.lit(0.0)
    mouse_models_pheno_scores = (
        scores
        .with_columns(pl.lit(maximum, dtype=pl.Float64).alias('maximum'))
        .with_columns(scaled.alias('scaledHarmonicSum'))
        .with_columns(
            pl
            .when(pl.col('scaledHarmonicSum') < LOW_THRESHOLD)
            .then(pl.lit(0.0))
            .otherwise(divide((pl.col('scaledHarmonicSum') - LOW_THRESHOLD) * -1.0, 1 - LOW_THRESHOLD))
            .alias('negScaledHarmonicSum')
        )
    )
    return queryset.join(
        mouse_models_pheno_scores, left_on='targetid', right_on='target_id_', how='left', maintain_order='left'
    )


def chemical_probes_query(queryset: pl.DataFrame, chemical_probes: pl.DataFrame) -> pl.DataFrame:
    """Check for high-quality chemical probes."""
    grouped = (
        chemical_probes
        .filter(pl.col('targetId').is_not_null())
        .select(
            pl.col('targetId').alias('targetid'),
            pl
            .when(pl.col('isHighQuality') == True)  # noqa: E712
            .then(int32(1))
            .when(pl.col('isHighQuality') == False)  # noqa: E712
            .then(int32(0))
            .alias('Nr_chprob'),
        )
        .group_by('targetid')
        .agg(pl.col('Nr_chprob').max())
    )
    return queryset.join(grouped, on='targetid', how='left', maintain_order='left')


def clin_trials_query(
    queryset: pl.DataFrame,
    mechanism_of_action: pl.DataFrame,
    molecule: pl.DataFrame,
) -> pl.DataFrame:
    """Compute maximum clinical stage score from MoA and molecule data.

    Every target of a mechanism is paired with every one of its drugs, the drug's maximum clinical
    stage is mapped to a 0-1 score, and the maximum is kept per target.
    """
    drug_targets = (
        mechanism_of_action
        .select(pl.col('targets').alias('target'), pl.col('chemblIds').alias('chemblId'))
        .filter((pl.col('target').list.len() > 0) & (pl.col('chemblId').list.len() > 0))
        .pipe(explode, 'target')
        .pipe(explode, 'chemblId')
        .unique(maintain_order=True)
    )
    molecule_phase = molecule.select(pl.col('id').alias('drugId'), pl.col('maximumClinicalStage'))
    drug_target_phase = drug_targets.join(
        molecule_phase, left_on='chemblId', right_on='drugId', how='left', maintain_order='left'
    )
    max_phase_per_target = (
        drug_target_phase
        .with_columns(
            pl
            .col('maximumClinicalStage')
            .replace_strict(STAGE_SCORES, default=None, return_dtype=pl.Float64)
            .alias('clinicalStageScore')
        )
        .unique(['target', 'maximumClinicalStage'], keep='first', maintain_order=True)
        .group_by('target')
        .agg(pl.col('clinicalStageScore').max().alias('inClinicalTrials'))
    )
    return queryset.join(max_phase_per_target, left_on='targetid', right_on='target', how='left', maintain_order='left')


def baseline_query(queryset: pl.DataFrame, baseline_expression: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Compute tissue and cell type specificity/distribution from baseline expression.

    For each target, picks the highest-specificity row per biosample type (tissue / celltype)
    across the GTEx and Tabula Sapiens datasources, then projects capped specificity and
    (1 - distribution) scores. Ties on specificity go to the lowest datasource, tissue, celltype and
    distribution, in that order.
    """
    tissue_id, celltype_id = pl.col('tissueBiosampleId'), pl.col('celltypeBiosampleId')
    best = (
        baseline_expression
        .lazy()
        .select(BASELINE_COLUMNS)
        .filter(
            pl.col('datasourceId').str.to_lowercase().is_in(BASELINE_DATASOURCES)
            & ~(tissue_id.is_not_null() & celltype_id.is_not_null())
            & pl.col('specificity_score').is_not_null()
        )
        .with_columns(
            pl
            .when(tissue_id.is_not_null())
            .then(pl.lit('tissue'))
            .when(celltype_id.is_not_null())
            .then(pl.lit('celltype'))
            .alias('biosample_type')
        )
        .filter(pl.col('biosample_type').is_not_null())
        .sort(
            ['specificity_score', 'datasourceId', 'tissueBiosampleId', 'celltypeBiosampleId', 'distribution_score'],
            descending=[True, False, False, False, False],
            nulls_last=True,
        )
        .group_by('targetId', 'biosample_type', maintain_order=True)
        .first()
        .collect()
    )

    def project(kind: str, spec_alias: str, dist_alias: str) -> pl.DataFrame:
        return best.filter(pl.col('biosample_type') == kind).select(
            pl.col('targetId').alias('targetid'),
            (2.0 * pl.col('specificity_score') - 1.0).alias(spec_alias),
            pl
            .when(pl.col('distribution_score') == 1)
            .then(pl.lit(-1.0))
            .otherwise(1.0 - pl.col('distribution_score'))
            .alias(dist_alias),
        )

    tissue = project('tissue', 'Nr_specificity', 'Nr_distribution')
    celltype = project('celltype', 'Nr_ct_specificity', 'Nr_ct_distribution')
    return queryset.join(tissue, on='targetid', how='left', maintain_order='left').join(
        celltype, on='targetid', how='left', maintain_order='left'
    )


def target_prioritisation(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Compute target prioritisation features and write the output dataset.

    Args:
        source: Mapping with keys ``targets``, ``tractability``, ``target_safety_event``,
            ``chemical_probes``, ``homology``, ``mouse_phenotypes``, ``molecule``,
            ``mechanism_of_action``, ``baseline_expression`` (parquet), ``uniprot_slterms`` (TSV)
            and ``mouse_pheno_scores`` (CSV).
        destination: Output path for the target prioritisation parquet.
        settings: Unused.
        config: Unused.
    """
    logger.info(f'loading data from {source}')
    frames = {
        'targets': scan_dataset(source['targets'])
        .select('id', 'biotype', 'subcellularLocations', 'constraint', 'hallmarks')
        .collect(),
        'tractability': scan_dataset(source['tractability']).select('targetId', 'category', 'value').collect(),
        'target_safety_event': scan_dataset(source['target_safety_event']).select('targetId', 'event').collect(),
        'chemical_probes': scan_dataset(source['chemical_probes']).select('targetId', 'isHighQuality').collect(),
        'homology': scan_dataset(source['homology'])
        .select('targetId', 'homologyType', 'speciesName', 'queryPercentageIdentity')
        .collect(),
        'mouse_phenotypes': scan_dataset(source['mouse_phenotypes'])
        .select('targetFromSourceId', 'modelPhenotypeClasses')
        .collect(),
        'molecule': scan_dataset(source['molecule']).select('id', 'maximumClinicalStage').collect(),
        'mechanism_of_action': scan_dataset(source['mechanism_of_action']).select('targets', 'chemblIds').collect(),
        'baseline_expression': scan_dataset(source['baseline_expression']),
        'uniprot_slterms': scan_dataset(source['uniprot_slterms'], format='tsv', infer_schema=False).collect(),
        'mouse_pheno_scores': scan_dataset(source['mouse_pheno_scores'], format='csv', infer_schema=False)
        .select('id', cast_double('score'))
        .collect(),
    }

    logger.info('computing target prioritisation features')
    result = compute_target_prioritisation(**frames)

    logger.info(f'writing {result.height} targets to {destination}')
    write_dataset(result, destination, schema=TargetPrioritisation)
