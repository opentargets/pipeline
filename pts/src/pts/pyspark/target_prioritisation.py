"""Target Prioritisation step.

Computes various features for target prioritisation including membrane location,
safety events, tractability, genetic constraint, homology, and clinical trial data.
"""

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame, Window
from pyspark.sql.types import DoubleType

from pts.pyspark.common.session import Session


def target_prioritisation(
    source: dict[str, str],
    destination: str,
    _settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Compute target prioritisation features.

    Args:
        source: Dictionary with paths to:
            - targets: Target parquet
            - mouse_phenotypes: Mouse phenotype parquet
            - molecule: Drug molecule parquet
            - mechanism_of_action: Mechanism of action parquet
            - hpa_data: HPA normal tissue TSV
            - uniprot_slterms: UniProt subcellular location terms TSV
            - mouse_pheno_scores: Mouse phenotype scores parquet
        destination: Path to write the output parquet file.
        _settings: Custom settings (not used).
        properties: Spark configuration options.
    """
    spark = Session(app_name='target_prioritisation', properties=properties)

    logger.info(f'Loading data from {source}')
    targets_df = spark.load_data(source['targets'])
    mouse_df = spark.load_data(source['mouse_phenotypes'])
    molecule_df = spark.load_data(source['molecule'])
    moa_df = spark.load_data(source['mechanism_of_action'])
    hpa_df = spark.load_data(source['hpa_data'], format='json')
    uniprot_df = spark.load_data(source['uniprot_slterms'], format='csv', sep='\t', header=True)
    mouse_pheno_scores_df = spark.load_data(source['mouse_pheno_scores'], format='csv', header=True)

    logger.info('Computing target prioritisation features')
    output_df = compute_target_prioritisation(
        targets_df,
        mouse_df,
        molecule_df,
        moa_df,
        hpa_df,
        uniprot_df,
        mouse_pheno_scores_df,
    )

    logger.info(f'Writing target prioritisation to {destination}')
    # Coalesce to single partition as per original implementation
    output_df.coalesce(1).write.parquet(destination, mode='overwrite')


def compute_target_prioritisation(
    targets: DataFrame,
    mouse_phenotypes: DataFrame,
    molecule: DataFrame,
    mechanism_of_action: DataFrame,
    hpa_data: DataFrame,
    uniprot_slterms: DataFrame,
    mouse_pheno_scores: DataFrame,
) -> DataFrame:
    """Compute all target prioritisation features.

    Args:
        targets: Target data.
        mouse_phenotypes: Mouse phenotype data.
        molecule: Drug molecule data.
        mechanism_of_action: Mechanism of action data.
        hpa_data: HPA normal tissue data.
        uniprot_slterms: UniProt subcellular location terms.
        mouse_pheno_scores: Mouse phenotype scores.

    Returns:
        DataFrame with target prioritisation features.
    """
    # Build parent-child-cousins lookup for membrane location
    parent_child_cousins = _find_parent_child_cousins(uniprot_slterms)

    # Start with target IDs
    queryset = targets.select(f.col('id').alias('targetid'))

    # Apply all transformations
    result = (
        queryset
        .transform(lambda df: _biotype_query(df, targets))
        .transform(lambda df: _target_membrane_query(df, targets, parent_child_cousins))
        .transform(lambda df: _ligand_pocket_query(df, targets))
        .transform(lambda df: _safety_query(df, targets))
        .transform(lambda df: _constraint_query(df, targets))
        .transform(lambda df: _paralogs_query(df, targets))
        .transform(lambda df: _orthologs_mouse_query(df, targets))
        .transform(lambda df: _driver_gene_query(df, targets))
        .transform(lambda df: _tep_query(df, targets))
        .transform(lambda df: _mouse_model_query(df, mouse_phenotypes, mouse_pheno_scores))
        .transform(lambda df: _chemical_probes_query(df, targets))
        .transform(lambda df: _clin_trials_query(df, mechanism_of_action, molecule))
        .transform(lambda df: _tissue_specific_query(df, hpa_data))
    )

    # Select final columns with renamed output
    return result.select(
        f.col('targetid').alias('targetId'),
        f.col('Nr_mb').alias('isInMembrane'),
        f.col('Nr_secreted').alias('isSecreted'),
        f.col('Nr_Event').alias('hasSafetyEvent'),
        f.col('Nr_Pocket').alias('hasPocket'),
        f.col('Nr_Ligand').alias('hasLigand'),
        f.col('Nr_sMBinder').alias('hasSmallMoleculeBinder'),
        f.col('cal_score').alias('geneticConstraint'),
        f.col('Nr_paralogs').alias('paralogMaxIdentityPercentage'),
        f.col('Nr_ortholog').alias('mouseOrthologMaxIdentityPercentage'),
        f.col('Nr_CDG').alias('isCancerDriverGene'),
        f.col('Nr_TEP').alias('hasTEP'),
        f.col('negScaledHarmonicSum').alias('mouseKOScore'),
        f.col('Nr_chprob').alias('hasHighQualityChemicalProbes'),
        f.col('inClinicalTrials').alias('maxClinicalStage'),
        f.col('Nr_specificity').alias('tissueSpecificity'),
        f.col('Nr_distribution').alias('tissueDistribution'),
    )


def _find_parent_child_cousins(uniprot_df: DataFrame) -> DataFrame:
    """Build parent-child-cousins lookup from UniProt subcellular location terms.

    Args:
        uniprot_df: UniProt subcellular location terms.

    Returns:
        DataFrame with Name, SubcellID, and toSearch columns.
    """
    # Explode Is a and Is part of columns
    exploded_is_a = uniprot_df.select(
        f.col('Name'),
        f.col('Subcellular location ID').alias('SubcellID'),
        f.col('Is part of'),
        f.explode_outer(f.split(f.col('Is a'), ',')).alias('Is_a_exploded'),
    )

    exploded_as_part_of = exploded_is_a.select(
        f.col('Name'),
        f.col('SubcellID'),
        f.col('Is_a_exploded'),
        f.explode_outer(f.split(f.col('Is part of'), ',')).alias('Is_part_exploded'),
    )

    first_df = exploded_as_part_of.select(
        f.col('Name'),
        f.col('SubcellID'),
        f.split(f.col('Is_a_exploded'), ';').getItem(0).alias('Is_a_exploded_SL'),
        f.split(f.col('Is_part_exploded'), ';').getItem(0).alias('Is_part_SL'),
    )

    parental_df = (
        first_df.select(
            f.col('Name'),
            f.col('SubcellID'),
            f.col('Is_a_exploded_SL').alias('Is_a'),
        )
        .distinct()
    )

    child_df = (
        first_df.select('SubcellID', 'Is_a_exploded_SL')
        .distinct()
        .groupBy('Is_a_exploded_SL')
        .agg(f.collect_list(f.col('SubcellID')).alias('SubcellID_child'))
    )

    parent_child_df = (
        parental_df.join(
            child_df,
            parental_df['SubcellID'] == child_df['Is_a_exploded_SL'],
            'left',
        )
        .select('Name', 'SubcellID', 'Is_a', 'SubcellID_child')
    )

    cousins_df = (
        first_df.groupBy(f.col('Is_part_SL'))
        .agg(f.collect_list('SubcellID').alias('SubcellID_are_part'))
    )

    parent_child_cousins = (
        parent_child_df.join(
            cousins_df,
            cousins_df['Is_part_SL'] == parent_child_df['SubcellID'],
            'left',
        )
        .select(
            f.col('Name'),
            f.col('SubcellID'),
            f.col('Is_a'),
            f.col('SubcellID_child').alias('Child_SLterms'),
            f.col('SubcellID_are_part').alias('Contains_SLterms'),
            f.concat_ws(
                ',',
                f.col('SubcellID'),
                f.col('SubcellID_child'),
                f.col('SubcellID_are_part'),
            ).alias('concat'),
        )
    )

    return parent_child_cousins.select(
        f.col('*'),
        f.split(f.col('concat'), ',').alias('toSearch'),
    )


def _biotype_query(queryset: DataFrame, targets: DataFrame) -> DataFrame:
    """Add biotype information."""
    pr_df = targets.select(
        f.col('id').alias('targetid'),
        f.col('biotype'),
        f.when(f.col('biotype') == 'protein_coding', 1)
        .when(f.col('biotype') == '', None)  # noqa: PLC1901
        .otherwise(0)
        .alias('Nr_biotype'),
    )
    return pr_df.join(queryset, on='targetid', how='left')


def _target_membrane_query(
    queryset: DataFrame,
    targets: DataFrame,
    parent_child_cousins: DataFrame,
) -> DataFrame:
    """Determine membrane and secreted status."""
    # Get membrane and secreted terms
    membrane_terms = (
        parent_child_cousins.filter(f.col('Name') == 'Cell membrane')
        .select(f.explode(f.col('toSearch')).alias('termSL'))
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    secreted_terms = (
        parent_child_cousins.filter(f.col('Name') == 'Secreted')
        .select(f.explode(f.col('toSearch')).alias('termSL'))
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    # Explode subcellular locations
    subcellular_locations = targets.select(
        f.col('id').alias('targetid'),
        f.explode_outer(f.col('subcellularLocations')).alias('col'),
    )

    location_info = (
        subcellular_locations.select(
            f.col('targetid'),
            f.when(f.col('col.location').isNull(), f.lit('noInfo'))
            .otherwise('hasInfo')
            .alias('result'),
        )
        .dropDuplicates()
    )

    source_list = [
        'HPA_1', 'HPA_secreted', 'HPA_add_1', 'uniprot_1', 'uniprot_secreted', 'HPA_dif'
    ]

    membrane_grouped = (
        subcellular_locations.select(f.col('targetid'), f.col('col.*'))
        .withColumn(
            'Count_mb',
            f.when(
                (f.col('source') == 'HPA_main') & f.col('termSL').isin(membrane_terms),
                f.lit('HPA_1'),
            )
            .when(
                (f.col('source') == 'HPA_main') & ~f.col('termSL').isin(membrane_terms),
                f.lit('HPA_dif'),
            )
            .when(f.col('source') == 'HPA_extracellular_location', f.lit('HPA_secreted'))
            .when(
                (f.col('source') == 'HPA_additional') & f.col('termSL').isin(membrane_terms),
                f.lit('HPA_add_1'),
            )
            .when(
                (f.col('source') == 'HPA_additional') & ~f.col('termSL').isin(membrane_terms),
                f.lit('HPA_dif'),
            )
            .when(
                (f.col('source') == 'uniprot') & f.col('termSL').isin(membrane_terms),
                f.lit('uniprot_1'),
            )
            .when(
                (f.col('source') == 'uniprot') & f.col('termSL').isin(secreted_terms),
                f.lit('uniprot_secreted'),
            )
            .otherwise(f.lit('Noinfo')),
        )
        .filter(f.col('Count_mb').isin(source_list))
        .select(f.col('targetid'), f.col('Count_mb'), f.col('source'))
        .dropDuplicates(['targetid', 'Count_mb'])
        .groupBy('targetid')
        .agg(
            f.collect_set('Count_mb').alias('mb'),
            f.count(f.col('source')).alias('counted'),
        )
    )

    protein_classification = membrane_grouped.select(
        f.col('*'),
        f.when(
            f.array_contains(f.col('mb'), 'HPA_1')
            | f.array_contains(f.col('mb'), 'HPA_add_1'),
            f.lit('yes'),
        )
        .when(f.array_contains(f.col('mb'), 'HPA_dif'), f.lit('dif'))
        .otherwise(f.lit('no'))
        .alias('HPA_membrane'),
        f.when(f.array_contains(f.col('mb'), 'HPA_secreted'), f.lit('yes'))
        .otherwise(f.lit('no'))
        .alias('HPA_secreted'),
        f.when(f.array_contains(f.col('mb'), 'uniprot_1'), f.lit('yes'))
        .otherwise(f.lit('no'))
        .alias('uniprot_membrane'),
        f.when(f.array_contains(f.col('mb'), 'uniprot_secreted'), f.lit('yes'))
        .otherwise(f.lit('no'))
        .alias('uniprot_secreted'),
    )

    membrane_with_loc = protein_classification.withColumn(
        'loc',
        f.when(
            (f.col('HPA_membrane') == 'yes') & (f.col('HPA_secreted') == 'no'),
            f.lit('inMembrane'),
        )
        .when(
            ((f.col('HPA_membrane') == 'no') | (f.col('HPA_membrane') == 'dif'))
            & (f.col('HPA_secreted') == 'yes'),
            f.lit('onlySecreted'),
        )
        .when(
            (f.col('HPA_membrane') == 'yes') & (f.col('HPA_secreted') == 'yes'),
            f.lit('secreted&inMembrane'),
        )
        .when(
            (f.col('HPA_membrane') == 'no') & (f.col('HPA_secreted') == 'no'),
            f.when(
                (f.col('uniprot_membrane') == 'yes') & (f.col('uniprot_secreted') == 'no'),
                f.lit('inMembrane'),
            )
            .when(
                (f.col('uniprot_membrane') == 'no') & (f.col('uniprot_secreted') == 'yes'),
                f.lit('onlySecreted'),
            )
            .when(
                (f.col('uniprot_membrane') == 'yes') & (f.col('uniprot_secreted') == 'yes'),
                f.lit('secreted&inMembrane'),
            ),
        )
        .when(f.col('HPA_membrane') == 'dif', f.lit('noMembraneHPA')),
    )

    joined = (
        membrane_with_loc.join(queryset, on='targetid', how='right')
        .join(location_info, on='targetid', how='left')
    )

    return joined.select(
        f.col('*'),
        f.when(
            (f.col('loc') == 'secreted&inMembrane') | (f.col('loc') == 'inMembrane'),
            f.lit(1),
        )
        .when(
            (f.col('loc') != 'secreted&inMembrane') | (f.col('loc') != 'inMembrane'),
            f.lit(0),
        )
        .when(f.col('loc').isNull() & (f.col('result') == 'hasInfo'), f.lit(0))
        .alias('Nr_mb'),
        f.when(
            (f.col('loc') == 'secreted&inMembrane') | (f.col('loc') == 'onlySecreted'),
            f.lit(1),
        )
        .when(
            (f.col('loc') == 'inMembrane') & (f.col('result') == 'hasInfo'),
            f.lit(0),
        )
        .when(
            ((f.col('loc') != 'onlySecreted') | (f.col('loc') != 'secreted&inMembrane'))
            & (f.col('result') == 'hasInfo'),
            f.lit(0),
        )
        .when(f.col('result') == 'noInfo', f.lit(None))
        .otherwise(f.lit(0))
        .alias('Nr_secreted'),
    )


def _ligand_pocket_query(queryset: DataFrame, targets: DataFrame) -> DataFrame:
    """Extract ligand, pocket, and small molecule binder status from tractability."""
    filtered_targets = (
        targets.select(
            f.col('id').alias('targetid'),
            f.explode_outer(f.col('tractability')).alias('new_struct'),
        )
        .filter(
            (f.col('new_struct.id') == 'High-Quality Ligand')
            | (f.col('new_struct.id') == 'High-Quality Pocket')
            | (f.col('new_struct.id') == 'Small Molecule Binder')
        )
        .withColumn('type', f.col('new_struct').getItem('id'))
        .withColumn('presence', f.col('new_struct').getItem('value').cast('int'))
        .groupBy('targetid')
        .pivot('type')
        .agg(f.sum('presence'))
        .select(
            f.col('*'),
            f.when(f.col('High-Quality Ligand') == 1, f.lit(1))
            .otherwise(f.lit(0))
            .alias('Nr_Ligand'),
            f.when(f.col('High-Quality Pocket') == 1, f.lit(1))
            .otherwise(f.lit(0))
            .alias('Nr_Pocket'),
            f.when(f.col('Small Molecule Binder') == 1, f.lit(1))
            .otherwise(f.lit(0))
            .alias('Nr_sMBinder'),
        )
    )

    return queryset.join(filtered_targets, on='targetid', how='left')


def _safety_query(queryset: DataFrame, targets: DataFrame) -> DataFrame:
    """Extract safety event information."""
    agg_events = (
        targets.withColumn(
            'info',
            f.when(f.size(f.col('safetyLiabilities')) > 0, f.lit('conInfo'))
            .otherwise(f.lit('noReported')),
        )
        .select(
            f.col('id').alias('targetid'),
            f.explode_outer(f.col('safetyLiabilities')).alias('col'),
            f.col('info'),
        )
        .groupBy('targetid', 'info')
        .agg(
            f.count(f.col('col.event')).alias('nEvents'),
            f.array_distinct(f.collect_list('col.event')).alias('events'),
        )
        .withColumn(
            'hasSafetyEvent',
            f.when((f.col('nEvents') > 0) & (f.col('info') == 'conInfo'), f.lit('Yes'))
            .otherwise(f.lit(None)),
        )
        .withColumn(
            'Nr_Event',
            f.when(f.col('hasSafetyEvent') == 'Yes', f.lit(-1)).otherwise(f.lit(None)),
        )
    )

    return queryset.join(agg_events, on='targetid', how='left')


def _constraint_query(queryset: DataFrame, targets: DataFrame) -> DataFrame:
    """Compute genetic constraint score (LoF)."""
    # Get min and max upper rank for LoF constraint
    constraint_stats = (
        targets.select(f.col('id').alias('constr_id'), f.explode(f.col('constraint')).alias('col'))
        .select(f.col('col.*'))
        .filter(f.col('constraintType') == 'lof')
        .groupBy('constraintType')
        .agg(
            f.min('upperRank').cast('int').alias('lowerRank'),
            f.max('upperRank').cast('int').alias('upperRank'),
        )
        .first()
    )

    if constraint_stats is None:
        return queryset.withColumn('cal_score', f.lit(None))

    min_upper_rank = constraint_stats['lowerRank']
    max_upper_rank = constraint_stats['upperRank']

    constraints = (
        targets.select(f.col('id').alias('targetid'), f.explode(f.col('constraint')).alias('col'))
        .select(f.col('targetid'), f.col('col.*'))
        .filter(f.col('constraintType') == 'lof')
        .select(
            f.col('targetid'),
            (
                f.lit(2) * ((f.col('upperRank') - min_upper_rank) / (max_upper_rank - min_upper_rank))
                - f.lit(1)
            ).alias('cal_score'),
            f.col('constraintType'),
        )
    )

    return queryset.join(constraints, on='targetid', how='left')


def _paralogs_query(queryset: DataFrame, targets: DataFrame) -> DataFrame:
    """Compute paralog maximum identity percentage."""
    exploded = (
        targets.select(
            f.col('id').alias('targetid'),
            f.when(f.size(f.col('homologues')) > f.lit(0), f.lit('hasInfo'))
            .otherwise('noInfo/null')
            .alias('hasInfo'),
            f.explode(f.col('homologues')).alias('col'),
        )
        .withColumn(
            'homoType',
            f.regexp_replace(
                f.regexp_replace(
                    f.split(f.col('col.homologyType'), '_').getItem(0),
                    'other',
                    'paralog_other',
                ),
                'within',
                'paralog_intrasp',
            ),
        )
        .withColumn('howmany', f.split(f.col('col.homologyType'), '_').getItem(1))
        .withColumn('queryPercentageIdentity', f.col('col.queryPercentageIdentity'))
    )

    paralog = (
        exploded.filter(f.col('homoType').contains('paralog'))
        .groupBy('targetid')
        .agg(f.max('queryPercentageIdentity').alias('max'))
        .withColumn(
            'Nr_paralogs',
            f.when(f.col('max') < f.lit(60), f.lit(0))
            .when(f.col('max') >= f.lit(60), -((f.col('max') - f.lit(60)) / f.lit(40))),
        )
    )

    return queryset.join(paralog, on='targetid', how='left')


def _orthologs_mouse_query(queryset: DataFrame, targets: DataFrame) -> DataFrame:
    """Compute mouse ortholog maximum identity percentage."""
    orthologs = (
        targets.select(f.col('id').alias('targetid'), f.explode(f.col('homologues')).alias('col'))
        .select(f.col('targetid'), f.col('col.*'))
        .withColumn('homoType', f.split(f.col('homologyType'), '_').getItem(0))
        .withColumn('howmany', f.split(f.col('homologyType'), '_').getItem(1))
        .filter(
            f.col('homoType').contains('ortholog') & (f.col('speciesName') == 'Mouse')
        )
        .select(
            'targetid',
            'homoType',
            'howmany',
            'targetGeneId',
            'targetPercentageIdentity',
            'queryPercentageIdentity',
        )
        .groupBy('targetid')
        .agg(f.max('queryPercentageIdentity').alias('max'))
        .withColumn(
            'Nr_ortholog',
            f.when(f.col('max') < 80, f.lit(0))
            .when(f.col('max') >= 80, (f.col('max') - 80) / 20),
        )
    )

    return queryset.join(orthologs, on='targetid', how='left')


def _driver_gene_query(queryset: DataFrame, targets: DataFrame) -> DataFrame:
    """Identify cancer driver genes."""
    oncotsg_list = [
        'TSG', 'oncogene', 'Oncogene', 'oncogene,TSG', 'TSG,oncogene',
        'fusion,oncogene', 'oncogene,fusion',
    ]

    onco_targets = (
        targets.select(
            f.col('id').alias('targetid'),
            f.explode_outer(f.col('hallmarks.attributes')).alias('col'),
        )
        .select(
            f.col('targetid'),
            f.col('col.description'),
            f.when(f.col('col.description').isin(oncotsg_list), f.lit(1))
            .otherwise(f.lit(0))
            .alias('annotation'),
        )
        .groupBy('targetid')
        .agg(f.max(f.col('annotation')).alias('counts'))
        .withColumn(
            'Nr_CDG',
            f.when(f.col('counts') != 0, f.lit(-1)).otherwise(f.lit(None)),
        )
    )

    return queryset.join(onco_targets, on='targetid', how='left')


def _tep_query(queryset: DataFrame, targets: DataFrame) -> DataFrame:
    """Check for Target Enabling Package."""
    tep = targets.select(
        f.col('id').alias('targetid'),
        f.col('tep.*'),
        f.when(
            f.col('tep.description').isNotNull() | (f.col('tep.description') != ''),  # noqa: PLC1901
            f.lit(1),
        )
        .otherwise(f.lit(None))
        .alias('Nr_TEP'),
    )

    return queryset.join(tep, on='targetid', how='left')


def _mouse_model_query(
    queryset: DataFrame,
    mouse_df: DataFrame,
    mouse_pheno_scores: DataFrame,
) -> DataFrame:
    """Compute mouse knockout score using harmonic sum."""
    low_threshold = 0.6

    # Register UDFs
    @f.udf(returnType=DoubleType())
    def harmonic_sum(scores):
        if scores is None or len(scores) == 0:
            return 0.0
        sorted_scores = sorted([s for s in scores if s is not None], reverse=True)
        if len(sorted_scores) == 0:
            return 0.0
        denominators = [(i + 1) ** 2 for i in range(len(sorted_scores))]
        return sum(s / d for s, d in zip(sorted_scores, denominators, strict=False))

    @f.udf(returnType=DoubleType())
    def max_harmonic_sum(scores):
        if scores is None or len(scores) == 0:
            return 0.0
        n = len([s for s in scores if s is not None])
        if n == 0:
            return 0.0
        denominators = [(i + 1) ** 2 for i in range(n)]
        return sum(1.0 / d for d in denominators)

    pheno_scores = mouse_pheno_scores.select(
        f.col('id').alias('idLabel'),
        f.when(f.col('score') == 0.0, f.lit(0)).otherwise(f.col('score')).alias('score'),
    )

    mouse_models = mouse_df.select(
        f.col('targetFromSourceId').alias('target_id_'),
        f.explode(f.col('modelPhenotypeClasses')).alias('classes'),
    )

    mouse_models_pheno_scores = (
        mouse_models.join(
            pheno_scores,
            mouse_models['classes.id'] == pheno_scores['idLabel'],
            'left',
        )
        .withColumn('score', f.col('score').cast('double'))
        .groupBy('target_id_')
        .agg(f.collect_list(f.col('score')).alias('score'))
        .withColumn('harmonicSum', harmonic_sum(f.col('score')).cast('double'))
        .withColumn('maxHarmonicSum', max_harmonic_sum(f.col('score')).cast('double'))
        .withColumn('maximum', f.max(f.col('maxHarmonicSum')).over(Window.orderBy()).cast('double'))
        .withColumn(
            'scaledHarmonicSum',
            f.when(f.col('maximum') > 0, f.col('harmonicSum') / f.col('maximum'))
            .otherwise(f.lit(0)),
        )
        .withColumn(
            'negScaledHarmonicSum',
            f.when(f.col('scaledHarmonicSum') < low_threshold, f.lit(0))
            .otherwise(
                (f.col('scaledHarmonicSum') - low_threshold) * -1 / (1 - low_threshold)
            ),
        )
    )

    return queryset.join(
        mouse_models_pheno_scores,
        f.col('target_id_') == queryset['targetid'],
        'left',
    )


def _chemical_probes_query(queryset: DataFrame, targets: DataFrame) -> DataFrame:
    """Check for high-quality chemical probes."""
    probes = targets.select(
        f.col('id').alias('targetid'),
        f.explode_outer(f.col('chemicalProbes')).alias('col'),
        f.when(f.size(f.col('chemicalProbes')) > f.lit(0), f.lit('hasInfo'))
        .otherwise(f.lit('noInfo'))
        .alias('info'),
    )

    grouped = (
        probes.withColumn(
            'Nr_chprob',
            f.when(
                (f.col('info') == 'hasInfo') & (f.col('col.isHighQuality') == True),  # noqa: E712
                f.lit(1),
            )
            .when(
                (f.col('info') == 'hasInfo') & (f.col('col.isHighQuality') == False),  # noqa: E712
                f.lit(0),
            )
            .otherwise(f.lit(None)),
        )
        .groupBy('targetid')
        .agg(f.max(f.col('Nr_chprob')).alias('Nr_chprob'))
    )

    return queryset.join(grouped, on='targetid', how='left')


def _clin_trials_query(
    queryset: DataFrame,
    mechanism_of_action: DataFrame,
    molecule: DataFrame,
) -> DataFrame:
    """Compute maximum clinical stage score from MoA and molecule data.

    Uses MoA to find drug-target relationships, joins with molecule to get
    clinical stage names, then maps each to a 0-1 score and takes the max.
    """
    # Get drug-target relationships from MoA
    drug_targets = (
        mechanism_of_action.select(
            f.explode(f.col('targets')).alias('target'),
            f.explode(f.col('chemblIds')).alias('chemblId'),
        )
        .distinct()
    )

    # Get clinical trial phase from molecule data
    molecule_phase = molecule.select(
        f.col('id').alias('drugId'),
        f.col('maximumClinicalStage'),
    )

    # Join drug-target with molecule to get clinical trial phase per target
    drug_target_phase = drug_targets.join(
        molecule_phase,
        drug_targets['chemblId'] == molecule_phase['drugId'],
        'left',
    )

    # Map clinical stage names to 0-1 scores and take max per target
    stage_score_map = f.create_map(
        f.lit('approved'), f.lit(1.0),
        f.lit('pre-approval'), f.lit(0.8),
        f.lit('phase III'), f.lit(0.7),
        f.lit('phase II/III'), f.lit(0.5),
        f.lit('phase II'), f.lit(0.2),
        f.lit('phase I/II'), f.lit(0.15),
        f.lit('phase I'), f.lit(0.1),
        f.lit('early phase I'), f.lit(0.05),
        f.lit('IND'), f.lit(0.05),
        f.lit('preclinical'), f.lit(0.01),
        f.lit('unknown'), f.lit(0.01),
    )

    max_phase_per_target = (
        drug_target_phase
        .withColumn('clinicalStageScore', stage_score_map[f.col('maximumClinicalStage')])
        .dropDuplicates(['target', 'maximumClinicalStage'])
        .groupBy('target')
        .agg(f.max('clinicalStageScore').alias('inClinicalTrials'))
    )

    return queryset.join(
        max_phase_per_target,
        queryset['targetid'] == max_phase_per_target['target'],
        'left',
    )


def _tissue_specific_query(queryset: DataFrame, hpa_data: DataFrame) -> DataFrame:
    """Compute tissue specificity and distribution from HPA data."""
    hpa = (
        hpa_data.select(
            'Ensembl',
            f.col('RNA tissue distribution').alias('Tissue_distribution_RNA'),
            f.col('RNA tissue specificity').alias('Tissue_specificity_RNA'),
        )
        .withColumn(
            'Nr_specificity',
            f.when(f.col('Tissue_specificity_RNA') == 'Tissue enriched', f.lit(1))
            .when(f.col('Tissue_specificity_RNA') == 'Group enriched', f.lit(0.75))
            .when(f.col('Tissue_specificity_RNA') == 'Tissue enhanced', f.lit(0.5))
            .when(f.col('Tissue_specificity_RNA') == 'Low tissue specificity', f.lit(-1))
            .when(f.col('Tissue_specificity_RNA') == 'Not detected', f.lit(None)),
        )
        .withColumn(
            'Nr_distribution',
            f.when(f.col('Tissue_distribution_RNA') == 'Detected in single', f.lit(1))
            .when(f.col('Tissue_distribution_RNA') == 'Detected in some', f.lit(0.5))
            .when(f.col('Tissue_distribution_RNA') == 'Detected in many', f.lit(0))
            .when(f.col('Tissue_distribution_RNA') == 'Detected in all', f.lit(-1))
            .when(f.col('Tissue_distribution_RNA') == 'Not detected', f.lit(None)),
        )
    )

    return queryset.join(hpa, queryset['targetid'] == hpa['Ensembl'], 'left')
