"""Interaction dataset generation.

Computes protein-protein and RNA interactions from IntAct and STRING, producing
aggregated interaction records, per-evidence records and the unmatched human
interactors.

``interaction`` has one row per interaction pair, keyed by ``interactionId``.
``interaction_evidence`` has one row per evidence entry and carries
``interactionId`` in place of the pair's identity columns.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import polars_hash as plh
from loguru import logger
from otter.config.model import Config

from pts.schemas.outputs.interaction import Interaction
from pts.schemas.outputs.interaction_evidence import InteractionEvidence
from pts.transformers.utils.dataset import scan_dataset, write_dataset

SWAP = {
    'targetA': 'targetB',
    'intA': 'intB',
    'intA_source': 'intB_source',
    'speciesA': 'speciesB',
    'intABiologicalRole': 'intBBiologicalRole',
}
SWAP |= {v: k for k, v in SWAP.items()}

BIDIRECTIONAL_SOURCES = ['reactome', 'intact', 'signor']

STRING_EVIDENCE_CHANNELS = [
    ('textmining', 'MI:0110'),
    ('database', ''),
    ('experimental', 'MI:0591'),
    ('fusion', 'MI:0036'),
    ('neighborhood', 'MI:0057'),
    ('cooccurence', 'MI:2231'),
    ('coexpression', 'MI:2231'),
    ('homology', 'MI:2163'),
]

INTERACTION_KEY = [
    'sourceDatabase',
    'targetA',
    'intA',
    'intABiologicalRole',
    'targetB',
    'intB',
    'intBBiologicalRole',
    'speciesA',
    'speciesB',
]

DIMENSION_COLUMNS = [
    'sourceDatabase',
    'targetA',
    'intA',
    'intA_source',
    'intABiologicalRole',
    'targetB',
    'intB',
    'intB_source',
    'intBBiologicalRole',
    'speciesA',
    'speciesB',
]

ID_SEPARATOR = '\x1f'
ID_NULL = '\x00'

# struct fields in alphabetical order, as spark's json inference produced them
_ORGANISM = pl.Struct({'mnemonic': pl.String, 'scientific_name': pl.String, 'taxon_id': pl.Int64})
_INTERACTOR = pl.Struct({
    'biological_role': pl.String,
    'id': pl.String,
    'id_source': pl.String,
    'organism': _ORGANISM,
})
_MI = pl.Struct({'mi_identifier': pl.String, 'short_name': pl.String})
_EVIDENCE = pl.Struct({
    'expansion_method_mi_identifier': pl.String,
    'expansion_method_short_name': pl.String,
    'host_organism_scientific_name': pl.String,
    'host_organism_tax_id': pl.Int64,
    'host_organism_tissue': pl.Struct({'full_name': pl.String, 'short_name': pl.String, 'xrefs': pl.List(pl.String)}),
    'interaction_detection_method_mi_identifier': pl.String,
    'interaction_detection_method_short_name': pl.String,
    'interaction_identifier': pl.String,
    'interaction_type_mi_identifier': pl.String,
    'interaction_type_short_name': pl.String,
    'participant_detection_method_A': pl.List(_MI),
    'participant_detection_method_B': pl.List(_MI),
    'pubmed_id': pl.String,
})
INTACT_SCHEMA = {
    'interaction': pl.Struct({
        'causal_interaction': pl.String,
        'evidence': pl.List(_EVIDENCE),
        'interaction_score': pl.Float64,
    }),
    'interactorA': _INTERACTOR,
    'interactorB': _INTERACTOR,
    'source_info': pl.Struct({'database_version': pl.String, 'source_database': pl.String}),
}


def _headerless(n: int) -> dict[str, pl.DataType]:
    return {f'_c{i}': pl.String() for i in range(n)}


def divide(expr: pl.Expr, divisor: float) -> pl.Expr:
    """Divide by a scalar with IEEE division.

    Polars rewrites ``x / c`` as ``x * (1 / c)``, which is not exact: 288 / 1000 gives 0.28800000000000003.
    """
    return expr.cast(pl.Float64).map_batches(
        lambda s: pl.Series(s.to_numpy() / float(divisor), nan_to_null=True),
        return_dtype=pl.Float64,
        is_elementwise=True,
    )


def explode(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Explode a list column, dropping rows whose list is null or empty."""
    return df.filter(pl.col(column).list.len() > 0).explode(column)


def get_code(expr: pl.Expr) -> pl.Expr:
    """Return the identifier prefix before the first ``_`` or ``-``, after stripping spaces."""
    return expr.str.strip_chars(' ').str.extract(r'^([^_\-]*)', 1)


def snake_to_lower_camel(name: str) -> str:
    """Convert a snake_case name to lowerCamelCase."""
    parts = name.split('_')
    return parts[0] + ''.join(p.capitalize() for p in parts[1:])


def _rename_nested(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    if isinstance(dtype, pl.Struct):
        names = [snake_to_lower_camel(f.name) for f in dtype.fields]
        expr = expr.struct.rename_fields(names)
        nested = [
            _rename_nested(pl.field(name), f.dtype).alias(name)
            for name, f in zip(names, dtype.fields, strict=True)
            if isinstance(f.dtype, (pl.Struct, pl.List))
        ]
        return expr.struct.with_fields(nested) if nested else expr
    if isinstance(dtype, pl.List) and isinstance(dtype.inner, (pl.Struct, pl.List)):
        return expr.list.eval(_rename_nested(pl.element(), dtype.inner))
    return expr


def to_camel_case(df: pl.DataFrame) -> pl.DataFrame:
    """Rename columns and nested struct fields to lowerCamelCase."""
    return df.select(
        _rename_nested(pl.col(name), dtype).alias(snake_to_lower_camel(name)) for name, dtype in df.schema.items()
    )


def _id_part(column: str, dtype: pl.DataType) -> pl.Expr:
    if isinstance(dtype, pl.Struct):
        fields = [pl.col(column).struct.field(f.name).cast(pl.String).fill_null(ID_NULL) for f in dtype.fields]
        return (
            pl
            .when(pl.col(column).is_null())
            .then(pl.lit(ID_NULL))
            .otherwise(pl.concat_str(fields, separator=ID_SEPARATOR))
        )
    return pl.col(column).cast(pl.String).fill_null(ID_NULL)


def interaction_id(schema: pl.Schema) -> pl.Expr:
    """Return a signed 64-bit xxhash64 over the interaction key columns."""
    parts = [_id_part(c, schema[c]) for c in INTERACTION_KEY]
    return (
        plh
        .concat_str(parts, separator=ID_SEPARATOR)
        .nchash.xxhash64(seed=0)
        .reinterpret(signed=True)
        .alias('interactionId')
    )


def with_interaction_id(df: pl.DataFrame) -> pl.DataFrame:
    """Append ``interactionId`` computed from the key columns."""
    return df.with_columns(interaction_id(df.schema))


def transform_rnacentral(df: pl.DataFrame) -> pl.DataFrame:
    """Map the RNAcentral file to (gene_id, mapped_id) pairs."""
    return df.select(pl.col('_c5').alias('gene_id'), pl.col('_c0').alias('mapped_id'))


def transform_human_mapping(df: pl.DataFrame) -> pl.DataFrame:
    """Collect the Ensembl entries of the UniProt human mapping file per gene."""
    return (
        df
        .filter(pl.col('_c1') == 'Ensembl')
        .group_by('_c2', maintain_order=True)
        .agg(pl.col('_c0').drop_nulls().alias('mapping_list'))
        .rename({'_c2': 'id'})
    )


def transform_gene_ids(mapping: pl.DataFrame, human_mapping: pl.DataFrame) -> pl.DataFrame:
    """Map gene names to gene ids through the UniProt accessions they share."""
    gene_ids = human_mapping.filter((pl.col('_c1') == 'Gene_Name') & pl.col('_c0').is_not_null()).select(
        '_c2', pl.col('_c0').alias('mapped_id')
    )
    combined = gene_ids.join(mapping, on='mapped_id', how='left')
    mapped = combined.filter(pl.col('gene_id').is_not_null()).select('_c2', 'gene_id').unique()
    not_mapped = combined.filter(pl.col('gene_id').is_null()).select('_c2', 'mapped_id')
    return not_mapped.join(mapped, on='_c2').select('gene_id', 'mapped_id').unique()


def generate_mapping(target: pl.DataFrame, rnacentral: pl.DataFrame, human_mapping: pl.DataFrame) -> pl.DataFrame:
    """Build the distinct gene_id to mapped_id lookup table."""
    proteins = target.select(
        'id',
        pl.col('proteinIds').list.eval(pl.element().struct.field('id')).fill_null([]).alias('proteins'),
    )
    hgnc = explode(
        target.select(
            'id',
            pl
            .col('dbXrefs')
            .list.filter(pl.element().struct.field('source') == 'HGNC')
            .list.eval(pl.element().struct.field('id'))
            .alias('hgnc'),
        ),
        'hgnc',
    ).select(pl.col('id').alias('gene_id'), ('HGNC:' + pl.col('hgnc')).alias('mapped_id'))

    mapped_proteins = proteins.join(transform_human_mapping(human_mapping), on='id', how='left').select(
        pl.col('id').alias('gene_id'),
        pl
        .when(pl.col('mapping_list').is_null())
        .then(pl.col('proteins'))
        .otherwise(pl.concat_list('proteins', 'mapping_list').list.unique(maintain_order=True))
        .alias('mapped_id'),
    )
    mapping = explode(mapped_proteins, 'mapped_id').unique()

    return pl.concat([
        mapping,
        transform_rnacentral(rnacentral),
        hgnc,
        transform_gene_ids(mapping, human_mapping),
    ]).unique()


def transform_ensembl_protein(gtf: pl.DataFrame) -> pl.DataFrame:
    """Extract distinct (gene_id, mapped_id) protein pairs from the CDS rows of an Ensembl GTF."""
    return (
        gtf
        .filter(pl.col('_c2') == 'CDS')
        .select(
            pl.col('_c8').str.extract(r'(ENSG[0-9A-Za-z_]{11})', 1).fill_null('').alias('gene_id'),
            pl.col('_c8').str.extract(r'(ENSP[0-9A-Za-z_]{11})', 1).fill_null('').alias('mapped_id'),
        )
        .unique()
    )


def _species(organism: pl.Expr) -> pl.Expr:
    return pl.struct(
        organism.struct.field('mnemonic').alias('mnemonic'),
        organism.struct.field('scientific_name').alias('scientificName'),
        organism.struct.field('taxon_id').alias('taxonId'),
    )


def generate_interactions(df: pl.DataFrame, mapping: pl.DataFrame, evidence: pl.Expr) -> pl.DataFrame:
    """Map interactors to targets and add the reverse direction of bidirectional sources.

    A missing interactor B falls back to interactor A. ``evidence`` is carried through unexploded.
    """
    a, b = pl.col('interactorA'), pl.col('interactorB')

    def b_or_a(*path: str) -> pl.Expr:
        fa, fb = a, b
        for p in path:
            fa, fb = fa.struct.field(p), fb.struct.field(p)
        return pl.when(b.struct.field(path[0]).is_null()).then(fa).otherwise(fb)

    score = pl.col('interaction').struct.field('interaction_score')
    # when/then over struct fields panics on a multi-chunk frame
    interactions = df.rechunk().select(
        a.struct.field('id').alias('intA'),
        a.struct.field('id_source').alias('intA_source'),
        _species(a.struct.field('organism')).alias('speciesA'),
        a.struct.field('biological_role').alias('intABiologicalRole'),
        b_or_a('id').alias('intB'),
        b_or_a('id_source').alias('intB_source'),
        pl.struct(
            b_or_a('organism', 'mnemonic').alias('mnemonic'),
            b_or_a('organism', 'scientific_name').alias('scientificName'),
            b_or_a('organism', 'taxon_id').alias('taxonId'),
        ).alias('speciesB'),
        b_or_a('biological_role').alias('intBBiologicalRole'),
        pl.col('source_info').struct.field('source_database').alias('sourceDatabase'),
        pl.col('source_info').alias('interactionResources'),
        evidence,
        pl.when(score > 1).then(divide(score, 1000)).otherwise(score.cast(pl.Float64)).alias('interactionScore'),
    )
    lookup = mapping.lazy()
    mapped = (
        interactions
        .lazy()
        .with_columns(get_code(pl.col('intA')).alias('_codeA'), get_code(pl.col('intB')).alias('_codeB'))
        .join(lookup.rename({'gene_id': 'targetA'}), left_on='_codeA', right_on='mapped_id', how='left')
        .join(lookup.rename({'gene_id': 'targetB'}), left_on='_codeB', right_on='mapped_id', how='left')
        .drop('_codeA', '_codeB')
        .collect()
    )
    reverse = (
        mapped
        .filter(pl.col('sourceDatabase').is_in(BIDIRECTIONAL_SOURCES) & (pl.col('intA') != pl.col('intB')))
        .select(pl.col(c).alias(SWAP.get(c, c)) for c in mapped.columns)
        .select(mapped.columns)
    )
    return pl.concat([mapped, reverse])


def transform_string(raw: pl.DataFrame, mapping: pl.DataFrame, score_threshold: int, version: str) -> pl.DataFrame:
    """Turn STRING pairs into interactions with one row per channel whose score is positive."""
    pairs = (
        raw
        .with_columns(
            pl.col('combined_score').str.strip_chars_start(' ').cast(pl.Int32, strict=False).alias('score'),
            *[pl.col(c).cast(pl.Int64, strict=False) for c, _ in STRING_EVIDENCE_CHANNELS],
        )
        .filter(
            (pl.col('score') >= score_threshold)
            & pl.col('protein1').str.contains('9606.', literal=True)
            & pl.col('protein2').str.contains('9606.', literal=True)
        )
        .with_row_index('_pair')
    )

    def interactor(protein: str) -> pl.Expr:
        return pl.struct(
            pl.lit('unspecified role').alias('biological_role'),
            pl.col(protein).str.replace_all('9606.', '', literal=True).alias('id'),
            pl.lit('ensembl_protein').alias('id_source'),
            pl.struct(
                pl.lit('human').alias('mnemonic'),
                pl.lit('Homo sapiens').alias('scientific_name'),
                pl.lit(9606, pl.Int64).alias('taxon_id'),
            ).alias('organism'),
        )

    interactions = generate_interactions(
        pairs.select(
            '_pair',
            interactor('protein1').alias('interactorA'),
            interactor('protein2').alias('interactorB'),
            pl.struct(pl.lit(version).alias('database_version'), pl.lit('string').alias('source_database')).alias(
                'source_info'
            ),
            pl.struct(pl.col('score').alias('interaction_score')).alias('interaction'),
        ),
        mapping,
        pl.col('_pair'),
    )
    evidence = pl.concat([
        pairs.filter(pl.col(channel) > 0).select(
            '_pair',
            pl.struct(
                pl.lit(channel).alias('interaction_detection_method_short_name'),
                pl.lit(mi).alias('interaction_detection_method_mi_identifier'),
                pl.col(channel).alias('evidence_score'),
                pl.lit(None, pl.String).alias('interaction_identifier'),
                pl.lit(None, pl.String).alias('pubmed_id'),
            ).alias('evidences'),
        )
        for channel, mi in STRING_EVIDENCE_CHANNELS
    ])
    return interactions.join(evidence, on='_pair', how='inner').drop('_pair')


INTACT_REQUIRED_FIELDS = {
    'interaction.evidence': pl.col('interaction').struct.field('evidence'),
    'interaction.interaction_score': pl.col('interaction').struct.field('interaction_score'),
    'interactorA.id': pl.col('interactorA').struct.field('id'),
    'source_info.source_database': pl.col('source_info').struct.field('source_database'),
}


def transform_intact(raw: pl.DataFrame, mapping: pl.DataFrame) -> pl.DataFrame:
    """Turn IntAct records into interactions with one row per evidence.

    Raises:
        ValueError: if a required field is null in every record, which is how a renamed upstream field reads
            through the fixed schema, or if no evidence row is left.
    """
    raw = raw.rechunk()
    nulls = raw.select(e.is_null().all().alias(name) for name, e in INTACT_REQUIRED_FIELDS.items())
    if missing := [name for name, all_null in nulls.row(0, named=True).items() if all_null]:
        raise ValueError(f'intact fields are null in every record: {missing}')
    interactions = generate_interactions(
        raw, mapping, pl.col('interaction').struct.field('evidence').alias('evidences')
    )
    exploded = explode(interactions, 'evidences')
    if exploded.is_empty():
        raise ValueError('intact has no evidence rows')
    return exploded


def unmatched(*frames: pl.DataFrame) -> pl.DataFrame:
    """Collect the distinct human interactor B ids that map to no target."""
    return pl.concat([
        df.filter(pl.col('targetB').is_null() & (pl.col('speciesB').struct.field('taxonId') == 9606)).select('intB')
        for df in frames
    ]).unique()


def select_fields(df: pl.DataFrame) -> pl.DataFrame:
    """Keep interactions with a target A and flatten the evidence struct."""
    return df.filter(pl.col('targetA').is_not_null()).select(
        'sourceDatabase',
        'targetA',
        'intA',
        'intA_source',
        'speciesA',
        'targetB',
        'intB',
        'intB_source',
        'speciesB',
        'interactionResources',
        'interactionScore',
        pl.col('evidences').struct.unnest(),
        'intABiologicalRole',
        'intBBiologicalRole',
    )


def aggregate(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate evidence rows into one row per interaction, scored by the maximum evidence score."""
    return with_interaction_id(
        df.group_by(INTERACTION_KEY).agg(
            pl.len().cast(pl.Int64).alias('count'), pl.col('interactionScore').max().alias('scoring')
        )
    )


def evidence(df: pl.DataFrame) -> pl.DataFrame:
    """Replace the identity columns of evidence rows with ``interactionId``."""
    return with_interaction_id(df).drop(DIMENSION_COLUMNS)


NOT_NULL_COLUMNS = ['interactionId', 'count', 'speciesA', 'speciesB']


def check_interactions(df: pl.DataFrame) -> None:
    """Raise unless ``interactionId`` is unique and the not-null columns hold no null."""
    if nulls := [c for c in NOT_NULL_COLUMNS if df[c].null_count()]:
        raise ValueError(f'interaction has nulls in {nulls}')
    if df['interactionId'].n_unique() != df.height:
        raise ValueError('interactionId is not unique')


def sort_interactions(df: pl.DataFrame) -> pl.DataFrame:
    """Order interactions by pair, with ``interactionId`` breaking ties."""
    return df.sort('sourceDatabase', 'targetA', 'targetB', 'intA', 'intB', 'interactionId')


def sort_evidences(evidences: pl.DataFrame, interactions: pl.DataFrame) -> pl.DataFrame:
    """Order evidences as their interactions, with the evidence columns breaking ties."""
    order = interactions.select('interactionId').with_row_index('_order')
    return evidences.join(order, on='interactionId', how='left').sort('_order', *evidences.columns).drop('_order')


def interaction(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Compute the interaction datasets from IntAct and STRING.

    Args:
        source: paths keyed by targets, rnacentral, humanmapping, ensproteins, intact and strings.
        destination: paths keyed by interactions, interactions_evidence and interactions_unmatched.
        settings: scorethreshold (default 0) and string_version (default '12').
        config: unused.
    """
    score_threshold = int(settings.get('scorethreshold', 0))
    string_version = str(settings.get('string_version', '12'))

    logger.info('building the id mapping')
    target = scan_dataset(source['targets']).select('id', 'proteinIds', 'dbXrefs').collect()
    rnacentral = scan_dataset(
        source['rnacentral'], format='tsv', schema=_headerless(6), has_header=False, quote_char=None
    ).collect()
    human_mapping = scan_dataset(
        source['humanmapping'], format='tsv', schema=_headerless(3), has_header=False, quote_char=None
    ).collect()
    mapping = generate_mapping(target, rnacentral, human_mapping)
    del target, rnacentral, human_mapping
    gtf = scan_dataset(
        source['ensproteins'],
        format='tsv',
        schema=_headerless(9),
        has_header=False,
        quote_char=None,
        comment_prefix='#',
    )
    string_mapping = transform_ensembl_protein(gtf.collect())
    logger.info(f'mapping has {mapping.height} rows, string mapping {string_mapping.height}')

    logger.info('transforming string')
    string_raw = pl.scan_csv(source['strings'], separator=' ', quote_char=None, infer_schema=False).collect()
    string = transform_string(string_raw, string_mapping, score_threshold, string_version)
    del string_raw, string_mapping

    logger.info('transforming intact')
    intact_raw = scan_dataset(source['intact'], format='ndjson', schema=INTACT_SCHEMA).collect()
    intact = transform_intact(intact_raw, mapping)
    del intact_raw, mapping
    logger.info(f'string has {string.height} evidence rows, intact {intact.height}')

    unmatched_ids = unmatched(intact, string)
    intact_fields, string_fields = select_fields(intact), select_fields(string)
    del intact, string

    interactions = sort_interactions(to_camel_case(pl.concat([aggregate(intact_fields), aggregate(string_fields)])))
    check_interactions(interactions)
    evidences = to_camel_case(
        pl.concat(
            [
                evidence(string_fields).with_columns(divide(pl.col('evidence_score'), 1000)),
                evidence(intact_fields),
            ],
            how='diagonal',
        )
    )
    del intact_fields, string_fields
    evidences = sort_evidences(evidences, interactions)
    unmatched_ids = unmatched_ids.sort('intB')

    if evidences['interactionId'].null_count():
        raise ValueError('interaction_evidence has a null interactionId')

    logger.info(f'writing {interactions.height} interactions to {destination["interactions"]}')
    write_dataset(interactions, destination['interactions'], schema=Interaction)
    logger.info(f'writing {evidences.height} evidences to {destination["interactions_evidence"]}')
    write_dataset(evidences, destination['interactions_evidence'], schema=InteractionEvidence)
    logger.info(f'writing {unmatched_ids.height} unmatched interactors to {destination["interactions_unmatched"]}')
    write_dataset(unmatched_ids, destination['interactions_unmatched'])
