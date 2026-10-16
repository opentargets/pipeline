"""Target safety event dataset generation.

Harmonizes raw target-safety evidence from six heterogeneous sources (adverse
events, safety risks, ToxCast, AOPWiki, Brennan/secondary pharmacology, and
pharmacogenetics) into a common schema, then resolves it against output/target
(ENSG ID resolution for symbol-keyed entries, target-ID validation) and
output/disease (obsolete EFO term remapping) to produce the final per-record
target safety event dataset.
"""

from __future__ import annotations

import io
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.target_safety_event import aopwiki_schema, biosample, brennan_schema, safety_evidence_schema, study
from pts.transformers.utils.dataset import scan_dataset, write_dataset
from pts.transformers.utils.target import ensg_lookup

EVIDENCE_UNIQUE_COLUMNS = ['id', 'targetFromSourceId', 'event', 'eventId', 'datasource', 'effects', 'literature', 'url']
OUTPUT_COLUMNS = ['targetId', 'event', 'eventId', 'effects', 'biosamples', 'datasource', 'literature', 'url', 'studies']

ACTIVATION = 'Activation/Increase/Upregulation'
INHIBITION = 'Inhibition/Decrease/Downregulation'
CLINPGX_DESCRIPTION = 'Genetic variation linked to this safety liability: '
CLINPGX_URL = 'https://www.clinpgx.org/search?query='
TOXCAST_URL = 'https://www.epa.gov/chemical-research/exploring-toxcast-data-downloadable-data'

#: Java's `\s`, `.` and `$`: ASCII whitespace, any char but a line terminator, end before an optional final one.
JAVA_WHITESPACE = r'[ \t\n\x0B\f\r]'
JAVA_DOT = r'[^\n\r\x{85}\x{2028}\x{2029}]'
JAVA_END = r'(?:\r\n|[\n\r\x{85}\x{2028}\x{2029}])?$'

WORDS_TO_REMOVE = (
    r' but not absent,|but not absent|, but not absent,|, but not absent|, but not non-existent,'
    r'(but not absent) |improved|risk and severity of|increased risk and increased severity of|'
    r'reduced risk and reduced severity of|decreased risk and reduced severity of|similar|'
    r'increased|decreased|not have altered risk of |greater|reduced|phenotype|risk of|'
    r'risk for|risk of developing|less likely to |more likely to |less severe|more severe|'
    r'smaller|likelihood of|severity of|developing|experiencing|experience|develop|'
    r'treatment related|drug-induced|oxcarbazepine-induced|aspirin induced|higher|lower|'
    r'INCREASED|Increased|including|drug toxicity, particularly |unknown likelihood of experiencing|'
    r'high frequency of'
)
CONTEXT_STOP_WORDS = (
    r'(as a result of taking|based on|when administered|during treatment|'
    r'when taking|due to unintentional|with docetaxel and thalidomide)'
)
PHENOTYPE_REPLACEMENTS = {
    'toxicity': 'drug toxicity',
    'response': 'drug response',
    'altered drug toxicity': 'drug toxicity',
    'risk and drug toxicity': 'drug toxicity',
    'risk': 'drug toxicity',
}
AE_STUDY_TYPES = {
    'Lynch et al. (2017)': 'preclinical',
    'Bowes et al. (2012)': 'preclinical',
    'Urban et al. (2012)': 'clinical',
}


def _null() -> pl.Expr:
    return pl.lit(None, dtype=pl.String)


def _trim(expr: pl.Expr) -> pl.Expr:
    """Strip spaces only, as spark's `trim` does."""
    return expr.str.strip_chars(' ')


def _as_int_text(expr: pl.Expr) -> pl.Expr:
    """Render a text column through an integer, as spark's inferred `int` column does."""
    return expr.cast(pl.Int64, strict=False).cast(pl.String)


def _direction(expr: pl.Expr, default: pl.Expr | None = None) -> pl.Expr:
    return (
        pl
        .when(expr == 'Activation')
        .then(pl.lit(ACTIVATION))
        .when(expr == 'Inhibition')
        .then(pl.lit(INHIBITION))
        .otherwise(_null() if default is None else default)
    )


def _biosample(**fields: pl.Expr) -> pl.Expr:
    return pl.struct(*(fields.get(name, _null()).alias(name) for name in biosample.to_schema()))


def _study(description: pl.Expr, name: pl.Expr, type_: pl.Expr) -> pl.Expr:
    return pl.struct(description.alias('description'), name.alias('name'), type_.alias('type'))


def _sorted_set(column: str) -> pl.Expr:
    """Distinct non-null values of a column, sorted."""
    return pl.col(column).drop_nulls().unique().sort()


def _conform(df: pl.DataFrame) -> pl.DataFrame:
    """Select the common evidence schema, adding the columns a source lacks as nulls."""
    return df.select(
        (pl.col(name) if name in df.columns else pl.lit(None)).cast(dtype).alias(name)
        for name, dtype in safety_evidence_schema.items()
    )


def harmonize_safety_evidence(safety_dfs: list[pl.DataFrame]) -> pl.DataFrame:
    """Union the per-source safety evidence frames into one common schema.

    Args:
        safety_dfs: per-source frames, each from its own `process_*` function.

    Returns:
        One row per distinct evidence record, with columns `id`, `targetFromSourceId`, `event`, `eventId`,
        `datasource`, `effects`, `literature`, `url`, `biosamples`, `studies`.
    """
    variation = CLINPGX_DESCRIPTION + _sorted_set('supporting_variation').str.join(', ')
    merged_study = (
        pl
        .when(pl.col('datasource') == 'ClinPGx')
        .then(pl.col('study').struct.with_fields(variation.alias('description')))
        .otherwise(pl.col('study'))
    )
    return (
        pl
        .concat([_conform(df) for df in safety_dfs])
        .rechunk()
        .group_by(EVIDENCE_UNIQUE_COLUMNS)
        .agg(
            _sorted_set('biosample').alias('biosamples'),
            merged_study.drop_nulls().unique().sort().alias('studies'),
        )
        .with_columns(
            pl.when(pl.col(c).list.len() != 0).then(pl.col(c)).alias(c) for c in ['biosamples', 'studies']
        )
    )


def process_aop(aopwiki_df: pl.DataFrame) -> pl.DataFrame:
    """Process the AOPWiki input."""
    return (
        aopwiki_df
        .rechunk()
        .select(
            'id',
            # data bug: some events have the substring "NA" at the start
            _trim(pl.col('event').str.replace(r'^NA', '')).alias('event'),
            'eventId',
            'datasource',
            'url',
            pl
            .col('effects')
            .list.eval(
                pl.struct(
                    _direction(pl.element().struct.field('direction')).alias('direction'),
                    pl.element().struct.field('dosing'),
                )
            )
            .alias('effects'),
            pl.col('biosamples').cast(pl.List(biosample)).alias('biosample'),
            _study(_null(), _null(), pl.lit('cell-based')).alias('study'),
        )
        .explode('biosample', empty_as_null=True)
    )


def process_adverse_events(adverse_events_df: pl.DataFrame) -> pl.DataFrame:
    """Process the adverse events input TSV.

    Multiple dosing effects of the same target, event and source are grouped into one list.
    """
    effect = pl.col('effect').str.split('_')
    ae = adverse_events_df.select(
        pl.col('ensemblId').alias('id'),
        pl.col('symptom').alias('event'),
        pl.col('efoId').alias('eventId'),
        pl.col('ref').alias('datasource'),
        _as_int_text(pl.col('pmid')).alias('literature'),
        'url',
        _biosample(tissueLabel=pl.col('biologicalSystem'), tissueId=pl.col('uberonCode')).alias('biosample'),
        pl
        .struct(
            pl
            .when(effect.list.first().str.contains('activation', literal=True))
            .then(pl.lit(ACTIVATION))
            .when(effect.list.first().str.contains('inhibition', literal=True))
            .then(pl.lit(INHIBITION))
            .alias('direction'),
            effect.list.get(1, null_on_oob=True).alias('dosing'),
        )
        .alias('effects'),
        _study(
            pl.col('symptom'),
            pl.col('efoId'),
            pl.col('ref').replace_strict(AE_STUDY_TYPES, default=pl.col('ref'), return_dtype=pl.String),
        ).alias('study'),
    )
    keys = ['id', 'event', 'datasource']
    effects = ae.group_by(keys).agg(pl.col('effects').unique().sort())
    return ae.drop('effects').join(effects, on=keys, how='left')


def process_brennan(brennan_df: pl.DataFrame) -> pl.DataFrame:
    """Process the Brennan input JSON prepared by the Target Safety team."""
    direction = pl.col('effects').struct.field('direction')
    event_id = _trim(pl.col('eventId'))
    return brennan_df.rechunk().select(
        'id',
        'event',
        # an empty eventId is a missing curated EFO term
        pl.when(event_id != '').then(event_id).alias('eventId'),
        'datasource',
        'literature',
        pl
        .concat_list(
            pl.struct(
                _direction(direction, direction).alias('direction'),
                pl.col('effects').struct.field('dosing'),
            )
        )
        .alias('effects'),
        pl.col('biosamples').cast(biosample).alias('biosample'),
        _study(*(pl.col('studies').struct.field(f) for f in study.to_schema())).alias('study'),
    )


def process_safety_risk(safety_risk_df: pl.DataFrame) -> pl.DataFrame:
    """Process the safety risk information input TSV."""
    ref = pl.col('ref')
    force = ref.str.contains('Force', literal=True)
    lamore = ref.str.contains('Lamore', literal=True)
    return safety_risk_df.select(
        pl.col('ensemblId').alias('id'),
        pl.when(force).then(pl.lit('heart disease')).when(lamore).then(pl.lit('cardiac arrhythmia')).alias('event'),
        pl.when(force).then(pl.lit('EFO_0003777')).when(lamore).then(pl.lit('EFO_0004269')).alias('eventId'),
        _as_int_text(pl.col('pmid')).alias('literature'),
        ref.alias('datasource'),
        _biosample(tissueLabel=pl.col('biologicalSystem'), tissueId=pl.col('uberonId')).alias('biosample'),
        _study(
            pl.col('liability'),
            _null(),
            pl.when(force).then(pl.lit('preclinical')).when(lamore).then(pl.lit('cell-based')),
        ).alias('study'),
    )


def process_toxcast(toxcast_df: pl.DataFrame) -> pl.DataFrame:
    """Process the ToxCast input table, keyed by gene symbol rather than ENSG id."""
    return toxcast_df.select(
        _trim(pl.col('official_symbol')).alias('targetFromSourceId'),
        pl.col('biological_process_target').alias('event'),
        'eventId',
        _biosample(
            tissueLabel=pl.col('tissue'),
            cellLabel=pl.col('cell_short_name'),
            cellFormat=pl.col('cell_format'),
        ).alias('biosample'),
        pl.lit('ToxCast').alias('datasource'),
        pl.lit(TOXCAST_URL).alias('url'),
        _study(
            pl.col('assay_component_desc'),
            pl.col('assay_component_endpoint_name'),
            pl.col('assay_format_type'),
        ).alias('study'),
    )


def process_pharmacogenetics(pgx_df: pl.DataFrame) -> pl.DataFrame:
    """Extract the pharmacogenetics evidence related to target toxicity.

    One row per supporting variation and drug, each drug being a study.
    """
    text = pl.col('phenotypeText')
    return (
        pgx_df
        .filter(
            pl.col('pgxCategory') == 'toxicity',
            pl.col('targetFromSourceId').is_not_null(),
            text.is_not_null(),
            ~(
                text.str.contains('no significant association', literal=True)
                | text.str.contains('not associated with', literal=True)
            ),
        )
        .with_columns(clean_phenotype_to_describe_safety_event(text).alias('event'))
        .filter(pl.col('event') != 'drug response')
        .select(
            pl.col('targetFromSourceId').alias('id'),
            'event',
            _null().alias('eventId'),
            pl.lit('ClinPGx').alias('datasource'),
            (CLINPGX_URL + pl.col('targetFromSourceId')).alias('url'),
            pl.col('drugs').list.eval(pl.element().struct.field('drugFromSource')).alias('drugFromSource'),
            pl.concat_list('genotypeId', 'haplotypeId').list.unique(maintain_order=True).alias('supporting_variation'),
        )
        .explode('supporting_variation')
        .explode('drugFromSource', keep_nulls=False)
        .select(
            pl.exclude('drugFromSource'),
            _study(_null(), pl.col('drugFromSource') + ' induced effect', pl.lit('clinical')).alias('study'),
        )
    )


def clean_phenotype_to_describe_safety_event(phenotype: pl.Expr) -> pl.Expr:
    """Reduce a ClinPGx phenotype text to the safety event it describes.

    Args:
        phenotype: the phenotype text.

    Returns:
        The cleaned event, with a few known phrases mapped to a fixed vocabulary.
    """
    pattern = rf'^({JAVA_DOT}*?)(?:{CONTEXT_STOP_WORDS}{JAVA_DOT}*)?{JAVA_END}'
    removed = phenotype.str.replace_all(WORDS_TO_REMOVE, '')
    extracted = pl.when(removed.is_not_null()).then(removed.str.extract(pattern, 1).fill_null(''))
    cleaned = _trim(extracted.str.replace_all(f'{JAVA_WHITESPACE}+', ' '))
    for original, replacement in PHENOTYPE_REPLACEMENTS.items():
        cleaned = pl.when(cleaned == original).then(pl.lit(replacement)).otherwise(cleaned)
    return cleaned


def build_target_safety_events(
    safety_df: pl.DataFrame,
    target_df: pl.DataFrame,
    diseases_df: pl.DataFrame,
) -> pl.DataFrame:
    """Build the target safety event output from harmonized safety evidence.

    Resolves missing ENSG ids via symbol/protein lookup, replaces obsolete EFO disease terms with their current
    equivalents, and drops any record whose (resolved) target id isn't an actual target.

    Args:
        safety_df: harmonized safety evidence from `harmonize_safety_evidence`.
        target_df: the target dataset, with `id`, `approvedSymbol` and `proteinIds`.
        diseases_df: the disease dataset, with `id` and `obsoleteTerms`.

    Returns:
        One row per target safety event record.
    """
    lookup = ensg_lookup(target_df.lazy()).collect()
    obsolete = (
        diseases_df
        .select(pl.col('id').alias('diseaseId'), pl.col('obsoleteTerms').alias('obsoleteTerm'))
        .explode('obsoleteTerm', keep_nulls=False)
    )
    return (
        safety_df
        .join(lookup, left_on='targetFromSourceId', right_on='name', how='left')
        .with_columns(pl.coalesce('id', 'ensgId').alias('id'))
        .join(obsolete, left_on='eventId', right_on='obsoleteTerm', how='left')
        .with_columns(pl.coalesce('diseaseId', 'eventId').alias('eventId'))
        .join(target_df.select('id'), on='id', how='semi')
        .select(pl.col('id').alias('targetId'), *OUTPUT_COLUMNS[1:])
    )


def _sort_output(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col('biosamples', 'studies').list.sort()).sort(pl.all())


def _read_json_array(path: str, schema: pl.Schema) -> pl.DataFrame:
    data, _ = StorageHandle(path).read()
    return pl.read_json(io.BytesIO(data), schema=schema)


def target_safety_event(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Generate the target safety event dataset.

    Args:
        source: paths keyed by adverse_events, safety_risks, toxcast, aopwiki, brennan (raw safety evidence),
            pharmacogenetics, target and diseases.
        destination: output path for the target safety event dataset.
        settings: unused.
        config: unused.
    """
    logger.info(f'load data from {source}')
    adverse_events = scan_dataset(source['adverse_events'], format='tsv', infer_schema=False).collect()
    safety_risks = scan_dataset(source['safety_risks'], format='tsv', infer_schema=False).collect()
    toxcast = scan_dataset(source['toxcast'], format='tsv', infer_schema=False).collect()
    aopwiki = _read_json_array(source['aopwiki'], aopwiki_schema)
    brennan = scan_dataset(source['brennan'], format='ndjson', schema=brennan_schema).collect()
    pharmacogenetics = (
        scan_dataset(source['pharmacogenetics'])
        .select('pgxCategory', 'targetFromSourceId', 'phenotypeText', 'genotypeId', 'haplotypeId', 'drugs')
        .collect()
    )
    target = scan_dataset(source['target']).select('id', 'approvedSymbol', 'proteinIds').collect()
    diseases = scan_dataset(source['diseases']).select('id', 'obsoleteTerms').collect()

    logger.info('harmonizing target safety evidence')
    safety = harmonize_safety_evidence([
        process_adverse_events(adverse_events),
        process_safety_risk(safety_risks),
        process_toxcast(toxcast),
        process_aop(aopwiki),
        process_pharmacogenetics(pharmacogenetics),
        process_brennan(brennan),
    ])
    result = _sort_output(build_target_safety_events(safety, target, diseases))

    logger.info(f'writing {result.height} target safety events to {destination}')
    write_dataset(result, destination)
