r"""Genomics England PanelApp evidence.

Parses the PanelApp gene panels into evidence and post-processes it.
Phenotype strings are cleaned and split with the spark step's regular expressions, rewritten for
the Rust engine (`\d` as `[0-9]`, `$1` as `${1}`). Diseases are mapped with the OnToma lookup
tables through `add_efo_mapping`, by label and by OMIM, Orphanet, HP or MONDO id.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.schemas.panel_app import PANEL_APP_INPUT_SCHEMA, PANELS_SCHEMA
from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.expressions import confidence_score
from pts.transformers.evidence.utils.ontology import add_efo_mapping
from pts.transformers.evidence.utils.run import run_evidence
from pts.transformers.utils.dataset import scan_dataset
from pts.transformers.utils.spark_csv import read_spark_csv
from pts.transformers.utils.spark_strings import regexp_extract, spark_trim

#: Fixes applied, in order, to the phenotypes string before it is split on semicolons.
PHENOTYPE_BEFORE_SPLIT = [
    (r'\(HP:0006574;\);', '(HP:0006574);'),
    (r'Abruzzo-Erickson;syndrome', 'Abruzzo-Erickson syndrome'),
    (r'Deafness, autosomal recessive; 12', 'Deafness, autosomal recessive, 12'),
    (r'Waardenburg syndrome, type; 3', 'Waardenburg syndrome, type 3'),
    (
        r'Ectrodactyly, ectodermal dysplasia, and cleft lip/palate syndrome; 3',
        'Ectrodactyly, ectodermal dysplasia, and cleft lip/palate syndrome, 3',
    ),
    # curly braces
    (r'[{}]', ''),
    # phenotypes glued after an OMIM number
    (r'([0-9]{6})([A-Za-z])', '${1};${2}'),
    (r'[\t ]+', ' '),
    (r' ?; ?', ';'),
]

#: Phrases removed, in order, from each split phenotype.
PHENOTYPE_AFTER_SPLIT = [
    r' \(no OMIM number\)',
    r' \(NO phenotype number in OMIM\)',
    r'(no|No|NO) OMIM( phenotype|number|entry|NUMBER|NUMBER OR DISEASE)?',
    r'[( ]*(from )?PMID:? *[0-9]+[ ).]*',
]

#: Working column: the OMIM id is six digits of a longer number.
_LONGER_NUMBER = '_omimInLongerNumber'

_LEADING = r'[ ,-]*'
_SEPARATOR = r'[:_ #]*'
_TRAILING = r'[:.]*'
#: An OMIM id: group 2 is its six digits, wherever they sit.
OMIM_RE = _LEADING + r'(OMIM|MIM)?' + _SEPARATOR + r'([0-9]{6})' + _TRAILING
#: An Orphanet, HP or MONDO id: group 1 is the namespace, group 2 the number.
OTHER_RE = _LEADING + r'(OrphaNet: ORPHA|Orphanet|ORPHA|HP|MONDO)' + _SEPARATOR + r'([0-9]+)' + _TRAILING

#: A malformed publication string removed before extraction.
PMID_FILTER_OUT_RE = r'224,614,752,030,146,000,000,000'
#: Publication string parts holding PMIDs: a leading list of numbers, or numbers after `PubMed` or `PMID`.
PMID_RE = r'(^[0-9, ]+(?: |$))|((?:PubMed|PMID)[: ]*[0-9, ]+)'

PANEL_APP = EvidenceDatasource(
    datasource_id='genomics_england',
    # order feeds the id, as in the spark config
    unique_fields=['targetId', 'targetFromSourceId', 'diseaseId', 'datasourceId', 'diseaseFromSource', 'studyId'],
    score=confidence_score({'amber': 0.5, 'green': 1.0}),
)


def extract_pmids(text: pl.Expr) -> pl.Expr:
    """The distinct PMIDs of a PanelApp publication string, first seen first; `0` and ids over 8 digits are dropped."""
    cleaned = spark_trim(text.str.replace_all(r'[^\x00-\x7F]|\n|\r', ''))
    excluded = cleaned.str.replace_all(PMID_FILTER_OUT_RE, '')
    # a space between matches keeps their digit runs apart
    pmids = excluded.str.extract_all(PMID_RE).list.join(' ').str.extract_all('[0-9]+')
    return pmids.list.eval(
        pl.element().filter((pl.element() != '0') & (pl.element().str.len_chars() <= 8))
    ).list.unique(maintain_order=True)


def _explode_non_empty(lf: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """Spark's `explode`: rows with a null or empty list are dropped."""
    return lf.filter(pl.col(column).list.len() > 0).explode(column)


def parse_literature(panels: pl.LazyFrame) -> pl.LazyFrame:
    """Sorted distinct PMIDs per (`Panel Id`, `Symbol`) from the panels' gene publications."""
    genes = _explode_non_empty(
        panels.filter(pl.col('id').is_not_null()).select(pl.col('id').alias('Panel Id'), 'genes'),
        'genes',
    )
    publications = genes.select(
        'Panel Id',
        Symbol=pl.col('genes').struct.field('gene_data').struct.field('gene_symbol'),
        publication_string=pl.col('genes').struct.field('publications'),
    )
    pmids = _explode_non_empty(publications, 'publication_string').select(
        'Panel Id', 'Symbol', literature=extract_pmids(pl.col('publication_string'))
    )
    return (
        _explode_non_empty(pmids, 'literature')
        .group_by('Panel Id', 'Symbol')
        .agg(pl.col('literature').unique().sort())
        .sort('Panel Id', 'Symbol', nulls_last=True)
    )


def _read_panel_app(path: str) -> pl.LazyFrame:
    """The TSV's used columns; a `Panel Id` that is not an integer raises, other bad values are null."""
    columns = list(PANEL_APP_INPUT_SCHEMA)
    return (
        read_spark_csv(path, separator='\t')
        .lazy()
        .select(columns)
        .cast({'Panel Id': PANEL_APP_INPUT_SCHEMA['Panel Id']}, strict=True)
        .cast(PANEL_APP_INPUT_SCHEMA, strict=False)
    )


def _read_panels(path: str) -> pl.DataFrame:
    """The panels' used fields.

    Raises:
        ValueError: if the file has panels but no `id`, or genes but no `gene_symbol` or no `publications`.
    """
    panels = scan_dataset(path, format='ndjson', schema=PANELS_SCHEMA).collect()
    genes = panels.select(pl.col('genes').explode()).filter(pl.col('genes').is_not_null()).unnest('genes')
    missing = [
        name
        for name, frame, column in [
            ('id', panels, pl.col('id')),
            ('genes', panels, pl.col('genes')),
            ('gene_data.gene_symbol', genes, pl.col('gene_data').struct.field('gene_symbol')),
            ('publications', genes, pl.col('publications')),
        ]
        if frame.height and frame.select(column.is_null().all()).item()
    ]
    if missing:
        msg = f'panels in {path} carry no {missing}'
        raise ValueError(msg)
    return panels


def _phenotypes(panel_app: pl.LazyFrame) -> pl.LazyFrame:
    """Green and amber genes of public panels, one row per split phenotype, Orphanet/HP/MONDO id extracted."""
    genes = (
        panel_app
        .filter(
            pl.col('List').is_in(['green', 'amber'])
            & (pl.col('Panel Version').cast(pl.Float32) >= 1.0)
            & (pl.col('Panel Status') == 'PUBLIC')
        )
        .select('Symbol', 'Panel Id', 'Panel Name', 'List', 'Mode of inheritance', 'Phenotypes')
        .unique(maintain_order=True)
    )
    cleaned = pl.col('Phenotypes')
    for pattern, replacement in PHENOTYPE_BEFORE_SPLIT:
        cleaned = cleaned.str.replace_all(pattern, replacement)
    split = (
        genes
        .with_columns(cleanedUpPhenotypes=cleaned)
        .with_columns(cohortPhenotypes=pl.col('cleanedUpPhenotypes').str.split(';').list.unique(maintain_order=True))
        .with_columns(phenotype=pl.col('cohortPhenotypes'))
    )
    disease = pl.col('phenotype')
    for pattern in PHENOTYPE_AFTER_SPLIT:
        disease = disease.str.replace_all(pattern, '')
    namespace = regexp_extract(pl.col('diseaseFromSource'), OTHER_RE, 1).str.replace_all('OrphaNet: ORPHA', 'ORPHA')
    number = regexp_extract(pl.col('diseaseFromSource'), OTHER_RE, 2)
    return (
        _explode_non_empty(split, 'phenotype')
        .with_columns(diseaseFromSource=disease)
        .with_columns(
            ontology=pl.when((namespace != '') & (number != '')).then(pl.concat_str(namespace, pl.lit(':'), number)),
            diseaseFromSource=pl.col('diseaseFromSource').str.replace_all(OTHER_RE, ''),
        )
    )


def _with_omim(phenotypes: pl.LazyFrame) -> pl.LazyFrame:
    """Extract the OMIM id, prefer it to the other id, and clean up the label; drop `?` labels and duplicates."""
    omim_id = regexp_extract(pl.col('diseaseFromSource'), OMIM_RE, 2)
    # the leftmost OMIM match is the first six digits of the first run of six or more
    longer_number = pl.col('diseaseFromSource').str.extract('([0-9]{6,})', 1).str.len_chars() > 6
    label = spark_trim(pl.col('diseaseFromSource').str.replace_all(OMIM_RE, '').str.replace_all(r'\(\)', ''))
    return (
        phenotypes
        .with_columns(
            diseaseFromSourceId=pl.coalesce(
                pl.when(omim_id != '').then(pl.concat_str(pl.lit('OMIM:'), omim_id)), pl.col('ontology')
            ),
            diseaseFromSource=pl.when(label != '').then(label),
            **{_LONGER_NUMBER: (omim_id != '') & longer_number.fill_null(False)},
        )
        .drop('ontology')
        .filter(~(pl.col('diseaseFromSource').is_not_null() & pl.col('diseaseFromSource').str.starts_with('?')))
        .unique(maintain_order=True)
        .with_columns(
            diseaseFromSource=pl
            .when(pl.col('diseaseFromSource').is_null() & pl.col('diseaseFromSourceId').is_null())
            .then(pl.col('Panel Name'))
            .otherwise(pl.col('diseaseFromSource'))
        )
    )


def _log_ruling_counts(evidence: pl.DataFrame, longer_number: int) -> None:
    omim = evidence.select(pl.col('diseaseFromSourceId').str.starts_with('OMIM:').sum()).item()
    logger.info(f'{omim} evidence strings with an OMIM id, {longer_number} of them six digits of a longer number')
    zeros = evidence.select(pl.col('literature').list.eval(pl.element().str.contains('^0+$')).list.any().sum()).item()
    logger.info(f'{zeros} evidence strings with an all-zero PMID in their literature')


def parse_panel_app(evidence: str, panels: str) -> pl.LazyFrame:
    """Raw PanelApp evidence from the gene list TSV and the panels JSON lines, before disease mapping."""
    literature = parse_literature(_read_panels(panels).lazy())
    rows = (
        _with_omim(_phenotypes(_read_panel_app(evidence)))
        .join(literature, on=['Panel Id', 'Symbol'], how='left', maintain_order='left')
        .select(
            studyId=pl.col('Panel Id').cast(pl.String),
            targetFromSourceId=pl.col('Symbol'),
            studyOverview=pl.col('Panel Name'),
            confidence=pl.col('List'),
            cohortPhenotypes=pl.col('cohortPhenotypes'),
            diseaseFromSource=pl.col('diseaseFromSource'),
            diseaseFromSourceId=pl.col('diseaseFromSourceId'),
            literature=pl.col('literature'),
            allelicRequirements=pl
            .when(pl.col('Mode of inheritance').is_not_null())
            .then(pl.concat_list('Mode of inheritance')),
            datasourceId=pl.lit('genomics_england'),
            datatypeId=pl.lit('genetic_literature'),
            **{_LONGER_NUMBER: pl.col(_LONGER_NUMBER)},
        )
        .collect()
    )
    raw = rows.drop(_LONGER_NUMBER).unique(maintain_order=True)
    _log_ruling_counts(raw, rows.filter(pl.col(_LONGER_NUMBER)).drop(_LONGER_NUMBER).unique().height)
    return raw.lazy()


def panel_app(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process Genomics England PanelApp evidence.

    Args:
        source: `evidence` TSV, `panels` JSON lines, `ontoma_disease_label_lut`, `ontoma_disease_id_lut`,
            `disease`, `target` and `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    logger.info(f'load panelapp evidence from {source["evidence"]} and {source["panels"]}')
    raw = add_efo_mapping(
        parse_panel_app(source['evidence'], source['panels']),
        disease_label_lut_path=source['ontoma_disease_label_lut'],
        disease_id_lut_path=source['ontoma_disease_id_lut'],
    ).collect()
    run_evidence(
        raw.lazy(),
        PANEL_APP,
        disease=source['disease'],
        target=source['target'],
        destination=destination,
        literature=source['literature'],
    )
