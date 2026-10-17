"""ClinPGx pharmacogenetics.

Polars port of the `pyspark pharmacogenetics` step. Each annotation gets the phenotypes curated
from its text, a variant id from its genotype, the EFO mapping of its phenotype through
`add_efo_mapping`, and ChEMBL ids for its drugs, flagged when the drug acts on the annotation's
target.
"""

from __future__ import annotations

import gzip
import io
import json
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.pharmacogenetics import CLINPGX_SCHEMA, PHARMACOGENOMICS, PHENOTYPES_SCHEMA
from pts.transformers.evidence.utils.ontology import MAPPED_ID_COLUMN, add_efo_mapping
from pts.transformers.utils.dataset import scan_dataset, write_dataset

#: Working column: one id per annotation row, to regroup its drugs.
ROW_ID = '_operationalRowId'

_TEXT = 'genotypeAnnotationText'


def _with_schema_columns(df: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    """Add the schema's columns absent from `df` as nulls, and type those read as all-null."""
    return df.with_columns(
        pl.lit(None, dtype).alias(name) if name not in df.columns else pl.col(name).cast(dtype)
        for name, dtype in schema.items()
        if name not in df.columns or df.schema[name] == pl.Null
    )


def _read_clinpgx(path: str) -> pl.DataFrame:
    """The annotations, newline-delimited JSON, optionally gzipped."""
    data, _ = StorageHandle(path).read()
    if path.endswith('.gz'):
        data = gzip.decompress(data)
    df = pl.read_ndjson(io.BytesIO(data), infer_schema_length=None) if data.strip() else pl.DataFrame()
    return _with_schema_columns(df, CLINPGX_SCHEMA)


def _read_phenotypes(path: str) -> pl.DataFrame:
    """The lookup, a JSON array of records."""
    data, _ = StorageHandle(path).read()
    df = pl.read_json(io.BytesIO(data), infer_schema_length=None)
    return _with_schema_columns(df, PHENOTYPES_SCHEMA).select(PHENOTYPES_SCHEMA.names()).cast(PHENOTYPES_SCHEMA)


def find_unparsed_texts(annotations: pl.DataFrame, lut: pl.DataFrame) -> list[str]:
    """Return the sorted `genotypeAnnotationText` values absent from the phenotypes lookup.

    A text present in the lookup counts as parsed even if its `phenotypeText` is empty.
    """
    unparsed = annotations.select(_TEXT).drop_nulls().unique().join(lut.select(_TEXT), on=_TEXT, how='anti')
    return sorted(unparsed[_TEXT].to_list())


def annotate_phenotype(annotations: pl.DataFrame, lut: pl.DataFrame) -> pl.DataFrame:
    """Replace the ClinPGx `phenotypeText` with the phenotypes curated from `genotypeAnnotationText`.

    One row per curated phenotype; rows without any keep a null `phenotypeText`. Identical rows come out once.
    """
    return (
        annotations
        .drop('phenotypeText', 'phenotypeFromSourceId', strict=False)
        .join(lut, on=_TEXT, how='left', maintain_order='left')
        .explode('phenotypeText', empty_as_null=True, keep_nulls=True)
        .unique(maintain_order=True)
    )


def _token(parts: pl.Expr, index: int) -> pl.Expr:
    """The `index`-th token, null when missing or empty, as spark's `from_csv` reads it."""
    token = parts.list.get(index, null_on_oob=True)
    return pl.when(token != '').then(token)


def add_variant_id(annotations: pl.DataFrame) -> pl.DataFrame:
    """Add `variantId`, one row per alternate allele of the `chr_pos_ref_alt[,alt]` `genotypeId`.

    Alleles equal to the reference are skipped; a genotype without any other allele, or with a
    missing field, keeps a null `variantId`.
    """
    parts = pl.col('genotypeId').str.split('_')
    variants = (
        annotations
        .select('genotypeId')
        .drop_nulls()
        .unique(maintain_order=True)
        .with_columns(
            chr=_token(parts, 0),
            pos=_token(parts, 1),
            ref=_token(parts, 2),
            alt=_token(parts, 3).str.split(','),
        )
        .explode('alt', empty_as_null=False, keep_nulls=False)
        .filter(pl.col('ref') != pl.col('alt'))
        .select(
            'genotypeId',
            variantId=pl.concat_str('chr', 'pos', 'ref', 'alt', separator='_', ignore_nulls=True),
        )
    )
    return annotations.join(variants, on='genotypeId', how='left', maintain_order='left')


def _drug_name_lut(chembl_molecule: pl.DataFrame) -> pl.DataFrame:
    """Lower-cased drug name to the greatest ChEMBL id carrying it."""
    return (
        chembl_molecule
        .select(drugFromSource=pl.col('name').str.to_lowercase(), drugId=pl.col('id'))
        .drop_nulls('drugFromSource')
        .group_by('drugFromSource')
        .agg(pl.col('drugId').max())
    )


def _drug_target_lut(drug_mechanism_of_action: pl.DataFrame) -> pl.DataFrame:
    """ChEMBL id to the targets of all its mechanisms of action."""
    return (
        drug_mechanism_of_action
        .filter(pl.col('targets').list.len() >= 1)
        .select(drugId=pl.col('chemblIds'), drugTargetIds=pl.col('targets'))
        .explode('drugId', empty_as_null=False, keep_nulls=False)
        .group_by('drugId')
        .agg(pl.col('drugTargetIds').list.explode(keep_nulls=False, empty_as_null=False))
    )


def enrich_with_drugs(
    annotations: pl.DataFrame,
    chembl_molecule: pl.DataFrame,
    drug_mechanism_of_action: pl.DataFrame,
) -> pl.DataFrame:
    """Map each drug's lower-cased name to a ChEMBL id and flag whether it acts on the annotation's target.

    Rows without drugs are dropped. A row whose drugs differ in `isDirectTarget` is split into one
    row per flag value, each with its drugs as `{drugFromSource, drugId}`.
    """
    has_drugs = pl.col('drugs').list.len() > 0
    indexed = annotations.with_row_index(ROW_ID)
    logger.info(f'{indexed.filter(~has_drugs.fill_null(False)).height} rows without drugs dropped')

    target = pl.col('targetFromSourceId')
    exploded = (
        indexed
        .filter(has_drugs)
        .rename({'drugs': 'drug'})
        .explode('drug')
        .with_columns(drugFromSource=pl.col('drug').struct.field('drugFromSource').str.to_lowercase())
        .join(_drug_name_lut(chembl_molecule), on='drugFromSource', how='left', maintain_order='left')
        .join(_drug_target_lut(drug_mechanism_of_action), on='drugId', how='left', maintain_order='left')
        .with_columns(
            isDirectTarget=pl
            .when(target.is_not_null())
            .then(pl.col('drugTargetIds').list.contains(target))
            .fill_null(False)
        )
        .drop('drugTargetIds')
        .unique([ROW_ID, 'drug'], maintain_order=True)
    )
    kept = [c for c in exploded.columns if c not in {ROW_ID, 'isDirectTarget', 'drug', 'drugFromSource', 'drugId'}]
    return (
        exploded
        .group_by(ROW_ID, 'isDirectTarget', maintain_order=True)
        .agg(pl.col(kept).first(), drugs=pl.struct('drugFromSource', 'drugId'))
        .drop(ROW_ID)
    )


def pharmacogenetics(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Process the ClinPGx pharmacogenetics annotations.

    Args:
        source: `clinpgx` annotations, curated `phenotypes`, `ontoma_disease_label_lut`,
            `chembl_molecule` and `drug_mechanism_of_action` locations.
        destination: `associations` output directory, and `unparsed_texts`, a JSON list of the
            texts missing from the phenotypes lookup.
        settings: unused.
        config: unused.
    """
    logger.info(f'load data from {source}')
    clinpgx = _read_clinpgx(source['clinpgx'])
    phenotypes = _read_phenotypes(source['phenotypes'])
    chembl_molecule = scan_dataset(source['chembl_molecule']).select('id', 'name').collect()
    drug_mechanism_of_action = scan_dataset(source['drug_mechanism_of_action']).select('chemblIds', 'targets').collect()

    unparsed_texts = find_unparsed_texts(clinpgx, phenotypes)
    if unparsed_texts:
        logger.warning(f'{len(unparsed_texts)} genotypeAnnotationText values missing from the phenotypes lookup')
    else:
        logger.info('all phenotypes have been parsed')
    logger.info(f'save unparsed texts to {destination["unparsed_texts"]}')
    StorageHandle(destination['unparsed_texts']).write_text(json.dumps(unparsed_texts, indent=2))

    annotated = add_variant_id(annotate_phenotype(clinpgx, phenotypes))
    mapped = (
        add_efo_mapping(
            annotated.lazy(),
            disease_label_lut_path=source['ontoma_disease_label_lut'],
            label_col_name='phenotypeText',
            id_col_name=None,
        )
        .rename({MAPPED_ID_COLUMN: 'phenotypeFromSourceId'})
        .collect()
    )
    enriched = enrich_with_drugs(mapped, chembl_molecule, drug_mechanism_of_action)

    logger.info(f'save associations to {destination["associations"]}')
    write_dataset(
        enriched.select(pl.col(name).cast(dtype) for name, dtype in PHARMACOGENOMICS.items()),
        destination['associations'],
    )
