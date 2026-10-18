"""Disease lookup tables for OnToma."""

from typing import Any

from loguru import logger
from otter.config.model import Config

from pts.transformers.ontoma_lut.build import build_lut, disease_id_lut, disease_label_lut
from pts.transformers.ontoma_lut.inputs import curation_label_luts, read_parquet
from pts.transformers.utils.dataset import write_dataset


def ontoma(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Write the disease label and disease id lookup tables.

    Args:
        source: `disease_index`, `ot_disease_curation`, `eva_clinvar` and `clinvar_xrefs` locations.
        destination: `disease_label_lut` and `disease_id_lut` directories.
        settings: unused.
        config: unused.
    """
    logger.info(f'load data from: {source}')
    disease = read_parquet(source['disease_index'])

    logger.info(f'save disease label lookup table to {destination["disease_label_lut"]}')
    write_dataset(
        build_lut([disease_label_lut(disease), *curation_label_luts(source, disease)]),
        destination['disease_label_lut'],
    )
    logger.info(f'save disease id lookup table to {destination["disease_id_lut"]}')
    write_dataset(build_lut([disease_id_lut(disease)]), destination['disease_id_lut'])
