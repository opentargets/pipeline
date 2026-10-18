"""Disease, target and drug label lookup table for OnToma."""

from typing import Any

from loguru import logger
from otter.config.model import Config

from pts.transformers.ontoma_lut.build import build_lut, disease_label_lut, drug_label_lut, target_label_lut
from pts.transformers.ontoma_lut.inputs import curation_label_luts, load_dataset
from pts.transformers.utils.dataset import write_dataset


def literature_ontoma(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Write the disease, target and drug label lookup table.

    Args:
        source: `disease_index`, `target_index`, `drug_index`, `ot_disease_curation`, `eva_clinvar` and
            `clinvar_xrefs` locations.
        destination: `disease_target_drug_label_lut` directory.
        settings: unused.
        config: unused.
    """
    logger.info(f'load data from: {source}')
    disease = load_dataset(source['disease_index'])
    lut = build_lut([
        disease_label_lut(disease),
        *curation_label_luts(source, disease),
        target_label_lut(load_dataset(source['target_index'])),
        drug_label_lut(load_dataset(source['drug_index'])),
    ])
    logger.info(f'save disease_target_drug label lookup table to {destination["disease_target_drug_label_lut"]}')
    write_dataset(lut, destination['disease_target_drug_label_lut'])
