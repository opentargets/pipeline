"""Generate lookup tables with OnToma."""

from typing import Any

from loguru import logger
from ontoma import DiseaseCuration, OnToma, OpenTargetsDisease, OpenTargetsDrug, OpenTargetsTarget

from pts.pyspark.common.session import Session


def literature_ontoma_lut_generation(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    spark = Session(app_name='ontoma', properties=properties)

    logger.info(f'load data from: {source}')
    disease_index = spark.load_data(path=source['disease_index'])
    ot_disease_curation = spark.load_data(path=source['ot_disease_curation'], format='csv', header=True, sep='\t')
    eva_clinvar = spark.load_data(path=source['eva_clinvar'], format='csv', header=True, sep='\t')
    clinvar_xrefs = spark.load_data(path=source['clinvar_xrefs'], format='csv', header=True, sep='\t')
    target_index = spark.load_data(path=source['target_index'])
    drug_index = spark.load_data(path=source['drug_index'])

    disease_index_label_lut = OpenTargetsDisease.as_label_lut(disease_index)
    ot_disease_curation_label_lut = (
        DiseaseCuration.as_label_lut(ot_disease_curation, 'ot_curation', disease_index, True)
    )
    eva_clinvar_label_lut = DiseaseCuration.as_label_lut(eva_clinvar, 'eva_clinvar', disease_index, True)
    clinvar_xrefs_label_lut = DiseaseCuration.as_label_lut(clinvar_xrefs, 'clinvar_xrefs', disease_index, True)
    target_index_label_lut = OpenTargetsTarget.as_label_lut(target_index)
    drug_index_label_lut = OpenTargetsDrug.as_label_lut(drug_index)

    OnToma(
        spark=spark.spark,
        entity_lut_list=[
            disease_index_label_lut,
            ot_disease_curation_label_lut,
            eva_clinvar_label_lut,
            clinvar_xrefs_label_lut,
            target_index_label_lut,
            drug_index_label_lut
        ],
        cache_dir=destination['disease_target_drug_label_lut'],
    )
    logger.info(f'save disease_target_drug label lookup table to {destination["disease_target_drug_label_lut"]}')
