"""eQTL Catalogue steps."""

from __future__ import annotations

from pydantic import BaseModel

from pps.session import Session


class IngestionSettings(BaseModel):
    """Settings for `ingestion`."""

    eqtl_catalogue_dataset_metadata_path: str
    """Path to the eQTL Catalogue dataset metadata file."""
    credible_set_input_glob: str
    """Glob pattern to read SuSIE credible set parquet files. example of files in
    `gs://bucket/susie/QTS*/QTD*/QTD*.credible_set.parquet`."""
    lbf_variable_input_glob: str
    """Glob pattern to read SuSIE LBF parquet files. example of files in
    `gs://bucket/susie/QTS*/QTD*/QTD*.lbf_variable.parquet`."""
    study_index_output_path: str
    """Path to write the study index parquet file. Written as a coalesce(1) parquet dataset."""
    credible_set_output_path: str
    """Path to write the credible set parquet file. Written as a partitioned by studyId parquet dataset."""
    lead_pvalue_threshold: float = 0.001
    """P-value threshold to flag sub-significant loci."""
    mqtl_quantification_methods_blacklist: list[str] | None = None
    """List of quantification methods to exclude from the study index. Can be used to exclude mQTL studies from the
    study index. Allowed values are in gentropy.datasource.eqtl_catalogue.QuantificationMethod."""


def ingestion(session: Session, settings: IngestionSettings) -> None:
    """EQTL Catalogue ingestion step.

    Ported from `legacy_gentropy.eqtl_catalogue.EqtlCatalogueStep`.
    """
    raise NotImplementedError
