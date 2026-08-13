"""Dump the five ChEMBL tables `target` builds its protein classification from.

This step does nothing but restore and write: no joins, no flattening, no
reshaping. `target` is a Dataproc step with a dozen and a half other inputs and it
already knows how to derive the classification from these tables, so the only job
here is to get them out of the dump and into parquet where a Spark step can read
them.
"""

from pathlib import Path
from typing import Any

from loguru import logger
from otter.config.model import Config

from pts.postgres import read_dump_tables

SCHEMA_NAME = 'public'
"""Schema the ChEMBL tables live in inside the restored dump."""

TABLES = {
    'target_dictionary': ['tid', 'chembl_id', 'pref_name', 'target_type'],
    'target_components': ['targcomp_id', 'tid', 'component_id'],
    'component_sequences': ['component_id', 'accession'],
    'component_class': ['comp_class_id', 'component_id', 'protein_class_id'],
    'protein_classification': ['protein_class_id', 'parent_id', 'pref_name', 'class_level'],
}
"""ChEMBL tables and columns `target` needs, restored from the dump."""


def chembl_target_class_dump(
    source: Path,
    destination: dict[str, Path],
    _settings: dict[str, Any],
    config: Config,
) -> None:
    """Restore the ChEMBL target tables and write each one to parquet, untouched.

    Args:
        source: Path to the ChEMBL ``pg_dump`` archive.
        destination: Table name to the parquet path to write it to.
        _settings: Custom settings (not used).
        config: Config object, for ``work_path``.
    """
    logger.info(f'Restoring {list(TABLES)} from {source}')
    # scratch_root: the restore needs gigabytes, and `work_path` is the work disk.
    # See the note in drug_warning.
    tables = read_dump_tables(str(source), TABLES, schema_name=SCHEMA_NAME, scratch_root=config.work_path)

    for name, df in tables.items():
        logger.info(f'Writing {name} to {destination[name]}')
        df.write_parquet(destination[name], mkdir=True)
