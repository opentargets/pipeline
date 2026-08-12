"""Stage the five ChEMBL tables `target` reads.

`target` is a Dataproc step with a dozen and a half other inputs, and the protein
classification it builds from these tables took three rounds to get right against
the release -- it is not being re-derived here. This step restores the five tables
`target` already expects and writes each straight to parquet: no joins, no
flattening, nothing `target`'s own verified implementation doesn't already do.
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


def chembl_target(
    source: Path,
    destination: dict[str, Path],
    _settings: dict[str, Any],
    _config: Config,
) -> None:
    """Restore the ChEMBL target tables and write each one to parquet, untouched.

    Args:
        source: Path to the ChEMBL ``pg_dump`` archive.
        destination: Table name to the parquet path to write it to.
        _settings: Custom settings (not used).
        _config: Config object (not used).
    """
    logger.info(f'Restoring {list(TABLES)} from {source}')
    tables = read_dump_tables(str(source), TABLES, schema_name=SCHEMA_NAME)

    for name, df in tables.items():
        logger.info(f'Writing {name} to {destination[name]}')
        df.write_parquet(destination[name], mkdir=True)
