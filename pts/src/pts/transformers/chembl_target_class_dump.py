"""Dump the five ChEMBL tables `target` builds its protein classification from.

`target` runs on Spark and so cannot query the restored postgres itself. This
restores the tables it needs and writes them to parquet unchanged -- no joins, no
flattening, no reshaping.
"""

from pathlib import Path
from typing import Any

from loguru import logger
from otter.config.model import Config

from pts.postgres import read_dump_tables
from pts.transformers.utils.dataset import write_dataset

SCHEMA_NAME = 'public'

TABLES = {
    'target_dictionary': ['tid', 'chembl_id', 'pref_name', 'target_type'],
    'target_components': ['targcomp_id', 'tid', 'component_id'],
    'component_sequences': ['component_id', 'accession'],
    'component_class': ['comp_class_id', 'component_id', 'protein_class_id'],
    'protein_classification': ['protein_class_id', 'parent_id', 'pref_name', 'class_level'],
}


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
    # scratch_root: the restore needs more room than the container filesystem has.
    tables = read_dump_tables(str(source), TABLES, schema_name=SCHEMA_NAME, scratch_root=config.work_path)

    for name, df in tables.items():
        logger.info(f'Writing {name} to {destination[name]}')
        write_dataset(df, str(destination[name]))
