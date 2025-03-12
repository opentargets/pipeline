import gc
import json
from pathlib import Path

import polars as pl
from loguru import logger


class FilteredJSONDecoder(json.JSONDecoder):
    """JSON Decoder that filter keys in the JSON object.

    This decoder calls a hook on each JSON object that filters out keys not in
    allowed_keys set. It also processes the root_key, so we can get to the data.

    This saves us a lot of memory and time, and serves as a workaround for the
    bug in polars that causes it to crash when loading large JSON objects:

    https://github.com/pola-rs/polars/issues/17677
    """

    def __init__(self, root_key='genes', allowed_keys=None, *args, **kwargs):
        self.root_key = root_key
        self.allowed_keys = allowed_keys or {'id', 'name'}
        super().__init__(*args, **kwargs, object_hook=self.filter_keys)

    def filter_keys(self, obj: dict) -> dict:
        if self.root_key in obj:
            return {self.root_key: obj[self.root_key]}
        return {k: v for k, v in obj.items() if k in self.allowed_keys}


def homology(source: Path, destination: Path) -> None:
    # load the homology and parse the json with our decoder
    data = json.loads(source.read_bytes(), cls=FilteredJSONDecoder)
    df = pl.from_dicts(data['genes'])
    # free up memory
    del data
    gc.collect()

    # write the result locally
    df.write_parquet(destination, compression='gzip')
    logger.info('transformation complete')
