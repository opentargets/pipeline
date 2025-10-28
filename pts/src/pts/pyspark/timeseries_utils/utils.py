"""Utility function to compute timeseries."""

from typing import Any

import yaml


def read_yaml_config(path:str) -> dict[str, Any]:
    """Read yaml configuration.

    Args:
        path (str): path to config file.

    Returns:
        dict[str, Any]: configuration.

    Raises:
        ValueError: Configuration is empty.
    """
    with open(path) as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Configuarion is empty.")

    return config

def get_weight_for_datasource(data_sources: list[dict[str,Any]]) -> list[tuple[str, float]]:
    """Get list of data sources' weights for overall score.

    Args:
        data_sources (list[dict[str,Any]]): pathway configuration

    Returns:
        list[tuple[str, float]]:
    """
    return [(datasource["id"], datasource["weight"]) for datasource in data_sources]
