"""CLI for gentropy."""

from __future__ import annotations

import hydra  # ty: ignore[unresolved-import]
from gentropy.config import Config, register_config  # ty: ignore[unresolved-import]
from hydra.utils import instantiate  # ty: ignore[unresolved-import]
from omegaconf import OmegaConf  # ty: ignore[unresolved-import]

register_config()


@hydra.main(version_base='1.3', config_path=None, config_name='config')
def main(cfg: Config) -> None:
    """Gentropy CLI.

    Args:
        cfg (Config): configuration object.
    """
    print(OmegaConf.to_yaml(cfg))  # noqa: T201
    # Initialise and run step
    instantiate(cfg.step)


if __name__ == '__main__':
    main()
