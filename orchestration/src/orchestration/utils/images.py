"""Resolve every pipeline image a DAG config pins.

Usage: ``python -m orchestration.utils.images <dag-config.yaml>``
"""

import subprocess
import sys
from pathlib import Path

import yaml

REGISTRY = 'europe-west1-docker.pkg.dev/open-targets-eu-dev/pipeline'

IMAGES = {'pis': ['pis'], 'pts': ['pts', 'pts-nlp']}
"""The images each package publishes under one version."""


def image_refs(package_versions: dict[str, str], registry: str) -> list[str]:
    """Return the image references a set of package versions pins.

    Args:
        package_versions: package name to version.
        registry: image registry path.

    Returns:
        One reference per published image, in package order.
    """
    return [
        f'{registry}/{image}:{version}'
        for package, version in package_versions.items()
        for image in IMAGES.get(package, [package])
    ]


def resolve(ref: str) -> str | None:
    """Return the digest of an image reference, or None if it does not resolve."""
    result = subprocess.run(
        ['gcloud', 'artifacts', 'docker', 'images', 'describe', ref, '--format=value(image_summary.digest)'],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return None
    return result.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    """Print the digest of every image a DAG config pins; exit 1 if any is missing."""
    (config_path,) = sys.argv[1:] if argv is None else argv
    config = yaml.safe_load(Path(config_path).read_text())
    versions = {package: str(config[f'{package}_version']) for package in IMAGES if f'{package}_version' in config}
    if not versions:
        sys.stdout.write(f'{config_path}: no pis_version or pts_version to resolve\n')
        return 1
    missing = []
    for ref in image_refs(versions, REGISTRY):
        digest = resolve(ref)
        sys.stdout.write(f'{ref} {digest or "MISSING"}\n')
        if digest is None:
            missing.append(ref)
    if missing:
        sys.stdout.write(f'{len(missing)} image(s) do not resolve: {", ".join(missing)}\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
