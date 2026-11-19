#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "packaging>=26.2",
# ]
# ///
# ruff: noqa: T201

# RC BUMP SCRIPT
################
#
# For each package changed since its last release tag on main:
# bump to the next rc version, commit, tag, and push.

import sys

from packaging.version import Version
from tag import PACKAGES, bail, get_package_version, get_release_tags, subprocess_run

DEPENDANTS: dict[str, tuple[str, ...]] = {'datamodels': ('pts', 'croissant')}


def has_changed(package: str, versions: list[Version]) -> bool:
    if not versions:
        return True  # never released
    diff = subprocess_run([
        'git',
        'diff',
        '--name-only',
        f'{package}@v{versions[-1]}',
        'HEAD',
        '--',
        f'{package}/',
        *(f'{dep}/' for dep, users in DEPENDANTS.items() if package in users),
    ])
    return bool(diff)


def get_next_version(package: str, current: Version, versions: list[Version]) -> Version:
    latest_rc = 0
    for v in versions:
        if v.release != current.release:
            continue
        if not v.is_prerelease:
            bail(f'final release {package}@v{v} already exists; bump the release base first')
        if v.pre and v.pre[0] == 'rc':
            latest_rc = max(latest_rc, v.pre[1])
    next_version = Version(f'{current.base_version}rc{latest_rc + 1}')
    if next_version < current:
        bail(f'{next_version} < current {current}, check pyproject.toml')
    if next_version == current:
        bail(
            f'{package} is already at {current} but tag {package}@v{current} does not exist; '
            f'set the version to {current.base_version}rc{latest_rc} in pyproject.toml'
        )
    return next_version


def packages_to_bump() -> list[str]:
    changed = [p for p in PACKAGES if has_changed(p, get_release_tags(p, merged='origin/main'))]
    return dependencies_first(changed)


def dependencies_first(packages: list[str]) -> list[str]:
    """Order `packages` so each one follows the packages in `DEPENDANTS` it depends on."""
    ordered: list[str] = []

    def visit(package: str) -> None:
        if package in ordered:
            return
        for dep, users in DEPENDANTS.items():
            if package in users and dep in packages:
                visit(dep)
        ordered.append(package)

    for package in packages:
        visit(package)
    return ordered


def main():
    bumps = []
    for package in packages_to_bump():
        tags = get_release_tags(package, merged='origin/main')
        current = get_package_version(package)
        bumps.append((package, get_next_version(package, current, tags)))

    if not bumps:
        print('no packages changed', file=sys.stderr)
        return

    for package, version in bumps:
        subprocess_run(['uv', '--directory', package, 'version', str(version)])

    for package, _ in bumps:
        subprocess_run(['uv', '--directory', package, 'lock', '--check'])
        subprocess_run(['git', 'add', f'{package}/pyproject.toml', f'{package}/uv.lock'])

    summary = ', '.join(f'{p} to {v}' for p, v in bumps)
    message = f'Bump {summary}'
    if len(message) > 50:
        message = f'Bump versions\n\n{summary}'
    subprocess_run(['git', 'commit', '-m', message])

    for package, version in bumps:
        subprocess_run(['git', 'tag', f'{package}@v{version}'])

    subprocess_run(['git', 'push', 'origin', 'HEAD:main'])
    for package, version in bumps:
        subprocess_run(['git', 'push', 'origin', f'refs/tags/{package}@v{version}'])

    print(f'bumped {len(bumps)} packages: {summary}', file=sys.stderr)


if __name__ == '__main__':
    main()
