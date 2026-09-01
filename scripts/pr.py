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
    return next_version


def main():
    bumps = []
    for package in PACKAGES:
        tags = get_release_tags(package, merged='origin/main')
        if not has_changed(package, tags):
            continue
        current = get_package_version(package)
        bumps.append((package, get_next_version(package, current, tags)))

    if not bumps:
        print('no packages changed', file=sys.stderr)
        return

    for package, version in bumps:
        subprocess_run(['uv', '--directory', package, 'version', str(version)])
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
