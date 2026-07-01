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
# Returns the next rc version for a package.
#
# This script finds out the next `rc` version for a given package and returns it.
#
# To merge a feature branch into main, we open a PR in GitHub. All versions in
# main branch must be `rc` or final, and `rc` versions must be sequential.

import sys

from packaging.version import Version
from tag import (
    bail,
    ensure_branch_pushed,
    get_branch,
    get_package_version,
    parse_args,
    refresh_repo,
    subprocess_run,
)


def ensure_clean():
    if subprocess_run(['git', 'status', '--porcelain']):
        bail('repo is dirty, commit or stash changes first')


def get_next_version(package: str, current_version: Version) -> Version:
    base = current_version.release
    tags = subprocess_run([
        'git',
        'for-each-ref',
        '--merged=origin/main',
        '--format=%(refname:short)',
        f'refs/tags/{package}@*',
    ])
    latest_rc = 0
    for tag in tags.splitlines():
        parts = tag.split('@')
        if len(parts) != 2 or parts[0] != package:
            print(f'warning: ignoring invalid tag {tag}', file=sys.stderr)
            continue
        try:
            v = Version(parts[1])
        except Exception:
            print(f'ignoring dev or bad tag {tag}', file=sys.stderr)
            continue
        if v.release != base:
            continue
        if not v.is_prerelease and not v.is_devrelease:
            bail(f'final release {tag} already exists; bump the release base first')
        if v.pre and v.pre[0] == 'rc':
            latest_rc = max(latest_rc, v.pre[1])

    next_version = Version(f'{".".join(map(str, base))}rc{latest_rc + 1}')

    if next_version < current_version:
        bail(f'{next_version} < current {current_version}, check pyproject.toml')

    return next_version


def main():
    package = parse_args()

    current_branch = get_branch()
    ensure_branch_pushed(current_branch)

    if current_branch == 'main':
        bail('cannot open a pr from main branch')

    refresh_repo()
    ensure_clean()

    package_version = get_package_version(package)
    next_version = get_next_version(package, package_version)

    print(next_version)


if __name__ == '__main__':
    main()
