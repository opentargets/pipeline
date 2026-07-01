#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "packaging>=26.2",
# ]
# ///
# ruff: noqa: T201

# TAG CREATION SCRIPT
#####################
#
# Builds the tag for a given package.
#
# This script builds the tag for a given package and returns it.
#
# For the pipeline repo, we need special tags given it contains more than one
# python package inside. The tags contain more information than the version. We
# have two shapes, one for dev builds and one for rc/final builds.
#
# Dev builds: (Must be created from a branch other than main)
# ----------
# (follows https://peps.python.org/pep-0440/#developmental-releases)
#
#     <package>@v<version>-<branch>-<sha>
#     pis@v26.6.0.dev6.my-branch.123abcd
#     orchestration@v26.9.0.dev.another-branch.456efgh
#
# RC/Final builds: (Must be created from main branch)
# ---------------
# (follows https://peps.python.org/pep-0440/#pre-releases)
# NOTICE: unlike in dev builds, there is no dot between the version and rc
#
#    <package>@v<version>
#    pts@v26.9.0rc7
#    croissant@v26.9.0
#

import re
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import tomllib
from packaging.version import Version

PACKAGES = ['orchestration', 'pis', 'pts', 'croissant']


def list_to_str(lst, sep: str = '|') -> str:
    return sep.join(lst)


def bail(msg: str) -> NoReturn:
    print(f'error: {msg}', file=sys.stderr)
    sys.exit(1)


def parse_args() -> str:
    if len(sys.argv) == 2 and sys.argv[1] in PACKAGES:
        return sys.argv[1]
    bail(f'usage: pr.py <{list_to_str(PACKAGES)}>')


def subprocess_run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        bail(f'{list_to_str(cmd, ", ")} failed: {result.stderr}')
    return result.stdout.strip()


def refresh_repo():
    subprocess_run(['git', 'fetch', 'origin'])


def get_branch() -> str:
    branch = subprocess_run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    if branch == '':
        bail('detached head state, first checkout a branch')
    return branch


def normalize_branch(branch: str) -> str:
    branch = branch.lower()  # convert to lowercase
    branch = re.sub(r'[^a-z0-9]+', '-', branch)  # replace anything not a-z0-9 with hyphens
    branch = re.sub(r'--+', '-', branch)  # collapse multiple hyphens into one
    return re.sub(r'^-+|-+$', '', branch)  # remove leading and trailing hyphens


def is_branch_ancestor(a: str, b: str) -> bool:
    return subprocess.run(['git', 'merge-base', '--is-ancestor', a, b]).returncode == 0


def ensure_branch_pushed(branch: str):
    if subprocess.run(['git', 'rev-parse', '--verify', f'origin/{branch}'], capture_output=True).returncode != 0:
        bail(f'{branch} is not on origin, push it first')
    if not is_branch_ancestor('HEAD', f'origin/{branch}'):
        bail(f'HEAD is not pushed to origin/{branch}, push before building')


def get_sha() -> str:
    sha = subprocess_run(['git', 'rev-parse', '--short=7', 'HEAD'])
    if sha == '':
        bail('could not get current commit sha')
    return sha


def infer_type(v: Version) -> str:
    if v.is_devrelease:
        return 'dev'
    if v.pre and v.pre[0] == 'rc':
        return 'rc'
    if not v.is_prerelease:
        return 'final'
    bail(f'unsupported prerelease {v}; only rc/dev/final')


def get_package_version(package: str) -> Version:
    root = subprocess_run(['git', 'rev-parse', '--show-toplevel'])
    try:
        data = tomllib.loads((Path(root) / package / 'pyproject.toml').read_text())
    except Exception as e:
        bail(f'reading package version for {package}: {e}')
    return Version(data['project']['version'])


def get_latest_version(package: str) -> Version | None:
    tags = subprocess_run([
        'git',
        'for-each-ref',
        '--merged=main',
        '--format=%(refname:short)',
        f'refs/tags/{package}@*',
    ])
    if not tags:
        return None
    tags = tags.splitlines()
    versions = []
    for tag in tags:
        parts = tag.split('@')
        if len(parts) != 2 or parts[0] != package:
            print(f'warning: ignoring invalid tag {tag}', file=sys.stderr)
            continue
        _, version = parts
        try:
            versions.append(Version(version))
        except Exception:
            print(f'ignoring non-final or bad tag: {tag}', file=sys.stderr)
    versions.sort(reverse=True)
    return versions[0] if versions else None


def main():
    package = parse_args()

    refresh_repo()

    current_branch = get_branch()
    ensure_branch_pushed(current_branch)

    current_sha = get_sha()
    package_version = get_package_version(package)
    build_type = infer_type(package_version)

    if build_type == 'dev':
        if current_branch == 'main':
            bail('cannot build dev version from main branch')

        print(f'{package}@v{package_version}.{normalize_branch(current_branch)}.{current_sha}')

    if build_type in {'rc', 'final'}:
        if current_branch != 'main':
            bail('can only build rc/final version from main branch')

        # for rc/final builds, we ensure the version is greater than the last
        latest_version = get_latest_version(package)
        if latest_version:
            if package_version <= latest_version:
                bail(f'package version {package_version} must be greater than latest tag {latest_version}')

        print(f'{package}@v{package_version}')


if __name__ == '__main__':
    main()
