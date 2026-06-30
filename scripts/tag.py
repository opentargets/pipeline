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
# Builds the tag for a given package and build type.
#
# For the pipeline repo, we need special tags given it contains more than one python package inside.
# The tags contain more information than the version. We have two shapes, one for dev builds and one
# for rc/final builds.
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

import tomllib
from packaging.version import Version

PACKAGES = ['orchestration', 'pis', 'pts', 'croissant']
BUILD_TYPES = ['dev', 'rc', 'final']


def list_to_str(lst, sep: str = '|') -> str:
    return sep.join(lst)


def bail(msg: str):
    print(f'error: {msg}', file=sys.stderr)
    sys.exit(1)


def subprocess_run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        bail(f'{list_to_str(cmd, ", ")} failed: {result.stderr}')
    return result.stdout.strip()


def parse_args() -> tuple[str, str]:
    args = sys.argv[1:]
    if len(args) != 2:
        bail(f'usage: version.py <{list_to_str(PACKAGES)}> <{list_to_str(BUILD_TYPES)}>')
    package, build_type = args
    if package not in PACKAGES:
        bail(f'invalid package {package}, must be one of {list_to_str(PACKAGES)}')
    if build_type not in BUILD_TYPES:
        bail(f'invalid build type {build_type}, must be one of {list_to_str(BUILD_TYPES)}')
    return package, build_type


def get_package_version(package: str) -> Version:
    try:
        data = tomllib.loads((Path(package) / 'pyproject.toml').read_text())
    except Exception as e:
        bail(f'reading package version for {package}: {e}')
    return Version(data['project']['version'])


def get_latest_version(package: str) -> Version | None:
    # only consider tags that are merged into main, and sort them by version (descending)
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
        if parts[0] != package or len(parts) != 2:
            print(f'warning: ignoring invalid tag {tag}', file=sys.stderr)
            continue
        _, version = parts
        try:
            versions.append(Version(version))
        except Exception:
            print(f'ignoring non-final or bad tag: {tag}', file=sys.stderr)
    versions.sort(reverse=True)
    return versions[0] if versions else None


def get_branch() -> str:
    branch = subprocess_run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    if branch == '':
        bail('detached head state, first checkout a branch')
    branch = branch.lower()  # convert to lowercase
    branch = re.sub(r'[^a-z0-9]+', '-', branch)  # replace anything not a-z0-9 with hyphens
    branch = re.sub(r'--+', '-', branch)  # collapse multiple hyphens into one
    return re.sub(r'^-+|-+$', '', branch)  # remove leading and trailing hyphens


def get_sha() -> str:
    sha = subprocess_run(['git', 'rev-parse', '--short=7', 'HEAD'])
    if sha == '':
        bail('could not get current commit sha')
    return sha


package, build_type = parse_args()
current_branch = get_branch()
current_sha = get_sha()
package_version = get_package_version(package)

if build_type == 'dev':
    if current_branch == 'main':
        bail('cannot build dev version from main branch')

    # for dev builds, we ensure the package has a dev version
    if package_version.dev is None:
        bail(f'package version in pyproject.toml is {package_version}, but you attempting to build a dev version')

    print(f'{package}@v{package_version}.{current_branch}.{current_sha}')

if build_type in {'rc', 'final'}:
    if current_branch != 'main':
        bail('can only build rc/final version from main branch')

    # do not let users build a final tag from an rc/dev version
    if build_type == 'final' and package_version.is_prerelease:
        bail(f'cannot build final version from {package_version}')

    # do not let users build an rc tag from a dev version or a final version
    if build_type == 'rc' and (package_version.is_devrelease or not package_version.is_prerelease):
        bail(f'cannot build rc version from {package_version}')

    # for rc/final builds, we ensure the version is greater than the last
    latest_version = get_latest_version(package)
    if latest_version:
        if package_version <= latest_version:
            bail(f'package version {package_version} must be greater than latest tag {latest_version}')

    print(f'{package}@v{package_version}')
