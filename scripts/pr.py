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

import subprocess
import sys

from packaging.version import Version
from tag import PACKAGES, bail, get_package_version, get_release_tags, subprocess_run

# a push to main during our run makes the bump non fast forward. that window is
# small but real: merging a second pull request while this is computing versions
# is enough. the concurrency group in bump.yaml serialises workflow runs, but
# nothing serialises a human pressing merge, so recompute against the new main
# and try again.
MAX_ATTEMPTS = 3


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


def push_succeeded(refspec: str) -> bool:
    """Push refspec to origin, reporting rejection instead of bailing."""
    result = subprocess.run(['git', 'push', 'origin', refspec], capture_output=True, text=True)
    if result.returncode == 0:
        return True
    print(result.stderr.strip(), file=sys.stderr)
    return False


def ensure_clean_tree():
    """Each attempt resets to origin/main, so refuse to run over local work.

    Only tracked files matter: `uv run` leaves a virtualenv behind, and build
    artefacts are none of our business.
    """
    dirty = subprocess_run(['git', 'status', '--porcelain', '--untracked-files=no'])
    if dirty:
        bail(f'working tree is not clean, refusing to bump:\n{dirty}')


def attempt_bump() -> bool | None:
    """Compute and push a bump against the current origin/main.

    Returns:
        True if the bump was pushed, None if there was nothing to bump, and
        False if main moved under us and the whole thing is worth retrying.
    """
    subprocess_run(['git', 'fetch', 'origin', '--tags'])
    subprocess_run(['git', 'reset', '--hard', 'origin/main'])

    bumps = []
    for package in PACKAGES:
        tags = get_release_tags(package, merged='origin/main')
        if not has_changed(package, tags):
            continue
        current = get_package_version(package)
        bumps.append((package, get_next_version(package, current, tags)))

    if not bumps:
        print('no packages changed', file=sys.stderr)
        return None

    for package, version in bumps:
        subprocess_run(['uv', '--directory', package, 'version', str(version)])
        subprocess_run(['git', 'add', f'{package}/pyproject.toml', f'{package}/uv.lock'])

    summary = ', '.join(f'{p} to {v}' for p, v in bumps)
    message = f'Bump {summary}'
    if len(message) > 50:
        message = f'Bump versions\n\n{summary}'
    subprocess_run(['git', 'commit', '-m', message])

    # tag only once the commit is on main, so a rejected push leaves no tag
    # pointing at a commit nobody else will ever see
    if not push_succeeded('HEAD:main'):
        return False

    for package, version in bumps:
        subprocess_run(['git', 'tag', f'{package}@v{version}'])
        subprocess_run(['git', 'push', 'origin', f'refs/tags/{package}@v{version}'])

    print(f'bumped {len(bumps)} packages: {summary}', file=sys.stderr)
    return True


def main():
    ensure_clean_tree()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt_bump() is not False:
            return
        print(f'main moved, retrying ({attempt}/{MAX_ATTEMPTS})', file=sys.stderr)
    bail(f'could not push a bump in {MAX_ATTEMPTS} attempts')


if __name__ == '__main__':
    main()
