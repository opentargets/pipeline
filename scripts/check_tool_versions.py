#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# ruff: noqa: T201

# TOOL VERSION CHECK
####################
#
# Fails if ruff, ty or uv are not pinned to the same version everywhere:
#
#   ruff, ty: the dev group of each package's pyproject.toml, each uv.lock and
#             the pre-commit hook revs.
#   uv:       the setup-uv steps in the workflows, the uv image in the
#             Dockerfiles and the installer in the dev VM startup script.
#
# Run it from anywhere in the repo:
#
#   ./scripts/check_tool_versions.py
#
import re
import sys
from collections import defaultdict
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ('pis', 'pts', 'orchestration', 'croissant')
PYTHON_TOOLS = ('ruff', 'ty')
HOOK_REPOS = {
    'https://github.com/astral-sh/ruff-pre-commit': 'ruff',
    'https://github.com/astral-sh/ty-pre-commit': 'ty',
}
MISSING = '<missing>'


def is_exact(version: str) -> bool:
    return re.fullmatch(r'\d+(\.\d+)*', version) is not None


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def pyproject_pins(found: dict[str, list[tuple[str, str]]]) -> None:
    for package in PACKAGES:
        path = ROOT / package / 'pyproject.toml'
        dev = tomllib.loads(path.read_text()).get('dependency-groups', {}).get('dev', [])
        for tool in PYTHON_TOOLS:
            specs = [d for d in dev if re.match(rf'{tool}\s*[=<>!~]', d)]
            m = re.fullmatch(rf'{tool}\s*==\s*(\S+)', specs[0]) if len(specs) == 1 else None
            found[tool].append((f'{rel(path)} [dev]', m.group(1) if m else (specs[0] if specs else MISSING)))


def lock_versions(found: dict[str, list[tuple[str, str]]]) -> None:
    for package in PACKAGES:
        path = ROOT / package / 'uv.lock'
        locked = {p['name']: p['version'] for p in tomllib.loads(path.read_text())['package']}
        for tool in PYTHON_TOOLS:
            found[tool].append((rel(path), locked.get(tool, MISSING)))


def hook_revs(found: dict[str, list[tuple[str, str]]]) -> None:
    path = ROOT / '.pre-commit-config.yaml'
    revs: dict[str, str] = {}
    tool = None
    for line in path.read_text().splitlines():
        if m := re.match(r'\s*-\s*repo:\s*(\S+)', line):
            tool = HOOK_REPOS.get(m.group(1))
        elif tool and (m := re.match(r'\s*rev:\s*[\'"]?v?([^\'"\s]+)', line)):
            revs[tool] = m.group(1)
    for tool in PYTHON_TOOLS:
        found[tool].append((f'{rel(path)} [{tool} hook]', revs.get(tool, MISSING)))


def setup_uv_versions(found: dict[str, list[tuple[str, str]]]) -> None:
    for path in sorted((ROOT / '.github' / 'workflows').glob('*.y*ml')):
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            if 'astral-sh/setup-uv@' not in line:
                continue
            indent = len(line) - len(line.lstrip(' '))
            if not line.lstrip(' ').startswith('-'):
                indent -= 2
            version = MISSING
            for following in lines[i + 1 :]:
                if following.strip() and len(following) - len(following.lstrip(' ')) <= indent:
                    break
                if m := re.match(r'\s*version:\s*[\'"]?([^\'"\s#]+)', following):
                    version = m.group(1)
            found['uv'].append((f'{rel(path)}:{i + 1}', version))


def docker_uv_images(found: dict[str, list[tuple[str, str]]]) -> None:
    for path in sorted(ROOT.glob('*/Dockerfile')):
        for m in re.finditer(r'ghcr\.io/astral-sh/uv:([^\s@]+)(@sha256:[0-9a-f]{64})?', path.read_text()):
            found['uv'].append((rel(path), m.group(1) if m.group(2) else f'{m.group(1)} (no digest)'))


def installer_versions(found: dict[str, list[tuple[str, str]]]) -> None:
    path = ROOT / 'orchestration' / 'deployment' / 'startup_user.sh'
    for m in re.finditer(r'astral\.sh/uv/(?:([^/\s]+)/)?install\.sh', path.read_text()):
        found['uv'].append((rel(path), m.group(1) or MISSING))


def main() -> int:
    found: dict[str, list[tuple[str, str]]] = defaultdict(list)
    pyproject_pins(found)
    lock_versions(found)
    hook_revs(found)
    setup_uv_versions(found)
    docker_uv_images(found)
    installer_versions(found)

    failed = False
    width = max(len(source) for rows in found.values() for source, _ in rows)
    for tool, rows in found.items():
        counts: dict[str, int] = defaultdict(int)
        for _, version in rows:
            counts[version] += 1
        expected = max(counts, key=lambda v: (is_exact(v), counts[v]))
        ok = len(counts) == 1 and is_exact(expected)
        failed |= not ok
        print(f'{tool}: {"ok" if ok else "MISMATCH"} ({expected})')
        for source, version in rows:
            mark = '  ' if version == expected and is_exact(version) else '✗ '
            print(f'  {mark}{source:<{width}}  {version}')
    if failed:
        print('\nversions disagree or are not exact; set every location marked ✗ to one version', file=sys.stderr)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
