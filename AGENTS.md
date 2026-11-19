# AGENTS.md

The **Open Targets Data Pipeline** monorepo generates an Open Targets data release.
It holds five independent Python packages, each with its own `pyproject.toml`,
`uv.lock` and `Dockerfile`:

- **`pis`** (Pipeline Input Stage) fetches, validates and arranges input data.
- **`pts`** (Pipeline Transformation Stage) turns inputs into the release datasets.
- **`orchestration`** holds the Airflow 3 DAGs that run the pipeline on Google Cloud.
- **`pos`** (Platform Output Support) publishes the release (GCS, FTP, AWS, tarballs) and
  generates its Croissant descriptor (`pos.croissant`).
- **`datamodels`** holds the dataset models (`ot_datamodels`). It is a library with no image; a change
  there bumps and tests pts and pos too.

Before changing a package, read its own `AGENTS.md` (`pis/`, `pts/`, `orchestration/`,
`pos/`, `datamodels/`).

## Commands

`make help` lists the root targets; `dev`, `lint`, `test` and `clean` also take a
package suffix (`make test-pts`). Run `make dev` first: it installs every package's dev
dependencies and the `prek` pre-commit hook. `make lint` runs `ruff check` and
`ty check`, after checking ruff, ty and uv are pinned to one version everywhere.

Run a package from inside its own directory; there is no root `pyproject.toml`:

```sh
cd pts && uv run pts --step <step>
cd pts && uv run --frozen pytest tests/test_foo.py::test_bar -rxs
```

- Pass `--frozen` to every lint and test command: the lockfile is authoritative.
- Relock with the uv version pinned in the Dockerfiles (`uvx uv@<pin> lock`). Older
  uv rewrites `uv.lock` from `revision = 5` to `revision = 3` with no package change,
  and CI does not catch it.
- Steps read inputs from `work/` (git-ignored) at the repo root, per `work_path` in
  each package's `config.yaml`.

## Dependencies

Build on what a package already depends on. A new dependency the unified pipeline
does not need at runtime (dev tooling, tests, one-off analysis) goes in a
`[dependency-groups]` group, so it stays out of the images.

## pis and pts

Both are thin wrappers around [Otter](https://opentargets.github.io/otter): `core.py`
builds an Otter `Runner` and registers the package's tasks. Each package's
`config.yaml` declares `steps`, each a list of tasks with a type, `source:` /
`destination:` and `settings:`. `requires:` orders tasks within a step, `foreach:`
fans a task over a list, and the `scratchpad:` block holds version variables
(`${efo_version}`) interpolated into task fields.

## Skills

Task guides live in `.agents/skills/<name>/SKILL.md` (`.claude/skills` links to the
same folder):

- `unified-pipeline-run`: running the unified pipeline on Google Cloud.
- `port-pyspark-to-polars`: moving a pts step from Spark to polars.

## Style

- `ruff.toml` at the root is shared by every package. Read it before disabling a
  rule inline.
- Keep code and config comments terse and descriptive. Rationale (measurements,
  costs, rejected alternatives) goes in the commit message and PR description.
- Start a package's commit messages with its name (`pts: only flag aact studies…`);
  other changes take a type prefix (`ci:`, `fix:`).

## Releases

Versions are per package, and tags name the package (`pts@v26.12.0rc48`).

- **rc versions come from CI.** `.github/workflows/bump.yaml` runs `scripts/pr.py` on
  every push to `main`: it bumps each package changed since its last tag, commits,
  tags and pushes, and the tag builds the image. Merging a PR is how an rc is cut.
- **A dev image comes from a dev version.** On a feature branch, set one with
  `uv --directory <package> version <base>.dev<N>` (`scripts/tag.py` builds only dev
  versions off `main`), then run `make build` inside the package directory. It pushes
  a `<package>@v<version>.<branch>.<sha>` tag, and `.github/workflows/tag.yaml` builds
  and publishes the image.
