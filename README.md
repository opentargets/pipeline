# Open Targets Data Pipeline

Generates an Open Targets Data Release.

## Packages

The Open Targets Data Pipeline is made up of four parts, each being a python package:

- [pis](pis/README.md) — Pipeline Input Stage: fetches, validates, and arranges
    input data.
- [pts](pts/README.md) — Pipeline Transformation Stage: transforms input data
    into the output files.
- [orchestration](orchestration/README.md) — Airflow DAGs that run the full pipeline.
- [croissant](croissant/README.md) — generates a Croissant-compliant dataset for
    the output data.


## Running

Refer to each project's README.md for instructions on how to run it.

### Requirements

- [uv](https://docs.astral.sh/uv/)
- git


## Development

> [!IMPORTANT]
> If using Visual Studio Code, remember to install recommended extensions, and to
> open the workspace file `pipeline.code-workspace` instead of the root folder!

| Make Target  | Description                                     |
| ------------ | ----------------------------------------------- |
| `make dev`   | Install dev dependencies and pre-commit hook.   |
| `make lint`  | Run `ruff` and `ty` across all packages.        |
| `make test`  | Run `pytest` across all packages.               |
| `make clean` | Remove build and test artifacts.                |

> [!TIP]
> Each target supports a package suffix (e.g. `make test-pts`) to limit scope.

### Workflows

> [!NOTE]
Take a look at [the naming conventions](https://opentargets.org/technical-knowledge-base/shared-practices/naming-conventions/)
and [git workflow](https://opentargets.org/technical-knowledge-base/shared-practices/git-workflow/)
articles of the Open Targets Technical Knowledge Base to understand our naming
conventions and the git workflow we follow.

#### Build a new version of a package

First, ensure you have committed and pushed all the changes. To trigger a new build,
just run:

```sh
make build
```

from inside the directory of the package you want to build. This will generate the
proper tag for you based on the package's `pyproject.toml`, the current branch, and
last commit's sha. After confirmation, it will create and push that tag, which will
trigger the [`tag.yaml`](.github/workflows/tag.yaml) workflow to run checks, build
and publish the image.


#### Open a PR

To prepare a branch for a PR, run:

```sh
make pr
```

on your feature branch (pushed, with a clean tree). This will compute the next sequential
release-candidate version for the package; bump `pyproject.toml`, update the lockfile,
commit, and push. Then you can open a PR against `main`.


## Additional information

### Tag format

#### Some notes on the commit messages:
* We do not use semantic commits anymore (see the naming conventions for reasons).
* When a commit relates to a specific package, the commit message must start with
    the package name (e.g. `pis: add new input data source`). This is not enforced.

#### Tagging
As this repository contains multiple Python packages, the tags are a bit different than
the usual `v<version>` format:

- **`dev`**: (can only be created in branches other than `main`)

    `<package>@v<version>.<branch>.<sha>`

    `pis@v26.6.0.dev6.my-branch.123abcd`


- **`rc` / `final`**: (can only be created on `main`)

    `<package>@v<version>`

    `pts@v26.9.0rc7`</br>
    `croissant@v26.9.0`


## Copyright

Copyright 2014-2026 EMBL - European Bioinformatics Institute, Genentech, GSK,
MSD, Pfizer, Sanofi and Wellcome Sanger Institute

This software was developed as part of the Open Targets project. For more
information please see: http://www.opentargets.org

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
