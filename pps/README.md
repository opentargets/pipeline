# PPS — Open Targets Pre-Pipeline Stage

This is the part of the Open Targets Data Pipeline that ingests and harmonises the
datasources which need heavy preprocessing (deCODE, eQTL Catalogue, FinnGen, UKB-PPP,
GWAS Catalog, …) before they become inputs to a run of the pipeline.


## Summary

This application uses the [Otter](http://github.com/opentargets/otter) library to
define the steps of the staging pipeline.

Step definitions live under `src/pps/<datasource>/`. Check out the [config.yaml](config.yaml)
file to see the steps and the tasks that make them up.


## Running

PPS uses [UV](https://docs.astral.sh/uv/) as its package manager. It is compatible
with PIP, so you can also fall back to it if you feel more comfortable.


```sh
uv run pps -h
```

> [!TIP]
> You can also use PPS with [Make](https://www.gnu.org/software/make/). Running
> `make` without any target shows help.


### Running with Docker

You can also launch PPS with Docker:
```sh
docker run ghcr.io/opentargets/pps:latest -h
```

To build your own Docker image, run the following command from the root of the
repository:

```sh
docker build -t pps .
```


## Development

> [!IMPORTANT]
> Remember to run `make dev` in the root of the repository before starting development.
> This will set up a pre-commit checks and install dependencies.

> [!TIP]
> Take a look at the [Otter docs](https://opentargets.github.io/otter), it is a
> very helpful guide when developing new tasks.


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
