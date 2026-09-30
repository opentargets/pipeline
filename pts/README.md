# PTS — Open Targets Pipeline Transformation Stage

This is the main part of the Open Targets Data Pipeline. It transforms input data
into the final output files that make up an Open Targets data release.


## Summary

This application uses the [Otter](http://github.com/opentargets/otter) library to
convert some input files into parquet, and transform some ontologies we use into
structures suitable for the Open Targets pipeline.

Check out the [config.yaml](config.yaml) file to see the steps and the tasks that
make them up.


## Running

PTS uses [UV](https://docs.astral.sh/uv/) as its package manager. It is compatible
with PIP, so you can also fall back to it if you feel more comfortable.


```sh
uv run pts -h
```

> [!TIP]
> You can also use PTS with [Make](https://www.gnu.org/software/make/). Running
> `make` without any target shows help.


### Running with Docker

You can also launch PTS with Docker:
```sh
docker run ghcr.io/opentargets/pts:latest -h
```

PTS can upload the files it fetches into different cloud storage services. Open
Targets uses Google Cloud. To enable it in a docker container, you must have a
credentials file. Assuming you do, you can run the following command:

```sh
docker run \
  -v /path/to/credentials.json:/app/credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  ghcr.io/opentargets/pts:latest -h
```

To build your own Docker image, run the following command from the `pts/`
directory:

```sh
docker build -t pts .
```

For AACT extraction evaluation, build the image variant with Karenina:

```sh
docker build --target with-karenina -t pts-with-karenina .
```

The regular image does not install Karenina. The evaluation image installs the
`evaluation` optional dependency from `pyproject.toml`; both images use the
same `uv.lock`. Update the locked dependencies with:

```sh
uv lock
```

The shared lock currently resolves OpenAI 2.54.0 because Karenina's LiteLLM
dependency requires OpenAI below 3. NumPy resolves to 2.4.6. These versions
are shared by the regular and evaluation images; only the Karenina extra and
its dependencies are exclusive to the evaluation image.

Karenina is temporarily pinned to the fork commit in `pyproject.toml` while
the NumPy compatibility change is under review. Update that source and
regenerate `uv.lock` after the change is merged upstream.

The AACT judge writes `evaluation/summary.json` for counts,
`evaluation/results.parquet` for one row per sampled trial, and
`evaluation/findings.jsonl` for one review item per failed criterion. A review
item includes the suspected error, a suggested fix, a trial-text quote, and a
boolean indicating whether that quote occurs in the prompt. An empty findings
file means the sampled trials passed all four criteria; it does not establish
that the full cache is error-free.
`results.parquet` also records the judge's recommended `drug_intent` and its
reason even when the extracted intent passes.

See the [AACT evaluation guide](../orchestration/docs/datasources/aact_data/evaluation.md)
for the meaning and denominators of both `analysis/summary.json` and
`evaluation/summary.json`, and how to investigate the resulting flags and
findings.


## Development

> [!IMPORTANT]
> Remember to run `make dev` in the root of the repository before starting development.
> This will set up a pre-commit checks and install dependencies.

Development of PTS can be done straight away in the local environment. You can run
the application just like before (`uv run pts`) to check the changes you make.

> [!TIP]
> Take a look at the [Otter docs](https://opentargets.github.io/otter), it is a
> very helpful guide when developing new tasks.

You can test the changes by running a small step, like `so`:

```sh
uv run pts --step so
```


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
