# pos: Open Targets Pipeline Output Stage

Loads a release's datasets into the Platform backends (ClickHouse, OpenSearch,
BigQuery) and publishes the release (GCS, AWS, FTP, tarballs, Croissant descriptor).

> This code moved from [`opentargets/pos`](https://github.com/opentargets/pos) into
> the pipeline monorepo. Contributor and agent guidance is in [AGENTS.md](AGENTS.md).

pos is an [Otter](https://opentargets.github.io/otter) application. The steps are in
[config/config.yaml](config/config.yaml): `backend`, `tarballs`, `bigquery`,
`gcs_sync`, `aws_sync` and `ftp_sync`.

```sh
pos -s <step> -r <release_uri> -w <work> -p <pool>
```

`release_uri` ends in the run name, `<product>-YYMM-N` (`platform` or `ppp`), which sets
the release, the database namespace and the product's buckets and datasets. Run it
from the image (`pos/Dockerfile`, built from the repo root) or with `uv run pos` inside
`pos/`. `backend` needs ClickHouse on `localhost:8123` and OpenSearch on
`localhost:9200`; pos does not start them. See [AGENTS.md](AGENTS.md) for each step,
the contract with the orchestration and how to run `backend` by hand.

## Configuration

- [config/config.yaml](config/config.yaml): the steps.
- [config/datasets.yaml](config/datasets.yaml): datasets, ClickHouse tables, OpenSearch indices.
- [config/clickhouse](config/clickhouse/): SQL, schemas and server config.
- [config/opensearch](config/opensearch/): index settings.

## Copyright

Copyright 2014-2025 EMBL - European Bioinformatics Institute, Genentech, GSK,
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
