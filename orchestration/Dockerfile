FROM apache/airflow:slim-3.2.2-python3.12
USER root
ENV DEBIAN_FRONTEND=noninteractive

# Add g++
RUN apt-get update \
  && apt-get install -y --no-install-recommends g++ \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY pyproject.toml uv.lock README.md ./

# Install into Airflow's existing environment. Export from the lockfile so the
# container uses the same dependency versions verified by the project checks.
RUN uv export --frozen --no-dev --no-emit-project --format requirements-txt -o requirements.txt && \
  uv pip install -r requirements.txt && \
  uv pip check

USER 0
RUN curl -sSL "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz" | \
  tar -xzf - -C /opt && \
  /opt/google-cloud-sdk/install.sh --bash-completion=false --path-update=false --usage-reporting=false --quiet \
  && ln -s /opt/google-cloud-sdk/bin/gcloud /usr/local/bin/gcloud \
  && ln -s /opt/google-cloud-sdk/bin/gsutil /usr/local/bin/gsutil \
  && ln -s /opt/google-cloud-sdk/bin/bq /usr/local/bin/bq
USER airflow
