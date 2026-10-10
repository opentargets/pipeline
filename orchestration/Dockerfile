FROM apache/airflow:slim-3.3.2-python3.12
ARG TARGETARCH

USER airflow

COPY --from=ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a /uv /bin/uv
COPY pyproject.toml uv.lock README.md ./

# We have to install dependencies like this because uv does not seem to install
# `psycopg2-binary` properly when using the `--no-install-project` flag.
# Install exactly the locked versions, wheels only. The base image ships its own uv earlier on PATH.
RUN /bin/uv export --frozen --format requirements.txt --no-dev --no-hashes --no-emit-project -o requirements.txt \
  && /bin/uv pip install --no-deps --only-binary :all: -r requirements.txt

USER 0
ARG GCLOUD_VERSION=588.0.0
RUN case "${TARGETARCH}" in \
    amd64) arch=x86_64; sha256=e38ceac43022bb5a4d94d5a4a9c9c51f90f28c910b59a018ae07011e220c4412 ;; \
    arm64) arch=arm; sha256=15d0365242d72caebda77647642ae73f68bde10b1f71338c8a704aea200d8ab1 ;; \
    *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
  esac \
  && curl -sSfL -o /tmp/gcloud.tar.gz \
    "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-${GCLOUD_VERSION}-linux-${arch}.tar.gz" \
  && echo "${sha256}  /tmp/gcloud.tar.gz" | sha256sum -c - \
  && tar -xzf /tmp/gcloud.tar.gz -C /opt \
  && rm /tmp/gcloud.tar.gz \
  && /opt/google-cloud-sdk/install.sh --bash-completion=false --path-update=false --usage-reporting=false --quiet \
  && ln -s /opt/google-cloud-sdk/bin/gcloud /usr/local/bin/gcloud \
  && ln -s /opt/google-cloud-sdk/bin/gsutil /usr/local/bin/gsutil \
  && ln -s /opt/google-cloud-sdk/bin/bq /usr/local/bin/bq
USER airflow
