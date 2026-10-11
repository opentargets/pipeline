FROM apache/airflow:slim-3.3.2-python3.12

USER airflow

COPY --from=ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a /uv /bin/uv
COPY pyproject.toml uv.lock README.md ./

# We have to install dependencies like this because uv does not seem to install
# `psycopg2-binary` properly when using the `--no-install-project` flag.
# Install exactly the locked versions, wheels only. The base image ships its own uv earlier on PATH.
RUN /bin/uv export --frozen --format requirements.txt --no-dev --no-hashes --no-emit-project -o requirements.txt \
  && /bin/uv pip install --no-deps --only-binary :all: -r requirements.txt
