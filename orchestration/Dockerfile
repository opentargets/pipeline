FROM apache/airflow:slim-3.3.2-python3.12

COPY --from=ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b /uv /bin/uv
COPY pyproject.toml uv.lock README.md ./

# Install exactly the locked versions, wheels only. The base image ships its own uv earlier on PATH.
RUN /bin/uv export --frozen --format requirements.txt --no-dev --no-hashes --no-emit-project -o requirements.txt \
  && /bin/uv pip install --no-deps --only-binary :all: -r requirements.txt
