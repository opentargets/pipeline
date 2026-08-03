FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY . /app

RUN uv sync --frozen

# duckdb downloads the postgres extension on first use, which the postgres_export
# task would otherwise do inside the pipeline VM, at the mercy of a third party
# CDN. bake it in instead. running locally needs none of this: duckdb just
# fetches it once into ~/.duckdb/extensions
RUN uv run python -c "import duckdb; duckdb.connect().install_extension('postgres')"

ENTRYPOINT ["uv", "run", "pis"]
