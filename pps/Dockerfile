FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# uv fetches the git dependencies (gentropy) with git
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "pps"]
