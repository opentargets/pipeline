FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && \
  apt-get install -y --no-install-recommends jq git && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "pts"]
