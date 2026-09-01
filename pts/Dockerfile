FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && \
  apt-get install -y --no-install-recommends jq git && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN uv sync --frozen

# pixeltable-pgserver runs postgres under a non-root account when it detects
# uid 0, and creates that account with useradd from PostgresServer.__init__ --
# which runs BEFORE start() takes the interprocess lock. Two concurrent tasks
# would race and the loser would fail. Creating it at build time removes the
# race for good. This cannot be caught in CI, which does not run as root.
RUN useradd -m -s /bin/bash pgserver

ENTRYPOINT ["uv", "run", "pts"]
