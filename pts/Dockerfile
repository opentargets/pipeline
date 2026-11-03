FROM python:3.12-slim-trixie@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159 AS base
COPY --from=ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a /uv /uvx /bin/

ARG TARGETARCH
RUN apt-get update && \
  apt-get install -y --no-install-recommends git && \
  if [ "$TARGETARCH" = "arm64" ]; then \
    apt-get install -y --no-install-recommends gcc libc6-dev zlib1g-dev; \
  fi && \
  rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/opentargets/pipeline/fastvep:9d84be0c14952243b4c34ffc6f3c179ae77d66ee@sha256:f47e0f1d3f734e4478471d84e40a111ffb859c9a00938852c1dcd1db03bde801 /fastvep /usr/local/bin/fastvep

# pixeltable-pgserver runs postgres under a non-root account when it detects
# uid 0, and creates that account with useradd from PostgresServer.__init__ --
# which runs BEFORE start() takes the interprocess lock. Two concurrent tasks
# would race and the loser would fail. Creating it at build time removes the
# race for good. This cannot be caught in CI, which does not run as root.
RUN useradd -m -s /bin/bash pgserver

WORKDIR /app
ENV UV_LINK_MODE=copy

# Version-free dependency lockfiles: a version bump changes pyproject.toml and
# uv.lock but not these, so the dependency layers below stay cached.
FROM base AS lock
COPY pyproject.toml uv.lock /src/
RUN uv export --project /src --quiet --frozen --no-dev --no-emit-project --format pylock.toml \
    --extra l2g -o /lock/pylock.lean.toml && \
  uv export --project /src --quiet --frozen --no-dev --no-emit-project --format pylock.toml \
    --extra l2g --extra nlp --extra llm -o /lock/pylock.nlp.toml

# Clinical staging steps: NER, stop-reason and LLM extraction.
FROM base AS nlp
COPY --from=lock /lock/pylock.nlp.toml /lock/pylock.nlp.toml
RUN --mount=type=cache,target=/root/.cache/uv \
  uv venv /app/.venv && \
  uv pip install --python /app/.venv --preview-features pylock -r /lock/pylock.nlp.toml
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev --extra l2g --extra nlp --extra llm
ENTRYPOINT ["uv", "run", "--frozen", "--no-sync", "pts"]

# Unified pipeline steps. Last stage, so it is the default build target.
FROM base AS lean
COPY --from=lock /lock/pylock.lean.toml /lock/pylock.lean.toml
RUN --mount=type=cache,target=/root/.cache/uv \
  uv venv /app/.venv && \
  uv pip install --python /app/.venv --preview-features pylock -r /lock/pylock.lean.toml
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev --extra l2g
ENTRYPOINT ["uv", "run", "--frozen", "--no-sync", "pts"]
