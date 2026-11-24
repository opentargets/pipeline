FROM python:3.13-slim-trixie@sha256:7249d5587b78209a006dc4a3b5340678f84ea729239193aa559962cd8c1fce50 AS base
COPY --from=ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b /uv /uvx /bin/
COPY --from=rclone/rclone:1.76.0@sha256:7cbeec24a44f206067597c0368be4de90ae19a64d5113c7d1938bd737a81e466 /usr/local/bin/rclone /usr/local/bin/rclone

RUN apt-get update && \
  apt-get install -y --no-install-recommends pigz && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_LINK_MODE=copy

# Version-free dependency lockfile: a version bump changes pyproject.toml and
# uv.lock but not this, so the dependency layer below stays cached.
FROM base AS lock
COPY pos/pyproject.toml pos/uv.lock /src/
COPY datamodels/pyproject.toml /datamodels/pyproject.toml
RUN uv export --project /src --quiet --frozen --no-dev --no-emit-project --no-emit-package ot-datamodels --format pylock.toml \
    -o /lock/pylock.toml

FROM base AS final
COPY --from=lock /lock/pylock.toml /lock/pylock.toml
RUN --mount=type=cache,target=/root/.cache/uv \
  uv venv /app/.venv && \
  uv pip install --python /app/.venv --preview-features pylock -r /lock/pylock.toml
COPY datamodels /datamodels
COPY pos /app
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev
ENTRYPOINT ["uv", "run", "--frozen", "--no-sync", "pos"]
