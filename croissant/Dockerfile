# Description: Dockerfile for the ot_croissant package

FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a /uv /uvx /bin/

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "ot_croissant"]