# Description: Dockerfile for the ot_croissant package

FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b /uv /uvx /bin/

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "--frozen", "--no-sync", "ot_croissant"]