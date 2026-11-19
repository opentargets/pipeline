FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b /uv /uvx /bin/

WORKDIR /app
COPY . /app

RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "--frozen", "--no-sync", "pis"]
