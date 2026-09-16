FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY . /app

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "pis"]
