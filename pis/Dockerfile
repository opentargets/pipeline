FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a /uv /uvx /bin/

WORKDIR /app
COPY . /app

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "pis"]
