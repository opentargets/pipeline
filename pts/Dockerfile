FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a /uv /uvx /bin/

RUN apt-get update && \
  apt-get install -y --no-install-recommends git && \
  rm -rf /var/lib/apt/lists/*

ADD --checksum=sha256:f867978bcfc23ee380336fe2832ea7363c983a852c2973705e2c3f6d83524032 \
  https://github.com/Huang-lab/fastVEP/releases/download/v0.4.0/fastVEP-v0.4.0-x86_64-unknown-linux-gnu.tar.gz \
  /tmp/fastvep.tar.gz
RUN tar -xzf /tmp/fastvep.tar.gz -C /tmp && \
  install -m 0755 /tmp/fastVEP-v0.4.0-x86_64-unknown-linux-gnu/fastvep /usr/local/bin/fastvep && \
  rm -rf /tmp/fastvep.tar.gz /tmp/fastVEP-v0.4.0-x86_64-unknown-linux-gnu

WORKDIR /app
COPY . /app

RUN uv sync --frozen --no-dev --extra nlp --extra l2g

# pixeltable-pgserver runs postgres under a non-root account when it detects
# uid 0, and creates that account with useradd from PostgresServer.__init__ --
# which runs BEFORE start() takes the interprocess lock. Two concurrent tasks
# would race and the loser would fail. Creating it at build time removes the
# race for good. This cannot be caught in CI, which does not run as root.
RUN useradd -m -s /bin/bash pgserver

ENTRYPOINT ["uv", "run", "--frozen", "--no-sync", "pts"]
