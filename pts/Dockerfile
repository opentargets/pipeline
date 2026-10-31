FROM rust:1.99.1-trixie@sha256:09027f873d45b8ecd1dc86f1fa13b7a62b875c961f017a0e331267b350430137 AS fastvep
ENV RUSTUP_TOOLCHAIN=1.99.1
WORKDIR /fastvep
RUN git init -q . && \
  git fetch -q --depth 1 https://github.com/Huang-lab/fastVEP 9d84be0c14952243b4c34ffc6f3c179ae77d66ee && \
  git checkout -q FETCH_HEAD && \
  cargo build --release --locked -p fastvep-cli

FROM python:3.12-slim-trixie@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
COPY --from=ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a /uv /uvx /bin/

RUN apt-get update && \
  apt-get install -y --no-install-recommends git && \
  rm -rf /var/lib/apt/lists/*

COPY --from=fastvep /fastvep/target/release/fastvep /usr/local/bin/fastvep

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
