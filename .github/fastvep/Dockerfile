FROM rust:1.99.1-trixie@sha256:09027f873d45b8ecd1dc86f1fa13b7a62b875c961f017a0e331267b350430137 AS build
ARG FASTVEP_COMMIT
ENV RUSTUP_TOOLCHAIN=1.99.1
WORKDIR /fastvep
RUN test -n "$FASTVEP_COMMIT" && \
  git init -q . && \
  git fetch -q --depth 1 https://github.com/Huang-lab/fastVEP "$FASTVEP_COMMIT" && \
  git checkout -q FETCH_HEAD && \
  cargo build --release --locked -p fastvep-cli

FROM scratch
COPY --from=build /fastvep/target/release/fastvep /fastvep
