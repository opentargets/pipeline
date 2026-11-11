#!/usr/bin/env bash
# Build the VIDRA Stan tarball: CmdStan and the two compiled models, at the paths the pts_vidra
# Dataproc cluster installs them to (orchestration/src/orchestration/assets/install_vidra_stan.sh).
#
# CmdStan executables find the TBB library through an rpath fixed at build time, so everything is
# built under /opt/vidra inside a Debian 13 container (the Dataproc 3.0 images are Debian 13) and
# extracted to the same path on the cluster nodes. Rebuild and re-upload whenever a .stan file or
# the CmdStan version changes; the tarball name carries both, and unified_pipeline.yaml's
# vidra_stan_uri must point at the new file.
#
# Usage, from pts/ (Docker needed, ~10 minutes):
#   tools/build_vidra_stan.sh                      # writes dist/vidra-stan-<cmdstan>-<hash>.tar.gz
#   gcloud storage cp dist/vidra-stan-*.tar.gz gs://ot-team/dconsidine/vidra/stan/
set -euo pipefail

CMDSTAN_VERSION="${CMDSTAN_VERSION:-2.34.1}"
PREFIX=/opt/vidra
IMAGE="${IMAGE:-debian:trixie-slim}"

cd "$(dirname "$0")/.."
STAN_DIR=src/pts/stan/vidra
HASH=$(cat "${STAN_DIR}/VIDRA.stan" "${STAN_DIR}/VIDRA_single_variant.stan" | shasum -a 256 | cut -c1-12)
NAME="vidra-stan-${CMDSTAN_VERSION}-${HASH}.tar.gz"
mkdir -p dist

docker run --rm --platform linux/amd64 \
  -v "${PWD}/${STAN_DIR}:/stan:ro" -v "${PWD}/dist:/dist" \
  -e CMDSTAN_VERSION="${CMDSTAN_VERSION}" -e PREFIX="${PREFIX}" -e NAME="${NAME}" \
  "${IMAGE}" bash -euo pipefail -c '
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends build-essential curl ca-certificates make >/dev/null
    mkdir -p "${PREFIX}/stan"
    cd "${PREFIX}"
    curl -sSL "https://github.com/stan-dev/cmdstan/releases/download/v${CMDSTAN_VERSION}/cmdstan-${CMDSTAN_VERSION}.tar.gz" | tar xz
    cd "cmdstan-${CMDSTAN_VERSION}"
    make -j"$(nproc)" build >/dev/null
    cp /stan/VIDRA.stan /stan/VIDRA_single_variant.stan "${PREFIX}/stan/"
    make -j2 "${PREFIX}/stan/VIDRA" "${PREFIX}/stan/VIDRA_single_variant"
    rm -f "${PREFIX}"/stan/*.hpp "${PREFIX}"/stan/*.o
    rm -rf examples src/test
    "${PREFIX}/stan/VIDRA" info >/dev/null
    tar czf "/dist/${NAME}" -C / opt/vidra
  '

echo "built dist/${NAME}"
echo "upload it with: gcloud storage cp dist/${NAME} gs://ot-team/dconsidine/vidra/stan/"
echo "then set vidra_stan_uri in orchestration/src/orchestration/dags/config/unified_pipeline.yaml"
