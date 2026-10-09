#!/usr/bin/env bash

set -euo pipefail

set -x



PTS_REF=$(/usr/share/google/get_metadata_value attributes/PTS_REF)
readonly PTS_REF
readonly REPO_URI="https://github.com/opentargets/pipeline"
readonly UV_VERSION="0.12.23"
DATAPROC_CLUSTER_NAME=$(/usr/share/google/get_metadata_value attributes/dataproc-cluster-name)
readonly DATAPROC_CLUSTER_NAME
echo "export DATAPROC_CLUSTER_NAME=${DATAPROC_CLUSTER_NAME}" >> /etc/profile.d/custom_env_vars.sh

function err() {
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
    exit 1
}

function run_with_retry() {
    local -r cmd=("$@")
    for ((i = 0; i < 3; i++)); do
        if "${cmd[@]}"; then
            return 0
        fi
        sleep 5
    done
    err "Failed to run command: ${cmd[*]}"
}

# The interpreter PySpark jobs run under. Dataproc exports it as PYSPARK_PYTHON
# from /etc/profile.d/effective-python.sh, which works the same on the 2.x
# (conda) and 3.0 (Pixi) images, so nothing here depends on where the image
# keeps its Python environment.
# https://cloud.google.com/dataproc/docs/tutorials/python-configuration
function spark_python() {
    set +u
    # shellcheck disable=SC1091
    source /etc/profile.d/effective-python.sh
    set -u
    if [[ -z "${PYSPARK_PYTHON:-}" || ! -x "${PYSPARK_PYTHON}" ]]; then
        err "PYSPARK_PYTHON is not set to an executable by /etc/profile.d/effective-python.sh"
    fi
    echo "${PYSPARK_PYTHON}"
}


function main() {
    if [[ -z "${PTS_REF}" ]]; then
        echo "ERROR: Must specify PTS_REF metadata key"
        exit 1
    fi
    local python
    python=$(spark_python)
    "${python}" --version

    # uv goes into the same environment, then installs into it explicitly with
    # --python, rather than into whichever interpreter is first on PATH.
    if ! "${python}" -m pip --version; then
        "${python}" -m ensurepip
    fi
    run_with_retry "${python}" -m pip install "uv==${UV_VERSION}"

    # Temporary: a GitHub HTTP/2 defect makes unauthenticated fetches fail on
    # some git builds -- the ref advertisement returns 200, the pack negotiation
    # 401. Covers the fetch below. Remove once GitHub resolves it.
    # https://github.com/orgs/community/discussions/206581
    git config --system http.version HTTP/1.1

    # PTS_REF is url-encoded (pts%40v...) for the git+ URL it used to go in
    local ref src
    ref=$(printf '%b' "${PTS_REF//%/\\x}")
    src=$(mktemp -d)
    git -C "${src}" init -q
    git -C "${src}" remote add origin "${REPO_URI}.git"
    git -C "${src}" sparse-checkout set pts
    run_with_retry git -C "${src}" fetch -q --depth 1 --filter=blob:none origin "${ref}"
    git -C "${src}" checkout -q FETCH_HEAD
    git -C "${src}" log -1 --format='pts source: %H %s'

    # pyspark and py4j come from the image's Spark
    "${python}" -m uv export --quiet --project "${src}/pts" --frozen --no-dev --no-emit-project \
        --format pylock.toml --no-emit-package pyspark --no-emit-package py4j \
        --output-file "${src}/pylock.toml"

    "${python}" -m uv pip uninstall --python "${python}" pts || true
    echo "Install package..."
    run_with_retry "${python}" -m uv pip install --python "${python}" --no-break-system-packages \
        --preview-features pylock -r "${src}/pylock.toml"
    run_with_retry "${python}" -m uv pip install --python "${python}" --no-break-system-packages \
        --no-deps "${src}/pts"
    # report only: packages the image ships may declare pins the lock does not meet
    "${python}" -m uv pip check --python "${python}" || true
    "${python}" -c 'import pyspark; print("pyspark", pyspark.__version__, pyspark.__file__)'
    echo "Get openai token secret..."
    # add openai token secret
    mkdir -p /var/run/secrets
    gcloud secrets versions access latest --secret="openai-token" > /var/run/secrets/openai_token
    # by name: the 'hadoop' group was gid 112 on the 2.x images, and a new base OS
    # need not keep it
    chown root:hadoop /var/run/secrets/openai_token
    chmod 440 /var/run/secrets/openai_token
}

main
