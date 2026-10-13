#!/usr/bin/env bash
# Dataproc init action installing pps at the version given by the PPS_REF metadata key.
# Uploaded and attached to every staging Dataproc cluster by the staging DAG builder.

set -euo pipefail

set -x

PPS_REF=$(/usr/share/google/get_metadata_value attributes/PPS_REF)
readonly PPS_REF
readonly REPO_URI="https://github.com/opentargets/pipeline"

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

function main() {
    if [[ -z "${PPS_REF}" ]]; then
        err "Must specify PPS_REF metadata key"
    fi
    pip install uv

    # Temporary: a GitHub HTTP/2 defect makes unauthenticated fetches fail on the
    # git this image ships (2.30.2). uv shells out to /usr/bin/git, so this covers
    # the install below. Remove once GitHub resolves it.
    # https://github.com/orgs/community/discussions/206581
    git config --system http.version HTTP/1.1

    echo "Install package..."
    run_with_retry uv pip install --no-break-system-packages --system "pps @ git+${REPO_URI}.git@${PPS_REF}#subdirectory=pps"
}

main
