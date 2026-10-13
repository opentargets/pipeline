#!/bin/bash
#########################################################################################
# Create or update the two Dataproc autoscaling policies the deCODE ingestion dag
# references by name: otg-decode-efm (Prod) and otg-decode-test (Test).
#
# The policies live in GCP rather than in this repository — decode_ingestion.yaml only
# carries the policy id in `cluster_autoscaling_policy`. This script is the source of
# truth for their contents, so a policy that is deleted or drifts can be restored.
#
# `import` is idempotent: it creates the policy when absent and replaces it when
# present. Updating a policy applies to clusters already using it.
#
# Usage:
#   ./deCODE-autoscaling.sh                 # create/update both
#   ./deCODE-autoscaling.sh --dry-run       # print what would be applied
#
# Environment:
#   PROJECT   defaults to open-targets-genetics-dev
#   REGION    defaults to europe-west1
#########################################################################################
set -euo pipefail

PROJECT="${PROJECT:-open-targets-genetics-dev}"
REGION="${REGION:-europe-west1}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# Both policies pin the primary pool (minInstances == maxInstances) and let only the
# secondary pool autoscale. That is what Enhanced Flexibility Mode requires of this
# cluster: with `dataproc:efm.spark.shuffle=primary-worker` the shuffle lives on the
# primaries, so a primary leaving takes shuffle data with it, while secondaries hold
# none and can come and go freely.
#
# gracefulDecommissionTimeout MUST be 0s. Dataproc rejects cluster creation outright
# with any other value while primary-worker shuffle is enabled:
#
#   InvalidArgument: 400 When Spark primary worker shuffle is enabled
#   (dataproc:efm.spark.shuffle=primary-worker), the graceful decommissioning timeout
#   must be 0. See SPARK-30873 for more information.
#
# Per that same message, removing a secondary with no graceful decommissioning is safe:
# in-progress tasks are retried.

read -r -d '' POLICY_EFM <<'YAML' || true
# Prod: 15 fixed primaries carrying the shuffle, up to 50 secondaries for compute.
basicAlgorithm:
  cooldownPeriod: 120s
  yarnConfig:
    gracefulDecommissionTimeout: 0s
    scaleUpFactor: 1.0
    scaleDownFactor: 0.5
workerConfig:
  minInstances: 15
  maxInstances: 15
  weight: 1
secondaryWorkerConfig:
  maxInstances: 50
  weight: 1
YAML

read -r -d '' POLICY_TEST <<'YAML' || true
# Test: the same shape at small scale. Sized for the 3-study (smp) / 6-study (raw)
# subset, though still 16-vCPU workers, because the gnomAD variant_direction join
# stays full size whatever the study subset.
basicAlgorithm:
  cooldownPeriod: 120s
  yarnConfig:
    gracefulDecommissionTimeout: 0s
    scaleUpFactor: 1.0
    scaleDownFactor: 0.5
workerConfig:
  minInstances: 4
  maxInstances: 4
  weight: 1
secondaryWorkerConfig:
  maxInstances: 8
  weight: 1
YAML

apply_policy() {
  local id="$1" body="$2" tmp
  tmp="$(mktemp -t "${id}.XXXXXX.yaml")"
  printf '%s\n' "$body" >"$tmp"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "--- ${id} (dry run) ---"
    cat "$tmp"
    rm -f "$tmp"
    return
  fi

  echo "applying ${id} to ${PROJECT}/${REGION}"
  gcloud dataproc autoscaling-policies import "$id" \
    --region="$REGION" --project="$PROJECT" --source="$tmp" --quiet
  rm -f "$tmp"

  gcloud dataproc autoscaling-policies describe "$id" \
    --region="$REGION" --project="$PROJECT" \
    --format='value[separator="  "](
      basicAlgorithm.yarnConfig.gracefulDecommissionTimeout,
      workerConfig.minInstances,
      workerConfig.maxInstances,
      secondaryWorkerConfig.maxInstances)'
}

apply_policy otg-decode-efm "$POLICY_EFM"
apply_policy otg-decode-test "$POLICY_TEST"

echo "done — verify with:"
echo "  gcloud dataproc autoscaling-policies list --region=${REGION} --project=${PROJECT}"
