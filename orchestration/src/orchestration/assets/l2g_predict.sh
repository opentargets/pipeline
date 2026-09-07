#!/bin/bash
#########################################################################################
# Run the locus-to-gene prediction step of the L2G pipeline using gentropy. This script is intended to
# be run on a Google Batch VM, where each task receives different INPUT_PARTITION and OUTPUT_PARTITION via the Batch environment, while feature_matrix_path and model_path are shared across all tasks in the job.
#
# Usage:
#   INPUT_PARTITION=gs://bucket/input_partition \
#   OUTPUT_PARTITION=gs://bucket/output_partition \
#   feature_matrix_path=gs://bucket/feature_matrix_path \
#   model_path=gs://bucket/etc/model/locus_to_gene_model \
#   l2g_predict.sh
#
#########################################################################################
set -euo pipefail
# Templated variables (defined in runnable_spec.script_variables)
readonly FEATURE_MATRIX_PATH="${feature_matrix_path}"
readonly MODEL_PATH="${model_path}"
# model_path is the DIRECTORY of this release's own l2g_training output, not a
# Hugging Face repo (opentargets/issues#4527): load_from_disk appends
# classifier.skops, while l2g_training's model_path names the file.
# The zstd codec below is a Dataproc cluster property (#31) that never reaches this
# Batch VM, so without it output/l2g_prediction ships spark's default snappy.
#########################################################################################
HYDRA_FULL_ERROR=1 gentropy \
  step=locus_to_gene \
  step.session.write_mode=overwrite \
  step.session.output_partitions=1 \
  step.run_mode="predict" \
  step.l2g_threshold=0.05 \
  step.explain_predictions=true \
  step.download_from_hub=false \
  step.model_path="${MODEL_PATH}" \
  step.credible_set_path="${INPUT_PARTITION}" \
  step.feature_matrix_path="${FEATURE_MATRIX_PATH}" \
  "+step.session.extended_spark_conf={spark.jars:https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar, spark.sql.parquet.compression.codec:zstd}" \
  step.predictions_path="${OUTPUT_PARTITION}"
