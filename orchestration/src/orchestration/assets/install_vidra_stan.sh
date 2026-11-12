#!/usr/bin/env bash
# Dataproc init action of the pts_vidra cluster: installs CmdStan and VIDRA's compiled Stan models.
#
# The tarball, built by pts/tools/build_vidra_stan.sh, holds /opt/vidra/cmdstan-<version>/ and the
# executables /opt/vidra/stan/VIDRA and /opt/vidra/stan/VIDRA_single_variant, built at these paths
# because the executables find the TBB library through an rpath. VIDRA_STAN_URI, the tarball's GCS
# location, comes from the cluster metadata (clusters.yaml, vidra_stan_uri in unified_pipeline.yaml).
# cmdstanpy itself is installed with pts by install_dependencies_on_cluster.sh (PTS_EXTRAS=vidra).

set -euo pipefail
set -x

VIDRA_STAN_URI=$(/usr/share/google/get_metadata_value attributes/VIDRA_STAN_URI)
readonly VIDRA_STAN_URI

if [[ -z "${VIDRA_STAN_URI}" ]]; then
    echo "ERROR: Must specify VIDRA_STAN_URI metadata key"
    exit 1
fi

tarball=$(mktemp)
for ((i = 0; ; i++)); do
    if gcloud storage cp --no-user-output-enabled "${VIDRA_STAN_URI}" "${tarball}"; then
        break
    fi
    if ((i == 4)); then
        echo "ERROR: could not download ${VIDRA_STAN_URI}"
        exit 1
    fi
    sleep $((5 * (i + 1)))
done
tar xzf "${tarball}" -C /
rm -f "${tarball}"

# The TBB shared library next to CmdStan, for any executable run without its rpath.
ls -d /opt/vidra/cmdstan-*/stan/lib/stan_math/lib/tbb > /etc/ld.so.conf.d/vidra_tbb.conf
ldconfig

# Both executables run.
/opt/vidra/stan/VIDRA info > /dev/null
/opt/vidra/stan/VIDRA_single_variant info > /dev/null
echo "VIDRA Stan models installed from ${VIDRA_STAN_URI}"
