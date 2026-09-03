#!/bin/bash
# Runs one `pipeline-supervisor observe` wakeup and exits. Invoked by cron every
# five minutes as the `orchestration` user, via the /etc/cron.d entry
# startup_machine.sh installs; that entry wraps this script in `flock -n` to
# guard against overlap -- see its comment for why -n specifically.
#
# Nothing that identifies a secret or a specific run lives on the cron line
# itself, since a crontab is world-readable: this script reads the Airflow
# password from .env and the run identifiers from observer_run.env instead.
set -euo pipefail

cd /opt/orchestration

# --- Airflow credentials -----------------------------------------------------
# Mirrors compose.yaml's own default (`${_AIRFLOW_WWW_USER_USERNAME:-airflow}`):
# an operator who has overridden these in .env is honoured, everyone else gets
# the same default the Airflow stack itself would use. Not hardcoded as a bare
# `airflow`/`airflow` literal below -- see cli.py's `_airflow_credentials`
# docstring for why a baked-in default is worth avoiding even though it
# happens to be true today. Read with grep rather than `source`d: .env also
# carries AIRFLOW__API__SECRET_KEY and AIRFLOW__API_AUTH__JWT_SECRET, and this
# script has no reason to load those into its environment at all.
ENV_FILE=/opt/pipeline/orchestration/.env
# strips one layer of matching quotes, since docker compose's own .env parsing
# accepts 'value' and "value" as well as a bare value (see .env.example, which
# quotes its own entries) and a plain `cut` would otherwise pass the quotes
# through into the exported credential.
_unquote() { sed -e "s/^['\"]//" -e "s/['\"]\$//"; }
if [ -r "$ENV_FILE" ]; then
  # `|| true` on each: .env.example does not define either key at all (compose.yaml
  # supplies the default inline), so grep finding no match is the common case, not
  # an error -- without this, `set -o pipefail` would abort the whole script right
  # here every time an operator has not overridden the default.
  FROM_ENV_USERNAME=$(grep -m1 '^_AIRFLOW_WWW_USER_USERNAME=' "$ENV_FILE" | cut -d= -f2- | _unquote) || true
  FROM_ENV_PASSWORD=$(grep -m1 '^_AIRFLOW_WWW_USER_PASSWORD=' "$ENV_FILE" | cut -d= -f2- | _unquote) || true
fi
export AIRFLOW_USERNAME="${FROM_ENV_USERNAME:-airflow}"
export AIRFLOW_PASSWORD="${FROM_ENV_PASSWORD:-airflow}"

# --- the run being watched ---------------------------------------------------
# OBSERVER_ISSUE, OBSERVER_RUN, OBSERVER_REFERENCE and OBSERVER_REFERENCE_BUCKET
# are explicit, human-chosen identifiers set in observer_run.env (seeded once
# from the tracked .example -- see that file for why none of them can be derived
# automatically). Updating them for a new run means editing that file, not this
# script or the crontab.
RUN_ENV_FILE=/opt/orchestration/observer_run.env
if [ -r "$RUN_ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$RUN_ENV_FILE"
fi

if [ -z "${OBSERVER_ISSUE:-}" ]; then
  echo "$(date -u -Iseconds) observer_run.env has no OBSERVER_ISSUE set -- nothing to comment on, skipping this wakeup"
  exit 0
fi

# observer_run.env is machine-local, is not in version control, and survives VM
# restarts, so it carries the previous run's identity into the next one unless
# someone remembers to edit it. That happened on do/platform-2609-1: the file
# still said 4505 / do/platform-2608-3, so the observer reported the new run's
# progress onto the finished run's issue and diffed the wrong bucket.
#
# OBSERVER_ISSUE cannot be checked against anything -- no artefact of a run knows
# its issue number. OBSERVER_RUN can: it must equal `run_name` in the DAG config
# this same checkout is serving. The two are edited together, so validating the
# one that is checkable catches a stale file in practice.
#
# Refuse rather than skip quietly. Silence is the observer's healthy state, so a
# quiet skip is indistinguishable from a well-behaved run -- which is why this
# went unnoticed until someone read the comments on the wrong issue.
DAG_CONFIG=/opt/orchestration/src/orchestration/dags/config/unified_pipeline.yaml
if [ -n "${OBSERVER_RUN:-}" ] && [ -r "$DAG_CONFIG" ]; then
  # sed rather than a yaml parser: this runs from cron, before `uv run` has
  # built anything. Strip the key, then a trailing comment, then trailing
  # whitespace, then one layer of quotes of either style -- in that order, so a
  # value followed by spaces still loses its closing quote. An unreadable or
  # unparseable value leaves CONFIGURED_RUN empty and skips the check below:
  # this guard exists to catch a known-wrong file, not to gate on its own
  # ability to read yaml.
  CONFIGURED_RUN=$(sed -n 's/^run_name:[[:space:]]*//p' "$DAG_CONFIG" | head -1 \
    | sed -e 's/[[:space:]]*#.*$//' -e 's/[[:space:]]*$//' -e 's/^["'"'"']//' -e 's/["'"'"']$//')
  if [ -n "$CONFIGURED_RUN" ] && [ "$OBSERVER_RUN" != "$CONFIGURED_RUN" ]; then
    echo "$(date -u -Iseconds) observer_run.env is stale: OBSERVER_RUN=${OBSERVER_RUN} but the DAG is configured for ${CONFIGURED_RUN}." >&2
    echo "  Refusing to comment -- OBSERVER_ISSUE=${OBSERVER_ISSUE} is probably the previous run's issue too." >&2
    echo "  Fix both in ${RUN_ENV_FILE}, then this resumes on the next wakeup." >&2
    exit 1
  fi
fi

ARGS=(observe --issue "$OBSERVER_ISSUE")
if [ -n "${OBSERVER_RUN:-}" ] && [ -n "${OBSERVER_REFERENCE:-}" ]; then
  ARGS+=(--run "$OBSERVER_RUN" --reference "$OBSERVER_REFERENCE")
  # A reference release does NOT always live in the bucket `--reference-bucket`
  # defaults to. Releases are published to open-targets-pre-data-releases first
  # and to the public open-targets-data-releases later, and the two do not hold
  # the same set: 26.06 exists only in the public one, while the pre-releases
  # bucket stops at 26.03.
  #
  # Getting this wrong does not fail. `collect_diffs` finds no objects under the
  # reference prefix and reports every dataset as "present in the run only" --
  # a comparison that looks like a dramatic finding and is really just the wrong
  # bucket. That is why this is worth an explicit override rather than a default
  # someone has to remember to doubt.
  if [ -n "${OBSERVER_REFERENCE_BUCKET:-}" ]; then
    ARGS+=(--reference-bucket "$OBSERVER_REFERENCE_BUCKET")
  fi
fi

# absolute path, not a bare `uv`: cron's default PATH (/usr/bin:/bin on Debian)
# does not include /usr/local/bin, where startup_machine.sh installs uv.
exec /usr/local/bin/uv run --frozen --directory /opt/orchestration pipeline-supervisor "${ARGS[@]}"
