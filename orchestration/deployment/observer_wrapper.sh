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
# OBSERVER_ISSUE, OBSERVER_RUN and OBSERVER_REFERENCE are explicit, human-chosen
# identifiers set in observer_run.env (seeded once from the tracked .example --
# see that file for why none of the three can be derived automatically).
# Updating them for a new run means editing that file, not this script or the
# crontab.
RUN_ENV_FILE=/opt/orchestration/observer_run.env
if [ -r "$RUN_ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$RUN_ENV_FILE"
fi

if [ -z "${OBSERVER_ISSUE:-}" ]; then
  echo "$(date -u -Iseconds) observer_run.env has no OBSERVER_ISSUE set -- nothing to comment on, skipping this wakeup"
  exit 0
fi

ARGS=(observe --issue "$OBSERVER_ISSUE")
if [ -n "${OBSERVER_RUN:-}" ] && [ -n "${OBSERVER_REFERENCE:-}" ]; then
  ARGS+=(--run "$OBSERVER_RUN" --reference "$OBSERVER_REFERENCE")
fi

# absolute path, not a bare `uv`: cron's default PATH (/usr/bin:/bin on Debian)
# does not include /usr/local/bin, where startup_machine.sh installs uv.
exec /usr/local/bin/uv run --frozen --directory /opt/orchestration pipeline-supervisor "${ARGS[@]}"
