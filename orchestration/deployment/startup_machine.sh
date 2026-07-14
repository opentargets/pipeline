#!/bin/bash
set -x
set -e

# google runs this on every boot, not just the first one, so every step below has to be
# safe to repeat. clear the readiness flag first: start.sh polls for it over ssh, and a
# flag left over from an earlier boot would report a run that never happened as a success.
rm -f /ready

# remove man
apt-get remove -y --purge man-db

# update package lists
apt-get update -y

# install dependencies
apt-get install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  git \
  build-essential

# add docker gpg key
curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --batch --yes --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# add docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# update package lists again
apt-get update -y

# install docker
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# clone repository
if [ ! -d /opt/pipeline/.git ]; then
  git clone https://github.com/opentargets/pipeline /opt/pipeline
fi
cd /opt/pipeline
BRANCH=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/orchestration_git_branch)
# fetch so the checkout resolves on a reboot too, when the clone above is skipped
git fetch --all --tags --prune
git checkout "$BRANCH"
ln -sfn /opt/pipeline/orchestration /opt/orchestration

# set proper ownership and permissions
# all google cloud iam users are members of google-sudoers, so we use that group to avoid having to
# add groups to users which seems to be a mess in google cloud vms
chgrp -R google-sudoers /opt/pipeline
chmod -R g+rw /opt/pipeline

# create orchestration user
if ! id -u orchestration > /dev/null 2>&1; then
  useradd -m -G google-sudoers,docker orchestration
fi

REMOTE_AIRFLOW_SERVICES="postgres airflow-init airflow-scheduler airflow-dag-processor airflow-triggerer airflow-apiserver"

# seed .env from the tracked example, then generate the airflow secrets into it.
# compose.yaml declares the secrets as required interpolation vars, so they have to be
# readable by every `docker compose` invocation (up, ps, logs) and not just the initial
# `up` -- the wait_for_* helpers below shell out to `docker compose ps` as root.
# .env is git-ignored, so the generated secrets never land in a tracked file.
# only seed once: regenerating on a reboot would rotate the signing secrets out from under
# the sessions and jwts the running stack already issued.
if [ ! -f /opt/pipeline/orchestration/.env ]; then
  cp /opt/pipeline/orchestration/.env.example /opt/pipeline/orchestration/.env
  cat >> /opt/pipeline/orchestration/.env <<EOF
AIRFLOW__API__SECRET_KEY=$(openssl rand -hex 32)
AIRFLOW__API_AUTH__JWT_SECRET=$(openssl rand -hex 32)
AIRFLOW__API_AUTH__JWT_ISSUER=airflow
EOF
fi

# the airflow signing secrets live in here, so keep it off the world bits: root writes it,
# google-sudoers (which the orchestration user is in) only needs to read it. enforced on
# every boot, since the recursive chmod above hands group write back to everything.
chgrp google-sudoers /opt/pipeline/orchestration/.env
chmod 640 /opt/pipeline/orchestration/.env

fail_service_startup() {
  SERVICE_NAME="$1"
  cd /opt/pipeline/orchestration
  docker compose ps --all "$SERVICE_NAME"
  docker compose logs --no-color --tail=50 "$SERVICE_NAME"
  exit 1
}

wait_for_airflow_init() {
  while true; do
    CONTAINER_ID=$(cd /opt/pipeline/orchestration && docker compose ps --all -q airflow-init)
    if [ -n "$CONTAINER_ID" ]; then
      STATUS=$(docker inspect --format '{{.State.Status}}' "$CONTAINER_ID")
      EXIT_CODE=$(docker inspect --format '{{.State.ExitCode}}' "$CONTAINER_ID")
      if [ "$STATUS" = "exited" ] && [ "$EXIT_CODE" = "0" ]; then
        break
      elif [ "$STATUS" = "exited" ]; then
        fail_service_startup airflow-init
      fi
    fi
    sleep 5
  done
}

wait_for_healthy_service() {
  SERVICE_NAME="$1"
  while true; do
    CONTAINER_ID=$(cd /opt/pipeline/orchestration && docker compose ps --all -q "$SERVICE_NAME")
    if [ -n "$CONTAINER_ID" ]; then
      STATUS=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$CONTAINER_ID")
      if [ "$STATUS" = "healthy" ]; then
        break
      elif [ "$STATUS" = "unhealthy" ] || [ "$STATUS" = "exited" ] || [ "$STATUS" = "dead" ]; then
        fail_service_startup "$SERVICE_NAME"
      fi
    fi
    sleep 5
  done
}

wait_for_apiserver() {
  until curl --fail --silent http://localhost:8080/api/v2/monitor/health > /dev/null; do
    sleep 5
  done
}

# run the Airflow stack used for remote development
su orchestration -c "
  cd /opt/pipeline/orchestration &&
  docker compose up -d --build ${REMOTE_AIRFLOW_SERVICES}
"
wait_for_airflow_init
wait_for_healthy_service postgres
wait_for_healthy_service airflow-scheduler
wait_for_healthy_service airflow-dag-processor
wait_for_healthy_service airflow-triggerer
wait_for_healthy_service airflow-apiserver
wait_for_apiserver

# signal that the script is done
touch /ready
