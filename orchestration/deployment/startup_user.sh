#!/bin/bash

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# shellcheck disable=SC1091
source "$HOME/.local/bin/env"

# install package dependencies for ide
cd /opt/orchestration && uv sync --all-groups --all-extras --dev
sudo chgrp -R google-sudoers .venv

# enable git for users. start.sh pipes this script over ssh on every run, and --add would
# append a duplicate entry to ~/.gitconfig each time
git config --global --replace-all safe.directory /opt/pipeline
