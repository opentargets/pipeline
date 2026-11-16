#!/bin/bash

# enable git for users. start.sh pipes this script over ssh on every run, and --add would
# append a duplicate entry to ~/.gitconfig each time
git config --global --replace-all safe.directory /opt/pipeline
