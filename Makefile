### HOUSEKEEPING TARGETS ###
.PHONY: help clean test dev coverage run

help:  ## Show the help message
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-9s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

clean:  ## Clean up
	@rm -rf .venv build dist pis.egg-info coverage.xml .coverage .pytest_cache .ruff_cache .git/hooks/pre-commit


### DEVELOPMENT TARGETS ###
.git/hooks/pre-commit:
	@ln -sf $(shell pwd)/pre-commit.githook .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo pre-commit hook installed

%/.venv: %/pyproject.toml
	@cd $* && uv sync --all-extras --quiet
	@touch $@
	@echo "dev dependencies installed for $*"

dev: pis/.venv pts/.venv orchestration/.venv croissant/.venv .git/hooks/pre-commit  ## Install dev dependencies for all components

test: .venv/bin/pytest  ## Run the tests
	@uv run pytest

coverage: .venv/bin/pytest  ## Generate and show coverage reports
	@uv run coverage run -m pytest -qq && uv run coverage xml && uv run coverage report -m

### MAIN TARGETS ###
run: ## Runs the step specified by `step` argument
	@[ -n "$(step)" ] && uv run pis -s $(step) || uv run pis -h
