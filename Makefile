### HOUSEKEEPING TARGETS ###
COMPONENTS := pis pts orchestration croissant

.PHONY: help clean test dev


#: HOUSEKEEPING TARGETS ############################################################################
help:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-9s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

clean-%:
	@rm -rf $*/.venv $*/coverage.xml $*/.coverage $*/.pytest_cache $*/.ruff_cache
	@echo "cleaned $*"

clean: $(addprefix clean-,$(COMPONENTS))  ## Remove build/test artifacts from all components
	@rm -rf .git/hooks/pre-commit
	@echo "cleaned all components"
####################################################################################################


#: TEST TARGETS ####################################################################################
lint-%: %/pyproject.toml
	@cd $* && uv run ruff check . && uv run ty check
	@echo "lint completed for $*"

lint: $(addprefix lint-,$(COMPONENTS))  ## Run linter for all components
	@echo "lint completed for all components"

test-%: dev-%
	@cd $* && uv run pytest -rxs
	@echo "tests completed for $*"

test: $(addprefix test-,$(COMPONENTS))  ## Run tests for all components
	@echo "tests completed for all components"
####################################################################################################


#: DEVELOPMENT TARGETS #############################################################################
.git/hooks/pre-commit:
	@ln -sf $(shell pwd)/pre-commit.githook .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo pre-commit hook installed

dev-%: %/pyproject.toml .git/hooks/pre-commit
	@cd $* && uv sync --all-groups --all-extras
	@echo "dev dependencies installed for $*"

dev: $(addprefix dev-,$(COMPONENTS))  ## Install dev dependencies for all components
	@echo "dev dependencies installed for all components"
####################################################################################################
