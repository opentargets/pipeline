PKG  := $(notdir $(CURDIR))
ROOT := $(shell git rev-parse --show-toplevel)

.PHONY: build
build: ## Build the package: create a tag and push to trigger CI/CD in GitHub
	@tag="$$(cd '$(ROOT)' && uv run ./scripts/tag.py '$(PKG)')" || exit 1; \
	[ -n "$$tag" ] || { echo 'no tag produced'; exit 1; }; \
	printf 'This will build `%s`, are you sure? [y/N] ' "$$tag"; \
	read ans; \
	[ "$$ans" = y ] || [ "$$ans" = Y ] || { echo aborted; exit 1; }; \
	cd '$(ROOT)' && \
		git tag "$$tag" && \
		git push origin "$$tag"; \
	repo_name="$$(git config --get remote.origin.url | sed -E 's/.*github.com[:\/](.*)\.git/\1/')"; \
	echo "tag $$tag pushed, head to https://github.com/$$repo_name/actions/workflows/tag.yaml to see the build progress"

pr: ## Prepare for opening a PR: bump to the next pre-release version and commit
	@version="$$(cd '$(ROOT)' && uv run ./scripts/pr.py '$(PKG)')" || exit 1; \
	[ -n "$$version" ] || { echo 'no version produced'; exit 1; }; \
	printf 'This will bump `%s` to version `%s`, commit and push it, are you sure? [y/N] ' "$(PKG)" "$$version"; \
	read ans; \
	[ "$$ans" = y ] || [ "$$ans" = Y ] || { echo aborted; exit 1; }; \
	cd '$(ROOT)' && \
		uv --directory '$(PKG)' version "$$version" && \
		uv --directory '$(PKG)' sync --frozen --all-groups --all-extras && \
		git add '$(PKG)/pyproject.toml' '$(PKG)/uv.lock' && \
		git commit -m "bump '$(PKG)' to $$version" && \
		git push; \
	repo_name="$$(git config --get remote.origin.url | sed -E 's/.*github.com[:\/](.*)\.git/\1/')"; \
	echo "bump pushed, head to https://github.com/$$repo_name/pull/new/$(shell git rev-parse --abbrev-ref HEAD) to open a PR"
