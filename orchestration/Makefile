.DEFAULT_GOAL := cloud-dev
LOCAL_DEV_CREDENTIALS ?= ~/.config/gcloud/application_default_credentials.json

### HOUSEKEEPING TARGETS ###
.PHONY: help clean clean-vm \
        test lint check local-airflow cloud-airflow tunnel \
				upload-eqtl-catalogue-bucket-readme upload-ukb-ppp-bucket-readme \
				upload-finngen-bucket-readme upload-gwas-catalog-buckets-readme \
				upload-gnomad-bucket-readme upload-intervals-bucket-readme \
				upload-finngen-meta-readme

help: ## Show the help message
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-36s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

include ../common.mk

clean: ## Clean the project
	@AIRFLOW__API__SECRET_KEY=$$(openssl rand -hex 32) \
	AIRFLOW__API_AUTH__JWT_SECRET=$$(openssl rand -hex 32) \
	AIRFLOW__API_AUTH__JWT_ISSUER=$${AIRFLOW__API_AUTH__JWT_ISSUER:-airflow} \
	docker compose down --remove-orphans && \
	rm -rf logs dist .venv .pytest_cache .ruff_cache deployment/.terraform deployment/plan.out

clean-vm: ## Destroy the Airflow development VM
	@echo "Destroying Airflow development VM..."
	@terraform -chdir=./deployment init -reconfigure > /dev/null 2>&1 || (echo "Failed to initialize Terraform" && exit 1)
	@VM_OUTPUT=$$(terraform -chdir=./deployment output -raw up_airflow_dev_vm 2>&1); \
	if echo "$$VM_OUTPUT" | grep -q "up-airflow-dev-"; then \
		echo "Found VM: $$VM_OUTPUT"; \
		echo "Destroying VM..."; \
		terraform -chdir=./deployment destroy -auto-approve; \
	else \
		echo "No VM found or already destroyed."; \
	fi

### DEVELOPMENT TARGETS ###
sync:
	@uv sync --all-groups --all-extras

test: sync ## Run unit tests
	@uv run --frozen pytest -rxs

lint: sync ## Lint codebase
	@uv run --frozen ruff check . && uv run --frozen ty check

check: lint test ## run all checks

# compose autoloads .env; it is git-ignored so that the airflow secrets the cloud VM
# generates into it can never be committed. seed it from the tracked example.
.env:
	@cp .env.example .env

local-airflow: .env  ## Start a local AirFlow instance used for development
	@AIRFLOW__API__SECRET_KEY=$$(openssl rand -hex 32) \
	AIRFLOW__API_AUTH__JWT_SECRET=$$(openssl rand -hex 32) \
	AIRFLOW__API_AUTH__JWT_ISSUER=$${AIRFLOW__API_AUTH__JWT_ISSUER:-airflow} \
	GOOGLE_APPLICATION_CREDENTIALS=$(LOCAL_DEV_CREDENTIALS) docker compose -f compose.yaml -f compose.local.yaml up -d --build --remove-orphans

teardown-local-airflow:
	@AIRFLOW__API__SECRET_KEY=$$(openssl rand -hex 32) \
	AIRFLOW__API_AUTH__JWT_SECRET=$$(openssl rand -hex 32) \
	AIRFLOW__API_AUTH__JWT_ISSUER=$${AIRFLOW__API_AUTH__JWT_ISSUER:-airflow} \
	GOOGLE_APPLICATION_CREDENTIALS="" \
	docker-compose down

cloud-airflow: ## Start a remote AirFlow instance used for running the unified pipeline
	@./deployment/start.sh

tunnel: ## Tunnel to the remote AirFlow instance
	@./deployment/tunnel.sh


### OTHER TARGETS ###

build-dag-svgs: ## Generate visual representations of Airflow DAGs for documentation purposes
	@uv sync --all-groups
	@AIRFLOW__CORE__DAGS_FOLDER=src/orchestration/dags uv run airflow db migrate
	@AIRFLOW__CORE__DAGS_FOLDER=src/orchestration/dags uv run airflow dags reserialize
	@$(foreach dag, \
		datasources/gwas_catalog_data/gwas_catalog_sumstats_pics \
		datasources/gwas_catalog_data/gwas_catalog_sumstats_susie_clumping \
		datasources/gwas_catalog_data/gwas_catalog_sumstats_susie_finemapping \
		datasources/gwas_catalog_data/gwas_catalog_top_hits \
		datasources/gwas_catalog_data/gwas_catalog_heritability_estimate \
		datasources/gnomad_data/gnomad_ingestion \
		datasources/ukb_ppp_eur_data/ukb_ppp_eur_finemapping \
		datasources/ukb_ppp_eur_data/ukb_ppp_eur_harmonisation \
		datasources/finngen_meta_data/finngen_ukb_mvp_meta \
		datasources/finngen_data/finngen_ingestion \
		datasources/eqtl_catalogue_data/eqtl_catalogue_ingestion \
		credible_set_qc/credible_set_qc \
		datasources/lof_annotations/lof_curation_ingestion \
		datasources/foldx_annotations/foldx_ingestion \
		unified_pipeline/unified_pipeline, \
		AIRFLOW__CORE__LOAD_EXAMPLES=false AIRFLOW__CORE__DAGS_FOLDER=src/orchestration/dags uv run airflow dags show --save docs/$(dag).svg $(notdir $(dag));)

upload-eqtl-catalogue-bucket-readme: ## Upload eqtl_catalogue_data readme to the bucket
	@gcloud storage rsync docs/datasources/eqtl_catalogue_data gs://eqtl_catalogue_data/docs

upload-ukb-ppp-bucket-readme: ## Upload ukb_ppp_eur_data readme to the bucket
	@gcloud storage rsync docs/datasources/ukb_ppp_eur_data gs://ukb_ppp_eur_data/docs
	@gcloud storage rsync docs/credible_set_qc gs://ukb_ppp_eur_data/docs/credible_set_qc

upload-finngen-bucket-readme: ## Upload finngen_data readme to the bucket
	@gcloud storage rsync docs/datasources/finngen_data gs://finngen_data/docs

upload-gwas-catalog-buckets-readme: ## Upload gwas_catalog readme to the bucket(s)
	@gcloud storage rsync docs/datasources/gwas_catalog_data gs://gwas_catalog_inputs/docs
	@gcloud storage rsync docs/datasources/gwas_catalog_data gs://gwas_catalog_sumstats_pics/docs
	@gcloud storage rsync docs/datasources/gwas_catalog_data gs://gwas_catalog_sumstats_susie/docs
	@gcloud storage rsync docs/datasources/gwas_catalog_data gs://gwas_catalog_top_hits/docs
	@gcloud storage rsync docs/credible_set_qc gs://gwas_catalog_sumstats_susie/docs/credible_set_qc

upload-gnomad-bucket-readme: ## Upload gnomad_data readme to the bucket
	@gcloud storage rsync docs/datasources/gnomad_data gs://gnomad_data_2/docs

upload-intervals-bucket-readme: ## Upload intervals readme to the bucket
	@gcloud storage rsync docs/datasources/interval_data gs://interval_data/docs

upload-finngen-meta-readme: ## Upload finngen-meta readme to the bucket
	@gcloud storage rsync docs/datasources/finngen_meta_data gs://finngen_ukb_mvp_meta_data/docs

update-bucket-docs: upload-eqtl-catalogue-bucket-readme upload-ukb-ppp-bucket-readme \
									  upload-finngen-bucket-readme upload-gwas-catalog-buckets-readme \
										upload-gnomad-bucket-readme upload-intervals-bucket-readme \
										upload-finngen-meta-readme
