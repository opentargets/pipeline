## pis

Tasks live in `src/pis/tasks/` and are resolved by name from `config.yaml`; to add
one, create the module, then wire it into `config.yaml`. `src/pis/validators/` holds
helpers that tasks import.

A config change that only uses existing tasks (`copy`, `explode`, …) runs on the
current image; one that needs new pis code needs a `pis_version` bump
(`orchestration/AGENTS.md` explains why).
