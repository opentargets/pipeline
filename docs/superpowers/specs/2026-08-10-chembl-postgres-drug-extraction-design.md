# Replace ChEMBL Elasticsearch extraction with PostgreSQL

**Date:** 2026-08-10
**Status:** approved, ready for implementation planning
**Stacks on:** `il-4458` (PR #17, `feat(pis): add postgres_export task`)

## Problem

PIS currently pulls four ChEMBL datasets out of the public ChEMBL Elasticsearch
cluster at `https://www.ebi.ac.uk/chembl/elk/es`:

| Output | Index |
| --- | --- |
| `input/drug/chembl_drug_warning.jsonl` | `chembl_${chembl_version}_drug_warning` |
| `input/drug/chembl_mechanism.jsonl` | `chembl_${chembl_version}_mechanism` |
| `input/drug/chembl_molecule.jsonl` | `chembl_${chembl_version}_molecule` |
| `input/drug/chembl_target.jsonl` | `chembl_${chembl_version}_target` |

There are five `elasticsearch` tasks over these four indexes: `chembl_target` is
queried twice with identical fields, landing at both
`input/drug/chembl_target.jsonl` (`pis/config.yaml:307`) and
`input/target/chembl/chembl_target.jsonl` (`pis/config.yaml:952`).

Depending on a third party's search cluster puts its uptime and its index
lifecycle on the release critical path, and the documents it returns are a
denormalised view we do not control. PR #17 already restores the ChEMBL
PostgreSQL dump — a dated, immutable artifact — into an ephemeral server. The
same restore can produce these four datasets.

## Goal

Produce the four datasets as parquet, from the ChEMBL PostgreSQL dump, such that
the existing PTS consumers work unchanged apart from the file format.

## Decisions

### Equivalence bar: same names, prune dead branches

Field names and nesting are preserved exactly for everything PTS reads,
including the `_metadata.*` structs. Branches with no consumer are dropped.

This keeps the PTS diff to a format change, and avoids reverse-engineering large
ES-only structures (`_metadata.target_component`, `_metadata.es_completion`,
`_metadata.related_*`, `_metadata.source`, `_metadata.organism_taxonomy`,
`_metadata.generated_resources`) that nothing validates.

The exhaustive per-column prune list, measured against the 26.06 release rather
than asserted, is in "Verification" below.

Dropped as provably dead — requested from ES today, read by no PTS module:
`first_approval`, `max_phase`, `withdrawn_flag`, `black_box_warning` on molecule.
(`drug_molecule.py:143` recomputes `max_phase` from `clinical_report`.)

### Cutover: replace outright

The PR removes the five `elasticsearch` tasks and wires the PostgreSQL path in.
Equivalence is proven during development against the 26.06 release files with a
throwaway comparison script; the evidence goes in the PR description, not into
the pipeline. The 26.06 files remain in GCS if the comparison ever needs redoing.

The `elasticsearch` task type and its dependency stay in the tree. Nothing else
uses them, but removing the task type is a separable cleanup that would make this
PR harder to review.

### Architecture: extend `postgres_export` with `queries:`

The restore is by far the expensive part of `postgres_export`, which is why the
task exports many tables from one restore. These four datasets come from the same
ChEMBL dump that PR #17 already restores, so they must not trigger a second
restore. That rules out a separate sibling task.

SQL lives in files under `pis/src/pis/sql/`, not inline in `config.yaml`. These
are 40-60 line queries with nested struct construction; `config.yaml` is already
~1000 lines, and the SQL is the artifact a ChEMBL-literate reviewer needs to
check.

## Design

### Spec changes

```python
class QuerySpec(BaseModel):
    query: str
    """Names the SQL file at pis/src/pis/sql/<query>.sql."""
    destination: str
    """Path for the parquet file, relative to the release root."""
    requires_tables: list[str]
    """Tables pg_restore must load for this query to run."""
```

`PostgresExportSpec` gains `queries: list[QuerySpec] = []`. At least one of
`tables` or `queries` must be non-empty (replacing the current `min_length=1` on
`tables`).

### Task changes

Two changes inside `PostgresExport`:

1. The restore table list becomes the union of `t.table for t in spec.tables` and
   every `q.requires_tables`. The restore stays selective, so ChEMBL's large
   tables (`activities` and friends) are never loaded.
2. After the existing per-table export loop, a second loop reads each `.sql`
   file and wraps it in the same
   `COPY (…) TO … (FORMAT parquet, COMPRESSION zstd)` used by `_build_copy_sql`,
   recording row counts identically.

`validate()` iterates both lists. Staging, extraction, the archive version check,
tuning, server lifecycle and artifact reporting are untouched.

The `SELECT DISTINCT` that `_build_copy_sql` applies to table exports is **not**
applied to query exports — the queries control their own grain.

### Why DuckDB executes the SQL

The queries run through DuckDB against the attached PostgreSQL server, not
through `psql`. The outputs are nested, and DuckDB has native `STRUCT`/`LIST`
that serialise straight to parquet: `list(struct_pack(...))` yields
`molecule_synonyms[{molecule_synonym, syn_type}]` directly. In PostgreSQL the
same shapes would mean building JSON and re-parsing it.

The `l1`…`l6` ancestor flattening for protein classification needs
`WITH RECURSIVE`, which DuckDB supports. It runs client-side over data pulled
from the attached server; `protein_classification` is a few thousand rows, so
this is not a concern.

### Configuration

```yaml
- name: postgres_export chembl tables
  requires:
    - copy chembl dump
  source: input/drug/chembl_${chembl_version}.tar.gz
  tables:
    # unchanged from PR #17
  queries:
    - query: chembl_molecule
      destination: input/drug/chembl_molecule.parquet
      requires_tables:
        - molecule_dictionary
        - compound_structures
        - molecule_hierarchy
        - molecule_synonyms
    - query: chembl_mechanism
      destination: input/drug/chembl_mechanism.parquet
      requires_tables:
        - drug_mechanism
        - mechanism_refs
        - molecule_dictionary
        - molecule_hierarchy
        - target_dictionary
    - query: chembl_target
      destination: input/drug/chembl_target.parquet
      requires_tables:
        - target_dictionary
        - target_components
        - component_sequences
        - component_class
        - protein_classification
    - query: chembl_drug_warning
      destination: input/drug/chembl_drug_warning.parquet
      requires_tables:
        - drug_warning
        - warning_refs
        - molecule_dictionary
        - molecule_hierarchy
```

`chembl_target` is produced once. The `target` step's PTS source is repointed
from `input/target/chembl/chembl_target.jsonl` to
`input/drug/chembl_target.parquet` rather than keeping a second identical copy.

## Schema mapping

Column names below are verified against ChEMBL 37
`schema_documentation.txt` (`ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/`).

### `chembl_molecule`

Base `molecule_dictionary` (`md`), one row per `molregno`.

| Output field | Source |
| --- | --- |
| `molecule_chembl_id` | `md.chembl_id` |
| `pref_name` | `trim(md.pref_name)` — 18 ChEMBL 37 values carry a trailing space that ChEMBL's own indexer trims |
| `molecule_type` | `md.molecule_type` |
| `molecule_structures.canonical_smiles` | `compound_structures.canonical_smiles` |
| `molecule_structures.standard_inchi_key` | `compound_structures.standard_inchi_key` |
| `molecule_structures.molfile` | `rtrim(compound_structures.molfile, chr(10)) \|\| chr(10)` — normalises the molblock terminator, since `compound_structures` is inconsistent about a trailing newline after `M  END` and pts's truncation regex requires it |
| `molecule_hierarchy.parent_chembl_id` | `md.chembl_id` via `molecule_hierarchy.parent_molregno` |
| `molecule_synonyms[].molecule_synonym` | `molecule_synonyms.synonyms` |
| `molecule_synonyms[].syn_type` | `molecule_synonyms.syn_type` |
| `cross_references[].xref_id` | not reconstructible; emitted empty (see open question 1) |
| `cross_references[].xref_src` | not reconstructible; emitted empty (see open question 1) |

Two name differences to note: the ES field `molecule_structures` comes from the
table **`compound_structures`**, and `molecule_synonym` comes from the column
**`synonyms`**.

Pruned as unread: `molecule_structures.standard_inchi` (only `standard_inchi_key`
is used, at `chembl_molecule.py:161`) and `molecule_hierarchy.active_chembl_id`
and `.molecule_chembl_id` (only `parent_chembl_id` is used, at
`chembl_molecule.py:174`). Both structs are still emitted under their ES names so
the existing field paths resolve.

### `chembl_mechanism`

Base `drug_mechanism` (`dm`), one row per `mec_id`.

| Output field | Source |
| --- | --- |
| `molecule_chembl_id` | `molecule_dictionary.chembl_id` via `dm.molregno` |
| `parent_molecule_chembl_id` | via `molecule_hierarchy.parent_molregno` |
| `_metadata.all_molecule_chembl_ids` | `list_distinct` of the two above, nulls removed |
| `target_chembl_id` | `target_dictionary.chembl_id` via `dm.tid` |
| `action_type` | `dm.action_type` |
| `mechanism_of_action` | `dm.mechanism_of_action` |
| `record_id` | `dm.record_id` |
| `mechanism_refs[]` | `mechanism_refs` on `mec_id` → `{ref_id, ref_type, ref_url}` |

Pruned as unread: `_metadata.parent_molecule_chembl_id`. The Elasticsearch
document carries the parent id twice — once at the top level and once inside
`_metadata` — and `drug_mechanism_of_action.py:67-68` takes only
`_metadata.all_molecule_chembl_ids` before dropping `_metadata` entirely. The
top-level `parent_molecule_chembl_id` is kept, so nothing is lost.

### `chembl_target`

Base `target_dictionary` (`td`), one row per `tid`.

| Output field | Source |
| --- | --- |
| `target_chembl_id` | `td.chembl_id` |
| `pref_name` | `td.pref_name` |
| `target_type` | `td.target_type` |
| `target_components[].accession` | `component_sequences.accession` via `target_components` |
| `_metadata.protein_classification[]` | `component_class` → `protein_classification`, `{protein_class_id, l1…l6}` |

`target_components` is emitted as `array<struct<accession>>`. That is all PTS
reads, and it preserves both `target_components.accession` and the
`size(target_components) == 1` filter in `target.py:1146`.

`_metadata.protein_classification` must stay positionally aligned with
`target_components`, because `target.py:1150` zips the two arrays. Both are built
from the same ordered component list, one classification entry per component,
which makes the alignment correct by construction. Sampling the 26.06 file
confirms the ES documents are 1:1 — observed `(0,0)`, `(1,1)`, `(2,2)` for
`(len(target_components), len(protein_classification))`.

`l1`…`l6` come from walking `protein_classification.parent_id` to the root with
`WITH RECURSIVE` and placing each ancestor's `pref_name` at its `class_level`.

Each classification entry carries only `protein_class_id` and `l1`…`l6`. The
`l*_desc`, `l*_short` and `l*_definition` variants present in the ES document are
pruned — `target.py:1162` reads only `protein_class_id` and the bare `l1`…`l6`
labels.

### `chembl_drug_warning`

Base `drug_warning` (`dw`), one row per `warning_id`.

| Output field | Source |
| --- | --- |
| `warning_id`, `warning_type`, `warning_class`, `warning_country`, `warning_description`, `warning_year` | `dw.*` |
| `efo_id`, `efo_term`, `efo_id_for_warning_class` | `dw.*` |
| `molecule_chembl_id` | `molecule_dictionary.chembl_id` via `dw.molregno` |
| `parent_molecule_chembl_id` | via `molecule_hierarchy.parent_molregno` |
| `_metadata.all_molecule_chembl_ids` | `list_distinct` of the two above, nulls removed |
| `warning_refs[]` | `warning_refs` on `warning_id` → `{ref_id, ref_type, ref_url}` |

### `_metadata.all_molecule_chembl_ids`

Verified empirically across 1,330 sampled documents from the 26.06 release (521
warning, 809 mechanism): this field is exactly the deduplicated set
`{molecule_chembl_id, parent_molecule_chembl_id}` — size 1 when they coincide,
size 2 when they differ, never wider. The implementation must re-confirm this
across the full files before relying on it.

## Open questions

Both are resolved by the comparison script, not by guessing.

1. **`cross_references` on molecule.** **Resolved: it cannot be rebuilt from the
   dump, and is emitted as an empty array.**

   The `compound_records` hypothesis is wrong: not one of the 6,011 `xref_id`
   values equals the `compound_records.src_compound_id` of its own molecule for
   its own source, and `DailyMed` is not a row of `source` at all.

   **Two tempting justifications for the conclusion are false. Do not repeat
   them.** The identifiers *are* in the dump: `fampyra`, `fampridine-accord`,
   `oxyglobin` and `inbrija` are all `molecule_synonyms` rows of the right
   molecule with `syn_type = 'TRADE_NAME'`. And registration *is* recorded:
   `compound_records` joined to `source` on `src_short_name = 'EMA'` identifies
   EMA-registered molecules almost exactly — 1,046 such molecules against 1,045
   with an EPAR link, 100% of linked molecules having a record.

   What is genuinely absent is the **product-level mapping**. ChEMBL holds one
   EMA record per molecule; EMA issues one EPAR per marketed product.
   TELMISARTAN has a single record (`compound_key='TELMISARTAN'`,
   `src_compound_id='3472'`) against eight EPARs — `micardis`, `pritor`,
   `tolura`, `telmisartan-teva`, `veterinary/EPAR/semintra` and others. Only 767
   of 1,045 molecules have matching counts, CHEMBL599 being 19 EPARs to 1 record,
   and `compound_key` never equals an EPAR slug (0 of 1,788). Nothing records the
   human-versus-veterinary branch either.

   Measured reconstruction quality, which is what actually decides this:

   | candidate rule | precision | recall |
   | --- | --- | --- |
   | every `molecule_synonyms` value, all four sources | 0.89% | 3,481/6,011 |
   | every `products.trade_name`, EMA only | 3.4% | 361/1,788 |
   | EMA-recorded molecules × their `TRADE_NAME` synonyms | **44%** | 1,566/1,788 |

   The best rule fabricates 1,991 links to recover 1,566 real ones. Per-source
   recall from synonyms is DailyMed 100%, EMA 88%, USAN 24%, INN 0% — INN's
   `xref_id` is a WHO list number, not a name. No rule reaches a precision worth
   shipping, and the project's standard is that no record may be fabricated.

   `chembl_molecule.sql` therefore emits
   `[]::STRUCT(xref_id VARCHAR, xref_src VARCHAR)[]`. The field is kept rather
   than pruned because `chembl_molecule.py:294` reads `cross_references.xref_id`
   and `.xref_src`; an empty array keeps PTS working unchanged and leaves one
   line to change if ChEMBL ever ships the data. The DrugBank cross-references
   PTS merges in come from a separate input and are unaffected.

   **This is a backwards-incompatible change and must be called out in the
   release notes.** 4,246 molecules lose their ChEMBL-side `crossReferences`,
   6,011 entries in total. 2,078 of those molecules are approved drugs
   (`max_phase = 4`) — LEVODOPA, TELMISARTAN, PROGESTERONE and DICLOFENAC among
   them — so the loss is visible on Platform drug pages, not confined to obscure
   compounds. Decision taken 2026-08-11: accepted in preference to keeping a
   narrow Elasticsearch fetch, which would leave the public ChEMBL cluster on the
   release critical path for one field.

   `compound_records` and `source` are consequently **not** in the query's
   `requires_tables`.
2. **Protein class selection.** **Resolved: there is no selection. The question
   was based on a false premise.**

   `_metadata.protein_classification` is **not** one entry per component. It is
   the concatenation of *every* `component_class` row of every component the
   target uses — no class is chosen and none is dropped. Proven by exact set
   equality over the whole 26.06 baseline: the 12,575 distinct
   `(component_id, protein_class_id)` pairs the baseline asserts are precisely
   `component_class` restricted to the 12,383 components those targets use —
   12,575 against 12,575, zero only in the baseline, zero only in the database.

   The earlier 1:1 impression came from sampling. Across all 17,284 component
   entries the class counts are 1 → 16,972, 2 → 295, 3 → 15, 5 → 2: the
   multi-class case is real but rare enough that a small sample misses it.

   Consequently `len(target_components) == len(_metadata.protein_classification)`
   is **false for 258 of 18,552 targets, and that is correct** — the baseline
   returns the same 258, and the two sides agree on both array lengths for every
   target. Enforcing 1:1 would have meant inventing a selection rule and
   discarding 605 real classification entries.

   **The rule:** keep every `component_class` row, grouped by component in
   `component_id` order, ordered within a component by ascending
   `protein_class_id`.

   **Ordering correction.** `target_components` follows **`component_id`, not
   `targcomp_id`**. The two disagree for 1,108 targets, and the baseline follows
   `component_id` for all 18,552. This was invisible to the comparison harness,
   which sorts lists before comparing, and was caught only by checking order
   explicitly.

   **Known downstream wrinkle, pre-existing and unchanged.**
   `pts/src/pts/pyspark/target.py:1150` zips `_metadata.protein_classification`
   with `target_components.accession`, having filtered to single-component
   targets at `:1146`. For the 162 single-component targets that carry more than
   one class, `arrays_zip` pads the one-element accession array with nulls, so
   the second and later classes pair with a null accession and are dropped. That
   is exactly what happens with the Elasticsearch input today — the rebuild
   reproduces it unchanged — so it is a separate follow-up, not something to fix
   under an equivalence change.

## Verification

A throwaway script, not shipped in the pipeline. For each of the four outputs it
compares the new parquet against the 26.06 JSONL:

- row counts;
- the full sorted key set per document, to catch a missing or misnamed field;
- a full-key join with a field-by-field diff, reporting mismatch counts and a
  sample of differing values per field.

Join keys: `molecule_chembl_id` for molecule, `warning_id` for drug warning,
`target_chembl_id` for target, and `(record_id, molecule_chembl_id,
target_chembl_id, mechanism_of_action)` for mechanism, which has no single
natural key in the ES document.

This script is also what answers the two open questions: write the query, diff,
adjust until the diff is empty. Results go in the PR description.

### Measured result

All four datasets were exported from one restore of the real ChEMBL 37 dump and
compared against the shipped 26.06 release. **Row counts and key sets are
identical on all four.** Of 35 leaf columns compared, 33 are identical and 2
differ, both for known reasons.

| Dataset | Rows | Columns identical | Columns differing | Columns pruned |
| --- | --- | --- | --- | --- |
| `chembl_drug_warning` | 2,304 | 13 of 13 | — | 0 |
| `chembl_mechanism` | 7,561 | 8 of 8 | — | 1 |
| `chembl_target` | 18,552 | 5 of 5 | — | 20 |
| `chembl_molecule` | 2,921,148 | 7 of 9 | 2 | 7 |

The two differing columns, both in `chembl_molecule`:

- `molecule_structures.molfile`, 2,897,819 rows — differs **by construction**.
  The Elasticsearch value is a full SD-file record; the relational column holds
  only the molblock. After `chembl_molecule.py:166`'s truncation regex is applied
  to both sides, zero of 2,921,148 rows differ.
- `cross_references`, 4,246 rows — the accepted, documented loss above.

The 28 pruned columns are exactly those no PTS module reads:
`_metadata.parent_molecule_chembl_id` on mechanism; the 20 ES-only `_metadata`
structures on target; and on molecule `molecule_structures.standard_inchi`,
`molecule_hierarchy.{active_chembl_id,molecule_chembl_id}`, `first_approval`,
`max_phase`, `withdrawn_flag`, `black_box_warning`.

One caveat on that prune list: the three small baselines were read whole, but
`chembl_molecule.jsonl` is 9.7 GiB, so its Elasticsearch schema was inferred
from a 400,000-row sample. A field null throughout that sample could be missing
from the list. The *compared* columns are unaffected — those come from the
parquet schema and were checked across all 2,921,148 rows.

## Testing

Unit tests in the style of the existing `TestRoundTrip`, against a small fixture
database built from hand-written ChEMBL rows covering the shapes that matter:

- a molecule with a parent and one without, to exercise
  `all_molecule_chembl_ids` at both sizes;
- a multi-component target and a single-component target, to exercise the
  positional alignment;
- a component with several protein classes, to pin the selection rule;
- a warning with no refs, and a mechanism with no refs, to check the arrays come
  out empty rather than null;
- a molecule with no `compound_structures` row.

Plus a spec-level test that `requires_tables` widens the restore list, since that
is the new failure mode introduced by this change.

## Error handling

- A missing or unreadable `.sql` file fails at spec validation, before the
  restore starts.
- A query referencing a table absent from `requires_tables` fails in DuckDB with
  a "table not found"; catch and re-raise with the query name attached, since the
  raw error does not say which query was at fault.
- Row counts and the empty-export guard behave as for table exports.

## Dependency on PR #17

The four blocking defects found in the PR #17 review apply to query exports as
much as to table exports, and should land first:

1. `rmtree(self.scratch)` at the top of `run()` — a killed run otherwise leaves
   `pgdata` behind and the retry doubles the data.
2. `--strict-names` on the data pass, and failing on a 0-row export.
3. `--schema <schema_name>` on the data pass.
4. Pre-create the `pgserver` user in the Dockerfile, or serialise the two
   `postgres_export` tasks.

Item 2 matters more here than for table exports: a query silently returning zero
rows because a table did not restore would ship an empty `chembl_molecule.parquet`
into the release.

## Downstream changes

| File | Change |
| --- | --- |
| `pts/src/pts/pyspark/drug_warning.py:32` | `format='json'` → parquet |
| `pts/src/pts/pyspark/drug_mechanism_of_action.py:36-37` | both loads → parquet |
| `pts/src/pts/pyspark/chembl_molecule.py` | molecule load → parquet |
| `pts/src/pts/pyspark/target.py:146` | `spark.read.json` → `spark.read.parquet` |
| `pts/config.yaml` | four `input/drug/*.jsonl` sources → `.parquet`; `target` step source repointed |
| `orchestration/src/orchestration/dags/config/unified_pipeline.yaml` | `pts_drug_warning`, `pts_chembl_molecule`, `pts_drug_mechanism_of_action`, and `pts_target` each gain `pis_clinical_report` in `depends_on` |

No logic changes, because column names and nesting are preserved.

The four `chembl_*` parquet files moved from the `drug` PIS step to the
`clinical_report` PIS step (`postgres_export chembl tables`), so the PTS steps
that read them need an explicit dependency edge on `pis_clinical_report`; their
existing `pis_drug` edge no longer covers the files they actually read.

## Out of scope

- Removing the `elasticsearch` task type and dependency from PIS.
- Any change to what PTS computes from these datasets.
- The `chembl_curation`, `drugbank` and other non-ChEMBL inputs to the drug step.
