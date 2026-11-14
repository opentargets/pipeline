---
name: port-pyspark-to-polars
description: Port a pts pyspark job to a polars transformer, or find why a ported transformer's output diverges from the spark output it replaces.
---

# Porting a pyspark step to polars

## Reuse the parity helpers first

`pts/src/pts/transformers/utils/` already reproduces spark semantics where polars
differs. Check these before writing an equivalent, and add new ones there:

- `spark_math`: `divide`, `round_half_up`, `spark_log10`, `spark_pow`, `spark_exp`,
  `pvalue_exponent`/`pvalue_mantissa`, `normalise_mantissa`, `substitute_zero_pvalues`
- `spark_csv`: `read_spark_csv`, `spark_csv_int`, `spark_csv_double` (spark's CSV
  reader and its numeric parsing)
- `spark_casts`: `to_long`, `to_int`, `to_float`, `cast_double` (non-ANSI casts)
- `spark_strings`: `spark_trim`, `spark_upper`, `java_double_to_string`, `regexp_extract`
- `spark_frames`: `spark_distinct`
- `lists`: `explode` (drops null and empty lists), `collect_set` (distinct, sorted)

## Divergences with no helper

Each of these has already produced a real defect, most of them silently:

- **Ordering.** `collect_set`/`collect_list` and set-backed traversals emit an
  arbitrary order that spark happened to keep stable. Sort explicitly, or order
  the frame before the aggregation and use `maintain_order`. Polars 2 left joins
  don't keep the left row order; pass `maintain_order='left'`.
- **Boolean casts.** Spark's `cast(string as boolean)` accepts `t/true/y/yes/1` and
  `f/false/n/no/0` case-insensitively and yields null otherwise; polars refuses the
  cast. Check every cast against spark.
- **Regex.** Rust's `\s` is unicode-aware, Java's is ASCII-only. Rust has no
  lookarounds. In spark's `regexp_replace`, `$1` is a group reference and `\1` a
  literal `1`; check which behaviour the data relies on.
- **Concat.** Spark's `concat` returns null if any argument is null; polars doesn't.
- **Division.** Polars 2 turns `x / <literal>` into `x * (1/<literal>)`, off by one
  ULP for many values. Use `spark_math.divide`.
- **Sums.** The optimizer drops a `sort()` before `sum()`, so an order-dependent float
  sum differs from spark's. Use `.sort().cum_sum().last()`.
- **Struct casts.** A struct cast matches fields by name, so casting to renamed fields
  gives all-null structs. Use `struct.rename_fields`.
- **Chunks.** A `when/then` whose condition reads a struct field panics on a
  multi-chunk frame, even when the output is scalar. `rechunk()` first. Joins over
  ~500 rows produce multi-chunk output by themselves, so small fixtures never
  reproduce the panic.
- **UDF copies.** `round_half_up` and `java_double_to_string` reference their input
  many times, and a `map_batches` UDF inside that input is copied per reference.
  Pass them a column or a materialised series.

## Verify

1. Re-run the *old* spark implementation on a real release's inputs, and confirm it
   reproduces the published output.
2. Run the port on the same inputs and diff it against that baseline, floats exactly.

Done when the diff is empty, or every remaining difference is explained and accepted
in the PR. Unit tests alone are not done.

Which steps run on Dataproc, and so when a port moves one off it, is the routing rule
in `orchestration/AGENTS.md`.
