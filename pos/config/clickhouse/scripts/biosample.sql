-- biosample_mergetree feeds the credible-set read models: EmbeddedRocksDB scans as one stream.
-- credible_sets_build_cleanup drops it.
CREATE TABLE IF NOT EXISTS biosample_mergetree ENGINE = MergeTree ORDER BY tuple() AS (
  SELECT
    *
  FROM
    biosample_log
);

SELECT throwIf(count() != uniqExact(biosampleId), 'duplicate biosampleId in biosample_mergetree') FROM biosample_mergetree;

CREATE TABLE IF NOT EXISTS biosample ENGINE = EmbeddedRocksDB () PRIMARY KEY biosampleId AS
SELECT * FROM biosample_mergetree;

CREATE VIEW IF NOT EXISTS biosample_names_source AS
SELECT
  biosampleId AS id,
  biosampleName AS name
FROM
  biosample;

CREATE DICTIONARY IF NOT EXISTS biosample_names (id String, name String) PRIMARY KEY id SOURCE (CLICKHOUSE (TABLE 'biosample_names_source')) LAYOUT (COMPLEX_KEY_HASHED ()) LIFETIME (0);

DROP TABLE IF EXISTS biosample_log
SYNC;
