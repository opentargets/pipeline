-- One row per pharmacogenomics record, keyed by rowId: pts' pharmacogenomicsId.
-- The targets, variants and drug tables each hold a pharmacogenomicsIds array of these ids.
-- NB: the load task splits on semicolons, so do not use them inside comments or strings.
SELECT throwIf(count() != uniqExact(pharmacogenomicsId), 'duplicate pharmacogenomicsId in pharmacogenomics_log') FROM pharmacogenomics_log;

CREATE TABLE IF NOT EXISTS pharmacogenomics ENGINE = EmbeddedRocksDB () PRIMARY KEY rowId AS (
    SELECT
        pharmacogenomicsId AS rowId,
        * EXCEPT (pharmacogenomicsId)
    FROM pharmacogenomics_log
);

DROP TABLE IF EXISTS pharmacogenomics_log SYNC;
