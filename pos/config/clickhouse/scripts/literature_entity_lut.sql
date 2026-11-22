CREATE TABLE IF NOT EXISTS literature_entity_lut ENGINE = MergeTree ()
ORDER BY (
        keywordId, year, month, relevance
    ) AS (
        SELECT -- a sort key cannot be Nullable
            pmid, pmcid, keywordId, relevance, date, ifNull(year, 0) AS year, ifNull(month, 0) AS month
        FROM literature_entity_lut_log
    );

OPTIMIZE TABLE literature_entity_lut FINAL;

DROP TABLE literature_entity_lut_log SYNC;
