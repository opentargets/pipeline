-- l2g_predictions_mergetree feeds the credible-set read models: EmbeddedRocksDB scans as one stream.
-- credible_sets_build_cleanup drops it.
CREATE TABLE IF NOT EXISTS l2g_predictions_mergetree ENGINE = MergeTree ORDER BY tuple() AS (
    SELECT credibleSetId, reverse(
            arraySort (
                l -> l.score, groupArray (
                    CAST(
                        (
                            credibleSetId, geneId, score, features, shapBaseValue
                        ), 'Tuple(credibleSetId String, geneId String, score Float64, features Array(Tuple(name LowCardinality(String), value Float64, shapValue Float64)), shapBaseValue Float64)'
                    )
                )
            )
        ) as l2g_predictions
    FROM l2g_predictions_log
    GROUP BY
        credibleSetId
);

SELECT throwIf(count() != uniqExact(credibleSetId), 'duplicate credibleSetId in l2g_predictions_mergetree') FROM l2g_predictions_mergetree;

CREATE TABLE IF NOT EXISTS l2g_predictions ENGINE = EmbeddedRocksDB () PRIMARY KEY credibleSetId AS
SELECT * FROM l2g_predictions_mergetree;

DROP TABLE IF EXISTS l2g_predictions_log SYNC;