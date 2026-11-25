-- The credible_sets_by_study and credible_sets_by_variant blocks of credible_sets.sql, which studies.sql and
-- variants.sql joined on.
-- credibleSetIds for each study, joined onto studies by studies.sql (which drops this table).
CREATE TABLE IF NOT EXISTS credible_sets_by_study ENGINE = EmbeddedRocksDB () PRIMARY KEY studyId AS (
    SELECT
        studyId,
        groupArray (id) AS credibleSetIds
    FROM
        credible_sets_log
    GROUP BY
        studyId
);

-- credibleSetIds for each variant, joined onto variants by variants.sql (which drops this table).
-- (credibleSetId, variantId) is unique in credible_set_variant, so groupArray has no duplicates.
CREATE TABLE IF NOT EXISTS credible_sets_by_variant ENGINE = EmbeddedRocksDB () PRIMARY KEY variantId AS (
    SELECT
        variantId,
        groupArray (credibleSetId) AS credibleSetIds
    FROM
        credible_set_variant_log
    GROUP BY
        variantId
);
