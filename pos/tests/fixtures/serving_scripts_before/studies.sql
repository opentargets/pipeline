-- studies_mergetree feeds the credible-set read models: EmbeddedRocksDB scans as one stream.
-- credible_sets_build_cleanup drops it.
CREATE TABLE IF NOT EXISTS studies_mergetree ENGINE = MergeTree ORDER BY tuple() AS (
    select *
    from
        studies_log
        left outer join credible_sets_by_study on studies_log.studyId = credible_sets_by_study.studyId
);

SELECT throwIf(count() != uniqExact(studyId), 'duplicate studyId in studies_mergetree') FROM studies_mergetree;

CREATE TABLE IF NOT EXISTS studies ENGINE = EmbeddedRocksDB () PRIMARY KEY studyId AS
SELECT * FROM studies_mergetree;

DROP TABLE IF EXISTS credible_sets_by_study SYNC;

DROP TABLE IF EXISTS studies_log SYNC;
