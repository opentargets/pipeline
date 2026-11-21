-- pharmacogenomics gives array(pharmacogenomicsIds) for each variant
-- variants_mergetree feeds the credible-set read models: EmbeddedRocksDB scans as one stream.
-- credible_sets_build_cleanup drops it.
CREATE TABLE IF NOT EXISTS variants_mergetree ENGINE = MergeTree ORDER BY tuple() AS (
    select *
    from variants_log
    left outer join credible_sets_by_variant using (variantId)
    left outer join (
        select
            assumeNotNull(variantId) as variantId,
            groupArray(rowId) as pharmacogenomicsIds
        from pharmacogenomics
        where variantId is not null
        group by variantId
    ) as pgx using (variantId)
);

SELECT throwIf(count() != uniqExact(variantId), 'duplicate variantId in variants_mergetree') FROM variants_mergetree;

CREATE TABLE IF NOT EXISTS variants ENGINE = EmbeddedRocksDB () PRIMARY KEY variantId AS
SELECT * FROM variants_mergetree;

DROP TABLE IF EXISTS credible_sets_by_variant SYNC;
DROP TABLE IF EXISTS variants_log SYNC;
