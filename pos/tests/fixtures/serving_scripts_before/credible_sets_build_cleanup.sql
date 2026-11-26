-- Drops the _mergetree inputs of the credible-set read models once all three are built,
-- so they are not in the release disk.
DROP TABLE IF EXISTS credible_sets_mergetree SYNC;
DROP TABLE IF EXISTS credible_sets_locus_mergetree SYNC;
DROP TABLE IF EXISTS studies_mergetree SYNC;
DROP TABLE IF EXISTS disease_mergetree SYNC;
DROP TABLE IF EXISTS l2g_predictions_mergetree SYNC;
DROP TABLE IF EXISTS targets_mergetree SYNC;
DROP TABLE IF EXISTS variants_mergetree SYNC;
DROP TABLE IF EXISTS biosample_mergetree SYNC;
