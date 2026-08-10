-- Rebuild the chembl_<version>_mechanism Elasticsearch document from the
-- ChEMBL relational schema.
--
-- Grain: one row per drug_mechanism.mec_id.
-- See docs/superpowers/specs/2026-08-10-chembl-postgres-drug-extraction-design.md
WITH parent AS (
    SELECT mh.molregno,
           pmd.chembl_id AS parent_chembl_id
    FROM molecule_hierarchy mh
    JOIN molecule_dictionary pmd ON pmd.molregno = mh.parent_molregno
),
refs AS (
    SELECT mr.mec_id,
           list(struct_pack(
               ref_id := mr.ref_id,
               ref_type := mr.ref_type,
               ref_url := mr.ref_url
           ) ORDER BY mr.mecref_id) AS mechanism_refs
    FROM mechanism_refs mr
    GROUP BY mr.mec_id
)
SELECT
    dm.record_id,
    dm.mechanism_of_action,
    dm.action_type,
    md.chembl_id AS molecule_chembl_id,
    p.parent_chembl_id AS parent_molecule_chembl_id,
    td.chembl_id AS target_chembl_id,
    struct_pack(
        all_molecule_chembl_ids := list_distinct(
            list_filter([md.chembl_id, p.parent_chembl_id], x -> x IS NOT NULL)
        )
    ) AS _metadata,
    coalesce(
        r.mechanism_refs,
        []::STRUCT(ref_id VARCHAR, ref_type VARCHAR, ref_url VARCHAR)[]
    ) AS mechanism_refs
FROM drug_mechanism dm
LEFT JOIN molecule_dictionary md ON md.molregno = dm.molregno
LEFT JOIN parent p ON p.molregno = dm.molregno
LEFT JOIN target_dictionary td ON td.tid = dm.tid
LEFT JOIN refs r ON r.mec_id = dm.mec_id
