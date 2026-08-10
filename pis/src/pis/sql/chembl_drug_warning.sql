-- Rebuild the chembl_<version>_drug_warning Elasticsearch document from the
-- ChEMBL relational schema.
--
-- Grain: one row per drug_warning.warning_id.
-- See docs/superpowers/specs/2026-08-10-chembl-postgres-drug-extraction-design.md
WITH parent AS (
    SELECT mh.molregno,
           pmd.chembl_id AS parent_chembl_id
    FROM molecule_hierarchy mh
    JOIN molecule_dictionary pmd ON pmd.molregno = mh.parent_molregno
),
refs AS (
    SELECT wr.warning_id,
           list(struct_pack(
               ref_id := wr.ref_id,
               ref_type := wr.ref_type,
               ref_url := wr.ref_url
           ) ORDER BY wr.warnref_id) AS warning_refs
    FROM warning_refs wr
    GROUP BY wr.warning_id
)
SELECT
    dw.warning_id,
    dw.warning_type,
    dw.warning_class,
    dw.warning_country,
    dw.warning_description,
    dw.warning_year,
    dw.efo_id,
    dw.efo_term,
    dw.efo_id_for_warning_class,
    md.chembl_id AS molecule_chembl_id,
    p.parent_chembl_id AS parent_molecule_chembl_id,
    struct_pack(
        all_molecule_chembl_ids := list_distinct(
            list_filter([md.chembl_id, p.parent_chembl_id], x -> x IS NOT NULL)
        )
    ) AS _metadata,
    coalesce(
        r.warning_refs,
        []::STRUCT(ref_id VARCHAR, ref_type VARCHAR, ref_url VARCHAR)[]
    ) AS warning_refs
FROM drug_warning dw
LEFT JOIN molecule_dictionary md ON md.molregno = dw.molregno
LEFT JOIN parent p ON p.molregno = dw.molregno
LEFT JOIN refs r ON r.warning_id = dw.warning_id
