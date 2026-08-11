-- Rebuild the chembl_<version>_target Elasticsearch document from the ChEMBL
-- relational schema.
--
-- Grain: one row per target_dictionary.tid.
-- See docs/superpowers/specs/2026-08-10-chembl-postgres-drug-extraction-design.md
--
-- `_metadata.protein_classification` is NOT one entry per component. It is the
-- concatenation of every component_class row of every component, which is why
-- it can be longer than `target_components`: in ChEMBL 37, 162 single-component
-- targets carry between 2 and 5 classes. pts zips the two arrays
-- (pts/src/pts/pyspark/target.py), so the ordering below matters -- components
-- in the same order as `target_components`, classes within a component by
-- ascending protein_class_id, which is the order the Elasticsearch document
-- uses. See the design doc's open question 2.
WITH RECURSIVE ancestry AS (
    SELECT pc.protein_class_id AS leaf_id,
           pc.parent_id,
           pc.pref_name,
           pc.class_level
    FROM protein_classification pc
    UNION ALL
    SELECT a.leaf_id,
           pc.parent_id,
           pc.pref_name,
           pc.class_level
    FROM ancestry a
    JOIN protein_classification pc ON pc.protein_class_id = a.parent_id
),
levels AS (
    -- one row per class, holding its ancestors' names at their own level. The
    -- tree's root, protein_class_id 0, sits at class_level 0 and so falls out
    SELECT leaf_id,
           max(CASE WHEN class_level = 1 THEN pref_name END) AS l1,
           max(CASE WHEN class_level = 2 THEN pref_name END) AS l2,
           max(CASE WHEN class_level = 3 THEN pref_name END) AS l3,
           max(CASE WHEN class_level = 4 THEN pref_name END) AS l4,
           max(CASE WHEN class_level = 5 THEN pref_name END) AS l5,
           max(CASE WHEN class_level = 6 THEN pref_name END) AS l6
    FROM ancestry
    GROUP BY leaf_id
),
components AS (
    -- both lists below are built from this one, so they cannot disagree about
    -- which components a target has, or in what order
    SELECT tc.tid,
           tc.component_id,
           cs.accession
    FROM target_components tc
    LEFT JOIN component_sequences cs ON cs.component_id = tc.component_id
),
component_list AS (
    SELECT c.tid,
           list(struct_pack(accession := c.accession) ORDER BY c.component_id) AS target_components
    FROM components c
    GROUP BY c.tid
),
classification_list AS (
    -- an inner join, so a component with no class contributes nothing rather
    -- than an empty entry, exactly as the Elasticsearch document has it
    SELECT c.tid,
           list(struct_pack(
               protein_class_id := cc.protein_class_id::BIGINT,
               l1 := l.l1, l2 := l.l2, l3 := l.l3, l4 := l.l4, l5 := l.l5, l6 := l.l6
           ) ORDER BY c.component_id, cc.protein_class_id) AS protein_classification
    FROM components c
    JOIN component_class cc ON cc.component_id = c.component_id
    LEFT JOIN levels l ON l.leaf_id = cc.protein_class_id
    GROUP BY c.tid
)
SELECT
    td.chembl_id AS target_chembl_id,
    td.pref_name,
    td.target_type,
    coalesce(cl.target_components, []::STRUCT(accession VARCHAR)[]) AS target_components,
    struct_pack(
        protein_classification := coalesce(
            pl.protein_classification,
            []::STRUCT(
                protein_class_id BIGINT,
                l1 VARCHAR, l2 VARCHAR, l3 VARCHAR, l4 VARCHAR, l5 VARCHAR, l6 VARCHAR
            )[]
        )
    ) AS _metadata
FROM target_dictionary td
LEFT JOIN component_list cl ON cl.tid = td.tid
LEFT JOIN classification_list pl ON pl.tid = td.tid
