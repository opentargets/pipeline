-- Rebuild the chembl_<version>_molecule Elasticsearch document from the ChEMBL
-- relational schema.
--
-- Grain: one row per molecule_dictionary.molregno.
-- See docs/superpowers/specs/2026-08-10-chembl-postgres-drug-extraction-design.md
--
-- Note the two name differences against the Elasticsearch document: the
-- `molecule_structures` field comes from the `compound_structures` table, and
-- `molecule_synonym` comes from the `synonyms` column.
--
-- first_approval, max_phase, withdrawn_flag and black_box_warning are
-- deliberately absent: no pts module reads them.
WITH synonyms AS (
    SELECT ms.molregno,
           list(struct_pack(
               molecule_synonym := ms.synonyms,
               syn_type := ms.syn_type
           ) ORDER BY ms.molsyn_id) AS molecule_synonyms
    FROM molecule_synonyms ms
    GROUP BY ms.molregno
),
parent AS (
    SELECT mh.molregno,
           pmd.chembl_id AS parent_chembl_id
    FROM molecule_hierarchy mh
    JOIN molecule_dictionary pmd ON pmd.molregno = mh.parent_molregno
)
SELECT
    md.chembl_id AS molecule_chembl_id,
    md.pref_name,
    md.molecule_type,
    CASE WHEN cs.molregno IS NULL THEN NULL ELSE struct_pack(
        canonical_smiles := cs.canonical_smiles,
        standard_inchi_key := cs.standard_inchi_key,
        molfile := cs.molfile
    ) END AS molecule_structures,
    struct_pack(parent_chembl_id := p.parent_chembl_id) AS molecule_hierarchy,
    coalesce(
        sy.molecule_synonyms,
        []::STRUCT(molecule_synonym VARCHAR, syn_type VARCHAR)[]
    ) AS molecule_synonyms,
    -- always empty, and not an oversight. The Elasticsearch document's
    -- cross_references point at external drug registries, and what the dump
    -- lacks is the product-level mapping: ChEMBL holds one EMA compound_records
    -- row per molecule, where EMA issues one EPAR per marketed product, so
    -- TELMISARTAN's single record stands against eight EPARs. The identifiers
    -- themselves are often present as TRADE_NAME synonyms, but no rule selects
    -- the right ones -- the best measured against the 26.06 release reaches 44%
    -- precision, fabricating 1991 links to recover 1566. The field is kept,
    -- empty, because pts reads cross_references.xref_id and .xref_src.
    -- See the design doc's open question 1 for the full measurements.
    []::STRUCT(xref_id VARCHAR, xref_src VARCHAR)[] AS cross_references
FROM molecule_dictionary md
LEFT JOIN compound_structures cs ON cs.molregno = md.molregno
LEFT JOIN parent p ON p.molregno = md.molregno
LEFT JOIN synonyms sy ON sy.molregno = md.molregno
