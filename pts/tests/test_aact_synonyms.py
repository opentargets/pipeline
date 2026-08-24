"""Tests for the AACT synonym-mining helpers in pts.transformers.utils.aact_synonyms.

These are a polars port of the ``pts.pyspark.drug_utils.aact_synonyms`` module, which
was deleted once nothing ran it; these tests are now the only ones on this logic. They
cover the private helper functions individually rather than through the two AACT
scenarios ported in
``test_chembl_molecule.py``. Not part of the fifteen ported chembl_molecule tests --
added because this ~500-line mining pipeline is new surface area that the ported tests
barely touch, and it is exactly the kind of translation where pyspark/polars null and
join semantics silently diverge.
"""

import polars as pl

from pts.transformers.utils.aact_synonyms import (
    _anchor_candidates,
    _apply_cleanup_rules,
    _build_chembl_indexes,
    _normalize_name,
    _rewrite_and_reclassify_codes,
    merge_aact_synonyms,
    mine_aact_synonyms,
    parse_aact_entries,
)

_ENTRIES_SCHEMA = {'nct_id': pl.Utf8, 'members': pl.List(pl.Utf8)}
_INDEX_SCHEMA = {'name_norm': pl.Utf8, 'ids': pl.List(pl.Utf8)}
_PC_SCHEMA = {'id': pl.Utf8, 'related': pl.List(pl.Utf8)}
_CAND_SCHEMA = {'id': pl.Utf8, 'candidate': pl.Utf8, 'nct_id': pl.Utf8, 'status': pl.Utf8}
_LABEL_SOURCE = pl.List(pl.Struct({'label': pl.Utf8, 'source': pl.Utf8}))


def _entries(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=_ENTRIES_SCHEMA) if rows else pl.DataFrame(schema=_ENTRIES_SCHEMA)


def _index(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=_INDEX_SCHEMA) if rows else pl.DataFrame(schema=_INDEX_SCHEMA)


def _pc(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=_PC_SCHEMA) if rows else pl.DataFrame(schema=_PC_SCHEMA)


def _cand(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=_CAND_SCHEMA) if rows else pl.DataFrame(schema=_CAND_SCHEMA)


class TestNormalizeName:
    def test_normalization(self):
        df = pl.DataFrame({'raw': ['  Revlimid®  ', 'G  CSF', 'Aspirin™']})
        out = dict(zip(df['raw'], df.with_columns(norm=_normalize_name(pl.col('raw')))['norm'], strict=True))
        assert out['  Revlimid®  '] == 'revlimid'
        assert out['G  CSF'] == 'g csf'
        assert out['Aspirin™'] == 'aspirin'

    def test_non_breaking_space_collapses_like_ascii_space(self):
        # Trial free text spells the same dose with an NBSP or a plain space, and both
        # must fold to one candidate -- split across two, each half can fall below
        # MIN_TRIALS and the label is lost. See the comment on `_normalize_name`.
        df = pl.DataFrame({'raw': ['rosuvastatin 20\xa0mg', 'rosuvastatin 20 mg']})
        out = df.with_columns(norm=_normalize_name(pl.col('raw')))['norm'].to_list()
        assert out[0] == out[1] == 'rosuvastatin 20 mg'


class TestParseAactEntries:
    def _batch(self, investigated: list[dict], comparator: list[dict], supportive: list[dict]) -> pl.DataFrame:
        drug_struct = pl.Struct({'drug': pl.Utf8, 'synonyms': pl.List(pl.Utf8)})
        schema = {
            'id': pl.Utf8,
            'investigated_drugs': pl.List(drug_struct),
            'comparator_drugs': pl.List(drug_struct),
            'supportive_drugs': pl.List(drug_struct),
        }
        return pl.DataFrame(
            {
                'id': ['NCT01'],
                'investigated_drugs': [investigated],
                'comparator_drugs': [comparator],
                'supportive_drugs': [supportive],
            },
            schema=schema,
        )

    def test_parse_extracts_all_roles(self):
        batch = self._batch(
            investigated=[{'drug': 'Lenalidomide', 'synonyms': ['Revlimid', 'CC-5013']}],
            comparator=[{'drug': 'Dexamethasone', 'synonyms': []}],
            supportive=[{'drug': 'Filgrastim', 'synonyms': ['G-CSF']}],
        )
        out = parse_aact_entries(batch)
        member_sets = [set(m) for m in out['members'].to_list()]
        assert {'cc-5013', 'lenalidomide', 'revlimid'} in member_sets
        assert {'filgrastim', 'g-csf'} in member_sets
        assert {'dexamethasone'} in member_sets
        assert all(x == 'NCT01' for x in out['nct_id'].to_list())

    def test_no_drug_entries_yields_no_rows(self):
        batch = self._batch(investigated=[], comparator=[], supportive=[])
        assert parse_aact_entries(batch).height == 0


class TestChemblIndexes:
    def _mol_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                'id': ['CHEMBL1', 'CHEMBL9', 'CHEMBL2'],
                'name': ['Filgrastim', 'Aspirin component of FOLFOX', 'Sub'],
                'synonyms': [
                    [{'label': 'Neupogen-syn', 'source': 'ChEMBL'}],
                    [{'label': 'ingredient X COMPONENT OF FOLFOX', 'source': 'ChEMBL'}],
                    [],
                ],
                'tradeNames': [[{'label': 'Neupogen', 'source': 'ChEMBL'}], [], []],
                'parentId': [None, None, 'CHEMBL1'],
                'childChemblIds': [['CHEMBL2'], [], []],
            },
            schema={
                'id': pl.Utf8,
                'name': pl.Utf8,
                'synonyms': _LABEL_SOURCE,
                'tradeNames': _LABEL_SOURCE,
                'parentId': pl.Utf8,
                'childChemblIds': pl.List(pl.Utf8),
            },
        )

    def test_name_index_covers_name_syn_trade(self):
        name_idx, _regimen, _pc = _build_chembl_indexes(self._mol_df())
        got = {r: set(i) for r, i in zip(name_idx['name_norm'], name_idx['ids'], strict=True)}
        assert got['filgrastim'] == {'CHEMBL1'}
        assert got['neupogen'] == {'CHEMBL1'}
        assert got['neupogen-syn'] == {'CHEMBL1'}

    def test_regimen_index_extracts_regimen(self):
        _name, regimen_idx, _pc = _build_chembl_indexes(self._mol_df())
        got = {r: set(i) for r, i in zip(regimen_idx['regimen_norm'], regimen_idx['ids'], strict=True)}
        assert got['folfox'] == {'CHEMBL9'}

    def test_parent_child_includes_children(self):
        _name, _regimen, pc = _build_chembl_indexes(self._mol_df())
        got = {r: set(i) for r, i in zip(pc['id'], pc['related'], strict=True)}
        assert 'CHEMBL2' in got['CHEMBL1']
        assert 'CHEMBL1' in got['CHEMBL2']


class TestAnchorCandidates:
    def test_synonym_anchors_novel_candidate(self):
        out = _anchor_candidates(
            _entries([{'nct_id': 'NCT1', 'members': ['filgrastim', 'g-csf']}]),
            _index([{'name_norm': 'filgrastim', 'ids': ['CHEMBL1']}]),
            _pc([{'id': 'CHEMBL1', 'related': []}]),
        )
        rows = set(zip(out['id'], out['candidate'], out['status'], strict=True))
        assert ('CHEMBL1', 'g-csf', 'NOVEL') in rows

    def test_over_ambiguous_member_skipped(self):
        out = _anchor_candidates(
            _entries([{'nct_id': 'NCT1', 'members': ['ssri', 'fluoxetine']}]),
            _index([{'name_norm': 'ssri', 'ids': [f'CHEMBL{i}' for i in range(11)]}]),
            _pc([]),
        )
        assert out.height == 0

    def test_conflict_status(self):
        # entry anchors CHEMBL1 (via 'filgrastim'); 'aspirin' resolves to unrelated
        # CHEMBL5 -> CONFLICT for CHEMBL1
        out = _anchor_candidates(
            _entries([{'nct_id': 'NCT1', 'members': ['filgrastim', 'aspirin']}]),
            _index([{'name_norm': 'filgrastim', 'ids': ['CHEMBL1']}, {'name_norm': 'aspirin', 'ids': ['CHEMBL5']}]),
            _pc([{'id': 'CHEMBL1', 'related': []}]),
        )
        rows = set(zip(out['id'], out['candidate'], out['status'], strict=True))
        assert ('CHEMBL1', 'aspirin', 'CONFLICT') in rows

    def test_parent_child_status(self):
        # entry anchors CHEMBL1; 'pegfilgrastim' resolves to CHEMBL2, a child of
        # CHEMBL1 -> PARENT_CHILD
        out = _anchor_candidates(
            _entries([{'nct_id': 'NCT1', 'members': ['filgrastim', 'pegfilgrastim']}]),
            _index([
                {'name_norm': 'filgrastim', 'ids': ['CHEMBL1']},
                {'name_norm': 'pegfilgrastim', 'ids': ['CHEMBL2']},
            ]),
            _pc([{'id': 'CHEMBL1', 'related': ['CHEMBL2']}]),
        )
        rows = set(zip(out['id'], out['candidate'], out['status'], strict=True))
        assert ('CHEMBL1', 'pegfilgrastim', 'PARENT_CHILD') in rows

    def test_exactly_cap_is_allowed(self):
        # 10 == cap -> entry NOT poisoned; 'g-csf' (unresolved) is a NOVEL candidate
        # for each of the 10
        out = _anchor_candidates(
            _entries([{'nct_id': 'NCT1', 'members': ['generic', 'g-csf']}]),
            _index([{'name_norm': 'generic', 'ids': [f'CHEMBL{i}' for i in range(10)]}]),
            _pc([]),
        )
        assert out.height != 0


class TestCleanupRules:
    def test_drops_parent_child_and_noise(self):
        regimen = _index([{'name_norm': 'folfox', 'ids': ['CHEMBLX']}]).rename({'name_norm': 'regimen_norm'})
        existing = pl.DataFrame(
            {'id': ['CHEMBL1'], 'existing': [['cyclosporin']]}, schema={'id': pl.Utf8, 'existing': pl.List(pl.Utf8)}
        )
        rows = [
            {'id': 'CHEMBL1', 'candidate': 'placebo', 'nct_id': 'N1', 'status': 'NOVEL'},
            {'id': 'CHEMBL1', 'candidate': 'dpp4 inhibitor', 'nct_id': 'N1', 'status': 'NOVEL'},
            # plural of a class keyword -- dropped only because the pattern expands them
            {'id': 'CHEMBL1', 'candidate': 'monoclonal antibodies', 'nct_id': 'N1', 'status': 'NOVEL'},
            # 'biosimilar' is on the exact-match blocklist, so a phrase containing it survives
            {'id': 'CHEMBL1', 'candidate': 'denosumab biosimilar (ct-p41)', 'nct_id': 'N1', 'status': 'NOVEL'},
            {'id': 'CHEMBL1', 'candidate': '1% lidocaine', 'nct_id': 'N1', 'status': 'NOVEL'},
            {'id': 'CHEMBL1', 'candidate': 'r', 'nct_id': 'N1', 'status': 'NOVEL'},
            {'id': 'CHEMBL1', 'candidate': 'folfox', 'nct_id': 'N1', 'status': 'NOVEL'},
            {'id': 'CHEMBL1', 'candidate': 'cyclosporins', 'nct_id': 'N1', 'status': 'NOVEL'},
            {'id': 'CHEMBL1', 'candidate': 'mtx', 'nct_id': 'N1', 'status': 'PARENT_CHILD'},
            {'id': 'CHEMBL1', 'candidate': 'g-csf', 'nct_id': 'N1', 'status': 'NOVEL'},
        ]
        out = _apply_cleanup_rules(_cand(rows), regimen, existing)
        assert set(out['candidate']) == {'g-csf', 'denosumab biosimilar (ct-p41)'}

    def test_conflict_kept(self):
        regimen = _index([]).rename({'name_norm': 'regimen_norm'})
        existing = pl.DataFrame(
            {'id': ['CHEMBL1'], 'existing': [[]]}, schema={'id': pl.Utf8, 'existing': pl.List(pl.Utf8)}
        )
        rows = [{'id': 'CHEMBL1', 'candidate': 'aspirin', 'nct_id': 'N1', 'status': 'CONFLICT'}]
        out = _apply_cleanup_rules(_cand(rows), regimen, existing)
        assert set(out['candidate']) == {'aspirin'}

    def test_word_boundary_not_substring(self):
        # 'nystatin' contains 'statin' and 'cellcept' contains 'cell' as SUBSTRINGS,
        # not whole words -> kept
        regimen = _index([]).rename({'name_norm': 'regimen_norm'})
        existing = pl.DataFrame(
            {'id': ['CHEMBL1'], 'existing': [[]]}, schema={'id': pl.Utf8, 'existing': pl.List(pl.Utf8)}
        )
        rows = [
            {'id': 'CHEMBL1', 'candidate': 'nystatin', 'nct_id': 'N1', 'status': 'NOVEL'},
            {'id': 'CHEMBL1', 'candidate': 'cellcept', 'nct_id': 'N1', 'status': 'NOVEL'},
        ]
        out = _apply_cleanup_rules(_cand(rows), regimen, existing)
        assert set(out['candidate']) == {'nystatin', 'cellcept'}


class TestRewriteAndReclassify:
    def _run(self, candidate: str, index_rows: list[dict], pc_rows: list[dict], status: str = 'NOVEL') -> set:
        cand = _cand([{'id': 'CHEMBL1', 'candidate': candidate, 'nct_id': 'N1', 'status': status}])
        out = _rewrite_and_reclassify_codes(cand, _index(index_rows), _pc(pc_rows))
        return set(zip(out['candidate'], out['status'], strict=True))

    def test_descriptor_code_extraction(self):
        assert self._run('akt inhibitor mk2206', [], []) == {('mk2206', 'NOVEL')}

    def test_phrase_with_code_rewritten_and_kept(self):
        assert self._run('mek inhibitor pd0325901', [], []) == {('pd0325901', 'NOVEL')}

    def test_rewritten_code_already_on_anchor_dropped(self):
        # the extracted code is already a label of the anchor CHEMBL1 -> redundant -> dropped
        out = self._run('mek inhibitor pd0325901', [{'name_norm': 'pd0325901', 'ids': ['CHEMBL1']}], [])
        assert out == set()

    def test_rewritten_code_on_parent_child_reclassified(self):
        # the extracted code resolves to CHEMBL2, a child of the anchor CHEMBL1 ->
        # PARENT_CHILD (this is the bug the reclassification fixes: it was stale
        # NOVEL before)
        out = self._run(
            'mek inhibitor pd0325901',
            [{'name_norm': 'pd0325901', 'ids': ['CHEMBL2']}],
            [{'id': 'CHEMBL1', 'related': ['CHEMBL2']}],
        )
        assert out == {('pd0325901', 'PARENT_CHILD')}

    def test_rewritten_code_unrelated_is_conflict(self):
        # the extracted code resolves to unrelated CHEMBL9 -> CONFLICT (kept, per design)
        out = self._run('mek inhibitor pd0325901', [{'name_norm': 'pd0325901', 'ids': ['CHEMBL9']}], [])
        assert out == {('pd0325901', 'CONFLICT')}

    def test_non_descriptor_candidate_passes_through(self):
        # 'g-csf' has no class keyword and no extractable code -> unchanged, stays NOVEL
        assert self._run('g-csf', [], []) == {('g-csf', 'NOVEL')}


class TestMineAactSynonyms:
    def _mol_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                'id': ['CHEMBL1'],
                'name': ['Filgrastim'],
                'synonyms': [[]],
                'tradeNames': [[]],
                'parentId': [None],
                'childChemblIds': [[]],
            },
            schema={
                'id': pl.Utf8,
                'name': pl.Utf8,
                'synonyms': _LABEL_SOURCE,
                'tradeNames': _LABEL_SOURCE,
                'parentId': pl.Utf8,
                'childChemblIds': pl.List(pl.Utf8),
            },
        )

    def test_min_trials_gate_and_anchor(self):
        entries = _entries([
            {'nct_id': 'NCT1', 'members': ['filgrastim', 'g-csf']},
            {'nct_id': 'NCT2', 'members': ['filgrastim', 'g-csf']},  # g-csf seen in 2 trials -> kept
            {'nct_id': 'NCT3', 'members': ['filgrastim', 'csa-once']},  # csa-once seen in 1 trial -> dropped
        ])
        out = set(zip(*(mine_aact_synonyms(self._mol_df(), entries)[c] for c in ('id', 'label')), strict=True))
        assert ('CHEMBL1', 'g-csf') in out
        assert ('CHEMBL1', 'csa-once') not in out

    def test_same_trial_duplicate_counts_once(self):
        entries = _entries([
            {'nct_id': 'NCT1', 'members': ['filgrastim', 'g-csf']},
            {'nct_id': 'NCT1', 'members': ['filgrastim', 'g-csf']},  # same trial, duplicate -> counts as 1
        ])
        out = set(zip(*(mine_aact_synonyms(self._mol_df(), entries)[c] for c in ('id', 'label')), strict=True))
        assert ('CHEMBL1', 'g-csf') not in out  # only 1 distinct trial -> below MIN_TRIALS


class TestMergeAactSynonyms:
    def test_aact_label_already_in_chembl_synonyms_not_duplicated(self):
        mol_combined = pl.DataFrame(
            {'id': ['CHEMBL1'], 'synonyms': [[{'label': 'G-CSF', 'source': 'ChEMBL'}]]},
            schema={'id': pl.Utf8, 'synonyms': _LABEL_SOURCE},
        )
        aact_df = pl.DataFrame({'id': ['CHEMBL1'], 'label': ['g-csf']}, schema={'id': pl.Utf8, 'label': pl.Utf8})
        row = merge_aact_synonyms(mol_combined, aact_df).to_dicts()[0]
        aact_labels = {s['label'] for s in row['synonyms'] if s['source'] == 'AACT'}
        assert aact_labels == set()  # 'g-csf' suppressed by existing 'G-CSF'
        assert any(s['label'] == 'G-CSF' and s['source'] == 'ChEMBL' for s in row['synonyms'])
