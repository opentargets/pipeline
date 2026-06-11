"""Tests for the aact_synonyms module (clinical-trial synonym mining)."""

import json

import pyspark.sql.functions as f
from pyspark.sql import Row
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

LABEL_SOURCE_SCHEMA_T = ArrayType(StructType([
    StructField('label', StringType()),
    StructField('source', StringType()),
]))


class TestNormalizeName:
    def test_normalization(self, spark):
        from pts.pyspark.drug_utils.aact_synonyms import _normalize_name
        data = [
            Row(raw='  Revlimid®  '),
            Row(raw='G  CSF'),
            Row(raw='Aspirin™'),
        ]
        df = spark.createDataFrame(data, StructType([StructField('raw', StringType())]))
        out = {r['raw']: r['norm'] for r in df.withColumn('norm', _normalize_name(f.col('raw'))).collect()}
        assert out['  Revlimid®  '] == 'revlimid'
        assert out['G  CSF'] == 'g csf'
        assert out['Aspirin™'] == 'aspirin'


class TestParseAactBatch:
    def _batch_df(self, spark, text_payload, custom_id='NCT01'):
        outer_schema = StructType([
            StructField('custom_id', StringType()),
            StructField('response', StructType([
                StructField('body', StructType([
                    StructField('output', ArrayType(StructType([
                        StructField('type', StringType()),
                        StructField('content', ArrayType(StructType([
                            StructField('text', StringType()),
                        ]))),
                    ]))),
                ])),
            ])),
        ])
        data = [Row(
            custom_id=custom_id,
            response=Row(body=Row(output=[
                Row(type='message', content=[Row(text=text_payload)]),
            ])),
        )]
        return spark.createDataFrame(data, outer_schema)

    def test_parse_extracts_all_roles(self, spark):
        from pts.pyspark.drug_utils.aact_synonyms import parse_aact_batch
        payload = json.dumps({
            'investigated_drugs': [{'drug': 'Lenalidomide', 'synonyms': ['Revlimid', 'CC-5013']}],
            'comparator_drugs': [{'drug': 'Dexamethasone', 'synonyms': []}],
            'supportive_drugs': [{'drug': 'Filgrastim', 'synonyms': ['G-CSF']}],
        })
        out = parse_aact_batch(self._batch_df(spark, payload)).collect()
        member_sets = [set(r['members']) for r in out]
        assert {'cc-5013', 'lenalidomide', 'revlimid'} in member_sets
        assert {'filgrastim', 'g-csf'} in member_sets
        assert {'dexamethasone'} in member_sets
        assert all(r['nct_id'] == 'NCT01' for r in out)

    def test_malformed_json_dropped(self, spark):
        from pts.pyspark.drug_utils.aact_synonyms import parse_aact_batch
        out = parse_aact_batch(self._batch_df(spark, 'not-valid-json')).collect()
        assert out == []


class TestChemblIndexes:
    def _mol_df(self, spark):
        schema = StructType([
            StructField('id', StringType()),
            StructField('name', StringType()),
            StructField('synonyms', LABEL_SOURCE_SCHEMA_T),
            StructField('tradeNames', LABEL_SOURCE_SCHEMA_T),
            StructField('parentId', StringType()),
            StructField('childChemblIds', ArrayType(StringType())),
        ])
        data = [
            Row(id='CHEMBL1', name='Filgrastim',
                synonyms=[Row(label='Neupogen-syn', source='ChEMBL')],
                tradeNames=[Row(label='Neupogen', source='ChEMBL')],
                parentId=None, childChemblIds=['CHEMBL2']),
            Row(id='CHEMBL9', name='Aspirin component of FOLFOX',
                synonyms=[Row(label='ingredient X COMPONENT OF FOLFOX', source='ChEMBL')],
                tradeNames=[], parentId=None, childChemblIds=[]),
            Row(id='CHEMBL2', name='Sub', synonyms=[], tradeNames=[],
                parentId='CHEMBL1', childChemblIds=[]),
        ]
        return spark.createDataFrame(data, schema)

    def test_name_index_covers_name_syn_trade(self, spark):
        from pts.pyspark.drug_utils.aact_synonyms import _build_chembl_indexes
        name_idx, _regimen, _pc = _build_chembl_indexes(self._mol_df(spark))
        got = {r['name_norm']: set(r['ids']) for r in name_idx.collect()}
        assert got['filgrastim'] == {'CHEMBL1'}
        assert got['neupogen'] == {'CHEMBL1'}
        assert got['neupogen-syn'] == {'CHEMBL1'}

    def test_regimen_index_extracts_regimen(self, spark):
        from pts.pyspark.drug_utils.aact_synonyms import _build_chembl_indexes
        _name, regimen_idx, _pc = _build_chembl_indexes(self._mol_df(spark))
        got = {r['regimen_norm']: set(r['ids']) for r in regimen_idx.collect()}
        assert got['folfox'] == {'CHEMBL9'}

    def test_parent_child_includes_children(self, spark):
        from pts.pyspark.drug_utils.aact_synonyms import _build_chembl_indexes
        _name, _regimen, pc = _build_chembl_indexes(self._mol_df(spark))
        got = {r['id']: set(r['related']) for r in pc.collect()}
        assert 'CHEMBL2' in got['CHEMBL1']
        assert 'CHEMBL1' in got['CHEMBL2']


class TestAnchorCandidates:
    def test_synonym_anchors_novel_candidate(self, spark):
        from pts.pyspark.drug_utils.aact_synonyms import _anchor_candidates
        entries = spark.createDataFrame(
            [Row(nct_id='NCT1', members=['filgrastim', 'g-csf'])],
            StructType([StructField('nct_id', StringType()), StructField('members', ArrayType(StringType()))]),
        )
        name_index = spark.createDataFrame(
            [Row(name_norm='filgrastim', ids=['CHEMBL1'])],
            StructType([StructField('name_norm', StringType()), StructField('ids', ArrayType(StringType()))]),
        )
        pc = spark.createDataFrame(
            [Row(id='CHEMBL1', related=[])],
            StructType([StructField('id', StringType()), StructField('related', ArrayType(StringType()))]),
        )
        out = _anchor_candidates(entries, name_index, pc).collect()
        rows = {(r['id'], r['candidate'], r['status']) for r in out}
        assert ('CHEMBL1', 'g-csf', 'NOVEL') in rows

    def test_over_ambiguous_member_skipped(self, spark):
        from pts.pyspark.drug_utils.aact_synonyms import _anchor_candidates
        entries = spark.createDataFrame(
            [Row(nct_id='NCT1', members=['ssri', 'fluoxetine'])],
            StructType([StructField('nct_id', StringType()), StructField('members', ArrayType(StringType()))]),
        )
        # 'ssri' resolves to 11 molecules -> entry must not anchor through it
        name_index = spark.createDataFrame(
            [Row(name_norm='ssri', ids=[f'CHEMBL{i}' for i in range(11)])],
            StructType([StructField('name_norm', StringType()), StructField('ids', ArrayType(StringType()))]),
        )
        pc = spark.createDataFrame(
            [], StructType([StructField('id', StringType()), StructField('related', ArrayType(StringType()))]),
        )
        out = _anchor_candidates(entries, name_index, pc).collect()
        assert out == []

    def test_conflict_status(self, spark):
        from pts.pyspark.drug_utils.aact_synonyms import _anchor_candidates
        # entry anchors CHEMBL1 (via 'filgrastim'); 'aspirin' resolves to unrelated CHEMBL5 -> CONFLICT for CHEMBL1
        entries = spark.createDataFrame(
            [Row(nct_id='NCT1', members=['filgrastim', 'aspirin'])],
            StructType([StructField('nct_id', StringType()), StructField('members', ArrayType(StringType()))]),
        )
        name_index = spark.createDataFrame(
            [Row(name_norm='filgrastim', ids=['CHEMBL1']), Row(name_norm='aspirin', ids=['CHEMBL5'])],
            StructType([StructField('name_norm', StringType()), StructField('ids', ArrayType(StringType()))]),
        )
        pc = spark.createDataFrame(
            [Row(id='CHEMBL1', related=[])],
            StructType([StructField('id', StringType()), StructField('related', ArrayType(StringType()))]),
        )
        out = {(r['id'], r['candidate'], r['status']) for r in _anchor_candidates(entries, name_index, pc).collect()}
        assert ('CHEMBL1', 'aspirin', 'CONFLICT') in out

    def test_parent_child_status(self, spark):
        from pts.pyspark.drug_utils.aact_synonyms import _anchor_candidates
        # entry anchors CHEMBL1; 'pegfilgrastim' resolves to CHEMBL2 which is a child of CHEMBL1 -> PARENT_CHILD
        entries = spark.createDataFrame(
            [Row(nct_id='NCT1', members=['filgrastim', 'pegfilgrastim'])],
            StructType([StructField('nct_id', StringType()), StructField('members', ArrayType(StringType()))]),
        )
        name_index = spark.createDataFrame(
            [Row(name_norm='filgrastim', ids=['CHEMBL1']), Row(name_norm='pegfilgrastim', ids=['CHEMBL2'])],
            StructType([StructField('name_norm', StringType()), StructField('ids', ArrayType(StringType()))]),
        )
        pc = spark.createDataFrame(
            [Row(id='CHEMBL1', related=['CHEMBL2'])],
            StructType([StructField('id', StringType()), StructField('related', ArrayType(StringType()))]),
        )
        out = {(r['id'], r['candidate'], r['status']) for r in _anchor_candidates(entries, name_index, pc).collect()}
        assert ('CHEMBL1', 'pegfilgrastim', 'PARENT_CHILD') in out

    def test_exactly_cap_is_allowed(self, spark):
        from pts.pyspark.drug_utils.aact_synonyms import _anchor_candidates
        entries = spark.createDataFrame(
            [Row(nct_id='NCT1', members=['generic', 'g-csf'])],
            StructType([StructField('nct_id', StringType()), StructField('members', ArrayType(StringType()))]),
        )
        name_index = spark.createDataFrame(
            [Row(name_norm='generic', ids=[f'CHEMBL{i}' for i in range(10)])],
            StructType([StructField('name_norm', StringType()), StructField('ids', ArrayType(StringType()))]),
        )
        pc = spark.createDataFrame(
            [], StructType([StructField('id', StringType()), StructField('related', ArrayType(StringType()))]),
        )
        # 10 == cap -> entry NOT poisoned; 'g-csf' (unresolved) is a NOVEL candidate for each of the 10
        out = _anchor_candidates(entries, name_index, pc).collect()
        assert out != []


class TestCleanupRules:
    def _df(self, spark, rows):
        schema = StructType([
            StructField('id', StringType()),
            StructField('candidate', StringType()),
            StructField('nct_id', StringType()),
            StructField('status', StringType()),
        ])
        return spark.createDataFrame([Row(**r) for r in rows], schema)

    def test_drops_parent_child_and_noise(self, spark):
        from pts.pyspark.drug_utils.aact_synonyms import _apply_cleanup_rules
        regimen = spark.createDataFrame(
            [Row(regimen_norm='folfox', ids=['CHEMBLX'])],
            StructType([StructField('regimen_norm', StringType()), StructField('ids', ArrayType(StringType()))]),
        )
        existing = spark.createDataFrame(
            [Row(id='CHEMBL1', existing=['cyclosporin'])],
            StructType([StructField('id', StringType()), StructField('existing', ArrayType(StringType()))]),
        )
        rows = [
            {'id': 'CHEMBL1', 'candidate': 'placebo', 'nct_id': 'N1', 'status': 'NOVEL'},
            {'id': 'CHEMBL1', 'candidate': 'dpp4 inhibitor', 'nct_id': 'N1', 'status': 'NOVEL'},
            {'id': 'CHEMBL1', 'candidate': '1% lidocaine', 'nct_id': 'N1', 'status': 'NOVEL'},
            {'id': 'CHEMBL1', 'candidate': 'r', 'nct_id': 'N1', 'status': 'NOVEL'},
            {'id': 'CHEMBL1', 'candidate': 'folfox', 'nct_id': 'N1', 'status': 'NOVEL'},
            {'id': 'CHEMBL1', 'candidate': 'cyclosporins', 'nct_id': 'N1', 'status': 'NOVEL'},
            {'id': 'CHEMBL1', 'candidate': 'mtx', 'nct_id': 'N1', 'status': 'PARENT_CHILD'},
            {'id': 'CHEMBL1', 'candidate': 'g-csf', 'nct_id': 'N1', 'status': 'NOVEL'},
        ]
        out = _apply_cleanup_rules(self._df(spark, rows), regimen, existing)
        kept = {r['candidate'] for r in out.collect()}
        assert kept == {'g-csf'}

    def test_conflict_kept(self, spark):
        from pts.pyspark.drug_utils.aact_synonyms import _apply_cleanup_rules
        regimen = spark.createDataFrame(
            [], StructType([StructField('regimen_norm', StringType()), StructField('ids', ArrayType(StringType()))]),
        )
        existing = spark.createDataFrame(
            [Row(id='CHEMBL1', existing=[])],
            StructType([StructField('id', StringType()), StructField('existing', ArrayType(StringType()))]),
        )
        rows = [{'id': 'CHEMBL1', 'candidate': 'aspirin', 'nct_id': 'N1', 'status': 'CONFLICT'}]
        out = {r['candidate'] for r in _apply_cleanup_rules(self._df(spark, rows), regimen, existing).collect()}
        assert out == {'aspirin'}

    def test_word_boundary_not_substring(self, spark):
        from pts.pyspark.drug_utils.aact_synonyms import _apply_cleanup_rules
        regimen = spark.createDataFrame(
            [], StructType([StructField('regimen_norm', StringType()), StructField('ids', ArrayType(StringType()))]),
        )
        existing = spark.createDataFrame(
            [Row(id='CHEMBL1', existing=[])],
            StructType([StructField('id', StringType()), StructField('existing', ArrayType(StringType()))]),
        )
        # 'nystatin' contains 'statin' and 'cellcept' contains 'cell' as SUBSTRINGS, not whole words -> kept
        rows = [
            {'id': 'CHEMBL1', 'candidate': 'nystatin', 'nct_id': 'N1', 'status': 'NOVEL'},
            {'id': 'CHEMBL1', 'candidate': 'cellcept', 'nct_id': 'N1', 'status': 'NOVEL'},
        ]
        out = {r['candidate'] for r in _apply_cleanup_rules(self._df(spark, rows), regimen, existing).collect()}
        assert out == {'nystatin', 'cellcept'}


class TestRewriteAndReclassify:
    def _cand(self, spark, rows):
        schema = StructType([
            StructField('id', StringType()),
            StructField('candidate', StringType()),
            StructField('nct_id', StringType()),
            StructField('status', StringType()),
        ])
        return spark.createDataFrame([Row(**r) for r in rows], schema)

    def _idx(self, spark, rows):
        return spark.createDataFrame(
            [Row(**r) for r in rows],
            StructType([StructField('name_norm', StringType()), StructField('ids', ArrayType(StringType()))]),
        )

    def _pc(self, spark, rows):
        return spark.createDataFrame(
            [Row(**r) for r in rows],
            StructType([StructField('id', StringType()), StructField('related', ArrayType(StringType()))]),
        )

    def _run(self, spark, candidate, name_index_rows, pc_rows, status='NOVEL'):
        from pts.pyspark.drug_utils.aact_synonyms import _rewrite_and_reclassify_codes
        cand = self._cand(spark, [{'id': 'CHEMBL1', 'candidate': candidate, 'nct_id': 'N1', 'status': status}])
        out = _rewrite_and_reclassify_codes(cand, self._idx(spark, name_index_rows), self._pc(spark, pc_rows))
        return {(r['candidate'], r['status']) for r in out.collect()}

    def test_descriptor_code_extraction(self, spark):
        out = self._run(spark, 'akt inhibitor mk2206', [], [])
        assert out == {('mk2206', 'NOVEL')}

    def test_phrase_with_code_rewritten_and_kept(self, spark):
        out = self._run(spark, 'mek inhibitor pd0325901', [], [])
        assert out == {('pd0325901', 'NOVEL')}

    def test_rewritten_code_already_on_anchor_dropped(self, spark):
        # the extracted code is already a label of the anchor CHEMBL1 -> redundant -> dropped
        out = self._run(spark, 'mek inhibitor pd0325901', [{'name_norm': 'pd0325901', 'ids': ['CHEMBL1']}], [])
        assert out == set()

    def test_rewritten_code_on_parent_child_reclassified(self, spark):
        # the extracted code resolves to CHEMBL2, a child of the anchor CHEMBL1 -> PARENT_CHILD
        # (this is the bug the reclassification fixes: it was stale NOVEL before)
        out = self._run(
            spark,
            'mek inhibitor pd0325901',
            [{'name_norm': 'pd0325901', 'ids': ['CHEMBL2']}],
            [{'id': 'CHEMBL1', 'related': ['CHEMBL2']}],
        )
        assert out == {('pd0325901', 'PARENT_CHILD')}

    def test_rewritten_code_unrelated_is_conflict(self, spark):
        # the extracted code resolves to unrelated CHEMBL9 -> CONFLICT (kept, per design)
        out = self._run(spark, 'mek inhibitor pd0325901', [{'name_norm': 'pd0325901', 'ids': ['CHEMBL9']}], [])
        assert out == {('pd0325901', 'CONFLICT')}

    def test_non_descriptor_candidate_passes_through(self, spark):
        # 'g-csf' has no class keyword and no extractable code -> unchanged, stays NOVEL
        out = self._run(spark, 'g-csf', [], [])
        assert out == {('g-csf', 'NOVEL')}


class TestMineAactSynonyms:
    def _mol_df(self, spark):
        return spark.createDataFrame(
            [Row(id='CHEMBL1', name='Filgrastim', synonyms=[], tradeNames=[], parentId=None, childChemblIds=[])],
            StructType([
                StructField('id', StringType()), StructField('name', StringType()),
                StructField('synonyms', LABEL_SOURCE_SCHEMA_T), StructField('tradeNames', LABEL_SOURCE_SCHEMA_T),
                StructField('parentId', StringType()), StructField('childChemblIds', ArrayType(StringType())),
            ]),
        )

    def test_min_trials_gate_and_anchor(self, spark):
        from pts.pyspark.drug_utils.aact_synonyms import mine_aact_synonyms
        entries = spark.createDataFrame(
            [
                Row(nct_id='NCT1', members=['filgrastim', 'g-csf']),
                Row(nct_id='NCT2', members=['filgrastim', 'g-csf']),   # g-csf seen in 2 trials -> kept
                Row(nct_id='NCT3', members=['filgrastim', 'csa-once']),  # csa-once seen in 1 trial -> dropped
            ],
            StructType([StructField('nct_id', StringType()), StructField('members', ArrayType(StringType()))]),
        )
        out = {(r['id'], r['label']) for r in mine_aact_synonyms(self._mol_df(spark), entries).collect()}
        assert ('CHEMBL1', 'g-csf') in out
        assert ('CHEMBL1', 'csa-once') not in out

    def test_same_trial_duplicate_counts_once(self, spark):
        from pts.pyspark.drug_utils.aact_synonyms import mine_aact_synonyms
        entries = spark.createDataFrame(
            [
                Row(nct_id='NCT1', members=['filgrastim', 'g-csf']),
                Row(nct_id='NCT1', members=['filgrastim', 'g-csf']),  # same trial, duplicate -> counts as 1
            ],
            StructType([StructField('nct_id', StringType()), StructField('members', ArrayType(StringType()))]),
        )
        out = {(r['id'], r['label']) for r in mine_aact_synonyms(self._mol_df(spark), entries).collect()}
        assert ('CHEMBL1', 'g-csf') not in out  # only 1 distinct trial -> below MIN_TRIALS


class TestMergeAactSynonyms:
    def test_aact_label_already_in_chembl_synonyms_not_duplicated(self, spark):
        """An AACT label matching an existing ChEMBL synonym (case-insensitively) is not added again."""
        from pts.pyspark.drug_utils.aact_synonyms import merge_aact_synonyms
        mol_combined = spark.createDataFrame(
            [Row(id='CHEMBL1', synonyms=[Row(label='G-CSF', source='ChEMBL')])],
            StructType([StructField('id', StringType()), StructField('synonyms', LABEL_SOURCE_SCHEMA_T)]),
        )
        aact_df = spark.createDataFrame(
            [Row(id='CHEMBL1', label='g-csf')],
            StructType([StructField('id', StringType()), StructField('label', StringType())]),
        )
        row = merge_aact_synonyms(mol_combined, aact_df).collect()[0]
        aact_labels = {s['label'] for s in row['synonyms'] if s['source'] == 'AACT'}
        assert aact_labels == set()  # 'g-csf' suppressed by existing 'G-CSF'
        assert any(s['label'] == 'G-CSF' and s['source'] == 'ChEMBL' for s in row['synonyms'])
