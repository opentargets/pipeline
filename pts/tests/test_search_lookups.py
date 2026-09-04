"""Tests for the search lookup tables and association aggregates."""

import polars as pl

from pts.transformers.search.helpers import LIST_STR
from pts.transformers.search.lookups import (
    association_scores,
    disease_lut,
    drug_associations,
    drug_associations_from_evidence,
    nct_by,
    nct_map,
    phenotype_names,
    resolve_ta_labels,
    scored_drug_associations,
    target_lut,
)

SYNONYM_STRUCT = pl.Struct(
    {
        'hasExactSynonym': LIST_STR,
        'hasRelatedSynonym': LIST_STR,
        'hasNarrowSynonym': LIST_STR,
        'hasBroadSynonym': LIST_STR,
    }
)
LABELLED = pl.List(pl.Struct({'label': pl.String, 'source': pl.String}))


def _diseases(rows):
    return pl.LazyFrame(
        rows,
        schema={
            'diseaseId': pl.String,
            'name': pl.String,
            'synonyms': SYNONYM_STRUCT,
            'therapeuticAreas': LIST_STR,
        },
    )


def test_resolve_ta_labels_maps_area_ids_to_their_names() -> None:
    frame = _diseases(
        [
            {'diseaseId': 'D1', 'name': 'child', 'synonyms': None, 'therapeuticAreas': ['D2']},
            {'diseaseId': 'D2', 'name': 'area', 'synonyms': None, 'therapeuticAreas': []},
        ]
    )

    result = resolve_ta_labels(frame).collect().sort('diseaseId')

    assert result.filter(pl.col('diseaseId') == 'D1')['therapeutic_labels'][0].to_list() == ['area']


def test_spark_resolve_ta_labels_leaves_a_disease_with_no_areas_null_not_empty() -> None:
    """This is what makes the disease index's `category` nullable. Spark's `explode` drops the
    empty array, so no row reaches the group-by, and the left join leaves NULL. Coalescing it
    to `[]` here would differ from the release on every top-level therapeutic area."""
    frame = _diseases([{'diseaseId': 'D2', 'name': 'area', 'synonyms': None, 'therapeuticAreas': []}])

    result = resolve_ta_labels(frame).collect()

    assert result['therapeutic_labels'][0] is None


def test_resolve_ta_labels_leaves_null_for_an_unresolvable_area_id() -> None:
    frame = _diseases([{'diseaseId': 'D1', 'name': 'child', 'synonyms': None, 'therapeuticAreas': ['MISSING']}])

    assert resolve_ta_labels(frame).collect()['therapeutic_labels'][0] is None


def test_phenotype_names_collects_hpo_labels_per_disease() -> None:
    disease_phenotype = pl.LazyFrame({'disease': ['D1', 'D1'], 'phenotype': ['HP1', 'HP2']})
    hpo = pl.LazyFrame({'id': ['HP1', 'HP2'], 'name': ['seizure', 'ataxia']})

    result = phenotype_names(disease_phenotype, hpo).collect()

    assert result['diseaseId'].to_list() == ['D1']
    assert sorted(result['phenotype_labels'][0].to_list()) == ['ataxia', 'seizure']


def test_disease_lut_merges_the_name_and_every_synonym_flavour() -> None:
    frame = _diseases(
        [
            {
                'diseaseId': 'D1',
                'name': 'asthma',
                'synonyms': {
                    'hasExactSynonym': ['exact'],
                    'hasRelatedSynonym': None,
                    'hasNarrowSynonym': ['narrow'],
                    'hasBroadSynonym': ['broad'],
                },
                'therapeuticAreas': [],
            }
        ]
    )

    result = disease_lut(resolve_ta_labels(frame)).collect()

    assert sorted(result['disease_labels'][0].to_list()) == ['asthma', 'broad', 'exact', 'narrow']
    assert result['disease_name'][0] == 'asthma'


def test_target_lut_merges_symbol_name_and_synonyms() -> None:
    frame = pl.LazyFrame(
        {
            'targetId': ['T1'],
            'approvedSymbol': ['EGFR'],
            'approvedName': ['receptor'],
            'synonyms': [[{'label': 'ERBB1', 'source': 'x'}]],
        },
        schema={
            'targetId': pl.String,
            'approvedSymbol': pl.String,
            'approvedName': pl.String,
            'synonyms': LABELLED,
        },
    )

    assert sorted(target_lut(frame).collect()['target_labels'][0].to_list()) == ['EGFR', 'ERBB1', 'receptor']


def test_association_scores_builds_the_id_as_disease_dash_target() -> None:
    frame = pl.LazyFrame({'diseaseId': ['D1'], 'targetId': ['T1'], 'associationScore': [0.5]})

    result = association_scores(frame).collect()

    assert result['associationId'][0] == 'D1-T1'
    assert result['score'][0] == 0.5


def test_spark_association_id_skips_a_null_component_with_no_dangling_separator() -> None:
    """`concat_ws` skips nulls -- it does not propagate them and does not leave a stray dash."""
    frame = pl.LazyFrame(
        {'diseaseId': [None], 'targetId': ['T1'], 'associationScore': [0.5]},
        schema={'diseaseId': pl.String, 'targetId': pl.String, 'associationScore': pl.Float64},
    )

    assert association_scores(frame).collect()['associationId'][0] == 'T1'


def test_scored_drug_associations_keeps_only_evidence_carrying_a_drug() -> None:
    evidence = pl.LazyFrame(
        {'drugId': ['CH1', None], 'targetId': ['T1', 'T1'], 'diseaseId': ['D1', 'D1']},
        schema={'drugId': pl.String, 'targetId': pl.String, 'diseaseId': pl.String},
    )
    scores = pl.LazyFrame({'associationId': ['D1-T1'], 'score': [0.5]})

    result = scored_drug_associations(drug_associations_from_evidence(evidence), scores).collect()

    assert result['drugId'].to_list() == ['CH1']
    assert result['score'].to_list() == [0.5]


def test_drug_associations_divides_the_association_count_by_the_global_total() -> None:
    scored = pl.LazyFrame(
        {
            'associationId': ['D1-T1', 'D2-T1'],
            'drugId': ['CH1', 'CH1'],
            'targetId': ['T1', 'T1'],
            'diseaseId': ['D1', 'D2'],
            'score': [0.4, 0.6],
        }
    )

    result = drug_associations(scored, total=4).collect()

    assert result['drug_relevance'].to_list() == [0.5]
    assert result['meanScore'].to_list() == [0.5]
    assert sorted(result['diseaseIds'][0].to_list()) == ['D1', 'D2']


def test_spark_the_relevance_denominator_counts_associations_before_the_score_join() -> None:
    """Spark counts drug-bearing associations straight off the evidence, BEFORE the inner join
    with the association scores. Counting after would drop every association that appears in
    evidence but not in the association dataset -- here 'D9-T9' -- and inflate the multiplier
    on every drug in the release."""
    evidence = pl.LazyFrame(
        {'drugId': ['CH1', 'CH1'], 'targetId': ['T1', 'T9'], 'diseaseId': ['D1', 'D9']},
        schema={'drugId': pl.String, 'targetId': pl.String, 'diseaseId': pl.String},
    )
    scores = pl.LazyFrame({'associationId': ['D1-T1'], 'score': [0.5]})

    from_evidence = drug_associations_from_evidence(evidence)
    total = from_evidence.select(pl.len()).collect().item()
    scored = scored_drug_associations(from_evidence, scores)

    assert total == 2, 'the unscored association must still count towards the denominator'
    assert scored.collect().height == 1
    assert drug_associations(scored, total).collect()['drug_relevance'].to_list() == [0.5]


def test_nct_map_keeps_only_nct_prefixed_report_ids() -> None:
    frame = pl.LazyFrame(
        {'clinicalReportIds': [['nct01', 'other'], ['other']], 'drugId': ['CH1', 'CH2'], 'diseaseId': ['D1', 'D2']},
        schema={'clinicalReportIds': LIST_STR, 'drugId': pl.String, 'diseaseId': pl.String},
    )

    result = nct_map(frame).collect()

    assert result['drugId'].to_list() == ['CH1']
    assert result['nctIds'][0].to_list() == ['nct01']


def test_spark_nct_map_drops_a_null_report_list() -> None:
    """Spark's `size(null)` is -1, so `size(nctIds) > 0` drops the row rather than erroring."""
    frame = pl.LazyFrame(
        {'clinicalReportIds': [None], 'drugId': ['CH1'], 'diseaseId': ['D1']},
        schema={'clinicalReportIds': LIST_STR, 'drugId': pl.String, 'diseaseId': pl.String},
    )

    assert nct_map(frame).collect().height == 0


def test_nct_by_merges_every_trial_for_one_key() -> None:
    frame = pl.LazyFrame(
        {'clinicalReportIds': [['nct01'], ['nct02']], 'drugId': ['CH1', 'CH1'], 'diseaseId': ['D1', 'D2']},
        schema={'clinicalReportIds': LIST_STR, 'drugId': pl.String, 'diseaseId': pl.String},
    )

    result = nct_by(frame, 'drugId').collect()

    assert sorted(result['nctIds'][0].to_list()) == ['nct01', 'nct02']
