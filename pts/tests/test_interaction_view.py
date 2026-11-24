"""Tests for the interaction view transformer.

Each case carries over a semantic of the `interaction.sql` serving script it replaces.
"""

from pathlib import Path

import polars as pl
import pytest
from ot_datamodels.outputs.interaction import Interaction
from ot_datamodels.outputs.interaction_evidence import InteractionEvidence
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.views.interaction_view import InteractionView

from pts.schemas.dataset.conform import conform
from pts.schemas.dataset.validate import validate_dataset
from pts.task_outputs import OutputValidation
from pts.transformers import interaction_view as module
from pts.transformers.interaction_view import interaction_view, interaction_view_frame

INTERACTION = to_polars_schema(Interaction)
EVIDENCE = to_polars_schema(InteractionEvidence)
ROW = [
    'targetA',
    'intA',
    'targetB',
    'intB',
    'intABiologicalRole',
    'intBBiologicalRole',
    'scoring',
    'count',
    'sourceDatabase',
    'speciesA',
    'speciesB',
    'evidences',
]


def _interactions(*rows: dict) -> pl.DataFrame:
    return pl.DataFrame([Interaction.example | row for row in rows], schema=INTERACTION)


def _evidence(*rows: dict) -> pl.DataFrame:
    return pl.DataFrame([InteractionEvidence.example | row for row in rows], schema=EVIDENCE)


def _view(interactions: pl.DataFrame, evidence: pl.DataFrame) -> pl.DataFrame:
    return interaction_view_frame(interactions.lazy(), evidence.lazy())


def _by_target(view: pl.DataFrame) -> dict[str, list[dict]]:
    return dict(zip(view['targetA'], view['interactions'].to_list(), strict=True))


def test_interactions_of_a_target_nest_in_output_order():
    interactions = _interactions(
        {'targetA': 'T1', 'targetB': 'B2', 'interactionId': 2},
        {'targetA': 'T2', 'targetB': 'B9', 'interactionId': 9},
        {'targetA': 'T1', 'targetB': 'B1', 'interactionId': 1},
    )
    evidence = _evidence({'interactionId': 2}, {'interactionId': 9}, {'interactionId': 1})

    view = _view(interactions, evidence)

    assert view.columns == ['targetA', 'interactions']
    assert [[i['targetB'] for i in v] for v in _by_target(view).values()] == [['B2', 'B1'], ['B9']]
    assert list(_by_target(view)['T1'][0]) == ROW


def test_evidences_keep_output_order_and_the_served_fields():
    interactions = _interactions({'interactionId': 1}, {'targetB': 'B2', 'interactionId': 2})
    evidence = _evidence(
        {'interactionId': 1, 'pubmedId': 'p3'},
        {'interactionId': 2, 'pubmedId': 'p1'},
        {'interactionId': 1, 'pubmedId': 'p2'},
    )

    (interactions_of,) = _by_target(_view(interactions, evidence)).values()

    assert [[e['pubmedId'] for e in i['evidences']] for i in interactions_of] == [['p3', 'p2'], ['p1']]
    assert list(interactions_of[0]['evidences'][0]) == list(InteractionView.example['interactions'][0]['evidences'][0])


def test_enums_the_api_reads_as_strings_are_strings():
    view = conform(_view(_interactions({'interactionId': 1}), _evidence({'interactionId': 1})), InteractionView)

    rows = view.select('interactions').explode('interactions').unnest('interactions')
    evidences = rows.explode('evidences').unnest('evidences')
    assert rows.schema['sourceDatabase'] == pl.Enum(['intact', 'reactome', 'signor', 'string'])
    assert rows.schema['speciesA'] == pl.Struct({
        'mnemonic': pl.String,
        'scientificName': pl.String,
        'taxonId': pl.Int64,
    })
    assert evidences.schema['interactionResources'] == pl.Struct({
        'databaseVersion': pl.String,
        'sourceDatabase': pl.String,
    })
    (interactions_of,) = _by_target(view).values()
    assert interactions_of[0]['evidences'][0]['interactionResources'] == {
        'databaseVersion': '253',
        'sourceDatabase': 'intact',
    }


@pytest.mark.parametrize(
    ('taxon', 'served'),
    [(9606, 9606), (-1, 4294967295), (-2, 4294967294), (-3, 4294967293), (-4, 4294967292), (None, None)],
)
def test_taxon_ids_wrap_to_unsigned_as_the_cast_did(taxon, served):
    species_b = Interaction.example['speciesB'] | {'taxonId': taxon if taxon is not None else 1}
    interactions = _interactions({'interactionId': 1, 'speciesB': species_b})
    view = _view(interactions, _evidence({'interactionId': 1, 'hostOrganismTaxId': taxon}))

    (interactions_of,) = _by_target(view).values()

    assert interactions_of[0]['evidences'][0]['hostOrganismTaxId'] == served
    if taxon is not None:
        assert interactions_of[0]['speciesB']['taxonId'] == served


def test_interaction_without_evidence_raises():
    interactions = _interactions({'interactionId': 1}, {'targetB': 'B2', 'interactionId': 2})

    with pytest.raises(ValueError, match='interactions with no evidence: 1'):
        _view(interactions, _evidence({'interactionId': 1}))


def test_evidence_of_no_interaction_is_dropped():
    view = _view(_interactions({'interactionId': 1}), _evidence({'interactionId': 1}, {'interactionId': 7}))

    (interactions_of,) = _by_target(view).values()
    assert len(interactions_of[0]['evidences']) == 1


def test_nulls_are_kept():
    species_b = Interaction.example['speciesB'] | {'scientificName': None}
    interactions = _interactions({'interactionId': 1, 'targetB': None, 'speciesB': species_b, 'scoring': None})
    evidence = _evidence({'interactionId': 1, 'participantDetectionMethodA': None, 'pubmedId': None})

    (interactions_of,) = _by_target(_view(interactions, evidence)).values()

    (row,) = interactions_of
    assert row['targetB'] is None
    assert row['scoring'] is None
    assert row['speciesB']['scientificName'] is None
    assert row['evidences'][0]['participantDetectionMethodA'] is None
    assert row['evidences'][0]['pubmedId'] is None


def test_many_parts_build():
    n = 3_000
    interactions = pl.concat(
        [
            _interactions(*({'targetA': f'T{i // 7:04d}', 'interactionId': i} for i in range(start, start + 600)))
            for start in range(0, n, 600)
        ],
        rechunk=False,
    )
    evidence = pl.concat(
        [_evidence(*({'interactionId': i % n} for i in range(start, start + 900))) for start in range(0, 2 * n, 900)],
        rechunk=False,
    )
    assert interactions.n_chunks() > 1

    view = _view(interactions, evidence)

    assert view.height == len({i // 7 for i in range(n)})
    assert view['interactions'].list.len().sum() == n
    assert view['interactions'].list.eval(pl.element().struct.field('evidences').list.len()).explode().sum() == len(
        evidence
    )


def test_writes_its_model(tmp_path: Path):
    source = {}
    for name, frame in {
        'interaction': _interactions({'interactionId': 1}, {'targetA': 'T2', 'interactionId': 2}),
        'evidence': _evidence({'interactionId': 1}, {'interactionId': 2, 'hostOrganismTaxId': -1}),
    }.items():
        (tmp_path / name).mkdir()
        frame.write_parquet(tmp_path / name / 'part-0.parquet')
        source[name] = str(tmp_path / name)
    destination = {'view': str(tmp_path / 'view')}

    interaction_view(source, destination, {}, None)

    assert OutputValidation.of(interaction_view, module).outputs == {'view': InteractionView}
    validate_dataset(InteractionView, destination['view'])
    written = pl.read_parquet(destination['view'])
    assert written.schema == to_polars_schema(InteractionView)
    assert written.height == 2
