import polars as pl

from pts.transformers.clinical_mira.providers.aact_context import render_trial, trial_context


def _tables(**overrides: pl.DataFrame) -> dict[str, pl.DataFrame]:
    tables = {
        'studies': pl.DataFrame({
            'nct_id': ['NCT1'],
            'official_title': ['Drug X added to cisplatin'],
            'phase': ['PHASE3'],
        }),
        'summaries': pl.DataFrame({'nct_id': ['NCT1'], 'description': ['Does X improve survival?']}),
        'details': pl.DataFrame({'nct_id': ['NCT1'], 'description': ['Randomised, open label.']}),
        'interventions': pl.DataFrame({
            'id': [1, 2],
            'nct_id': ['NCT1', 'NCT1'],
            'intervention_type': ['DRUG', 'DRUG'],
            'name': ['Cisplatin', 'Drug X'],
            'description': ['75 mg/m2 day 1', '10 mg daily'],
        }),
        'design_groups': pl.DataFrame({
            'id': [10, 11],
            'nct_id': ['NCT1', 'NCT1'],
            'group_type': ['EXPERIMENTAL', 'ACTIVE_COMPARATOR'],
            'title': ['X plus cisplatin', 'Cisplatin alone'],
            'description': ['X with standard cisplatin', 'Standard cisplatin'],
        }),
        'design_group_interventions': pl.DataFrame({
            'id': [100, 101, 102],
            'nct_id': ['NCT1', 'NCT1', 'NCT1'],
            'design_group_id': [10, 10, 11],
            'intervention_id': [1, 2, 1],
        }),
        'design_outcomes': pl.DataFrame({
            'nct_id': ['NCT1', 'NCT1'],
            'outcome_type': ['PRIMARY', 'SECONDARY'],
            'measure': ['Overall survival', 'Toxicity'],
        }),
        'designs': pl.DataFrame({
            'nct_id': ['NCT1'],
            'primary_purpose': ['TREATMENT'],
            'allocation': ['RANDOMIZED'],
            'intervention_model': ['PARALLEL'],
            'masking': ['NONE'],
        }),
    }
    return tables | overrides


def _rendered(**overrides: pl.DataFrame) -> str:
    context = trial_context(**_tables(**overrides))
    return render_trial(context.row(0, named=True))


def test_arms_list_their_interventions() -> None:
    row = trial_context(**_tables()).row(0, named=True)
    arms = {arm['title']: arm['interventions'] for arm in row['arms']}
    assert arms == {'X plus cisplatin': ['Cisplatin', 'Drug X'], 'Cisplatin alone': ['Cisplatin']}
    assert row['primary_outcomes'] == ['Overall survival']
    text = render_trial(row)
    assert 'X plus cisplatin' in text
    assert 'Interventions: Cisplatin; Drug X' in text


def test_trial_without_arms_still_renders() -> None:
    empty_groups = _tables()['design_groups'].clear()
    empty_links = _tables()['design_group_interventions'].clear()
    text = _rendered(design_groups=empty_groups, design_group_interventions=empty_links)
    assert 'Arms: none listed' in text
    assert 'Drug X' in text
    assert 'Cisplatin' in text


def test_render_is_deterministic() -> None:
    tables = _tables()
    shuffled = {name: frame.reverse() for name, frame in tables.items()}
    assert _rendered(**tables) == _rendered(**shuffled)


def test_render_fits_the_budget() -> None:
    n = 30
    names = [f'Drug {i:02d}' for i in range(n)]
    tables = _tables(
        details=pl.DataFrame({'nct_id': ['NCT1'], 'description': ['word ' * 4000]}),
        interventions=pl.DataFrame({
            'id': list(range(n)),
            'nct_id': ['NCT1'] * n,
            'intervention_type': ['DRUG'] * n,
            'name': names,
            'description': ['dose ' * 100] * n,
        }),
        design_groups=pl.DataFrame({
            'id': list(range(n)),
            'nct_id': ['NCT1'] * n,
            'group_type': ['EXPERIMENTAL'] * n,
            'title': [f'Arm {i:02d}' for i in range(n)],
            'description': ['arm text ' * 50] * n,
        }),
        design_group_interventions=pl.DataFrame({
            'id': list(range(n)),
            'nct_id': ['NCT1'] * n,
            'design_group_id': list(range(n)),
            'intervention_id': list(range(n)),
        }),
    )
    text = render_trial(trial_context(**tables).row(0, named=True))
    assert len(text) <= 12000
    assert all(name in text for name in names)
    assert all(f'Arm {i:02d}' in text for i in range(n))
    assert 'Drug X added to cisplatin' in text


def test_interventions_without_a_name_are_left_out() -> None:
    interventions = _tables()['interventions'].with_columns(name=pl.Series(['Cisplatin', None]))
    text = _rendered(interventions=interventions)
    assert 'Interventions: Cisplatin\n' in text
    assert 'None' not in text
