"""Try to parse and validate the DAGs."""

from airflow.models import DagBag

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.models.pts_step import pts_step_from_config


def test_no_import_errors(dag_bag: DagBag) -> None:
    """Test for import errors."""
    assert not dag_bag.import_errors, f'DAG import failures. Errors: {dag_bag.import_errors}'
    assert len(dag_bag.dags) > 0, 'No DAGs found. Check the DAG folder path and ensure DAGs are defined correctly.'


def test_requires_tags(dag_bag: DagBag) -> None:
    """Tags should be defined for each DAG."""
    for dag in dag_bag.dags.values():
        assert dag.tags, 'DAG should have at least one tag defined.'


def test_owner_len_greater_than_five(dag_bag: DagBag) -> None:
    """Owner should be defined for each DAG and be longer than 5 characters."""
    for dag in dag_bag.dags.values():
        assert len(dag.owner) > 5, 'DAG owner should be longer than 5 characters.'


def test_desc_len_greater_than_fifteen(dag_bag: DagBag) -> None:
    """Description should be defined for each DAG and be longer than 30 characters."""
    for dag in dag_bag.dags.values():
        if isinstance(dag.description, str):
            assert len(dag.description) > 30


def test_owner_not_airflow(dag_bag: DagBag) -> None:
    """Owner should not be 'airflow'."""
    for dag in dag_bag.dags.values():
        assert str.lower(dag.owner) != 'airflow'


def test_three_or_less_retries(dag_bag: DagBag) -> None:
    """Retries should be 3 or less."""
    for dag in dag_bag.dags.values():
        assert dag.default_args['retries'] <= 3


def test_the_polars_chembl_steps_run_on_gce_not_dataproc() -> None:
    """A restore belongs on a single VM, not on a spark cluster.

    pts_step_from_config routes a step to dataproc as soon as one of its tasks is
    named 'pyspark ...'. These steps restore a dump and read it into polars;
    adding a pyspark task to any of them would move that onto a cluster whose
    workers would sit idle throughout.
    """
    config = UnifiedPipelineConfig()
    for name in (
        'pts_chembl_target_class_dump',
        'pts_drug_warning',
        'pts_drug_mechanism_of_action',
        'pts_chembl_molecule',
    ):
        step = pts_step_from_config(name, config)
        assert step.is_gce, f'{name} must run on GCE: it restores a database'
        assert not step.is_dataproc
