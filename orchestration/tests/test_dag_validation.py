"""Try to parse and validate the DAGs."""

from airflow.models import DagBag


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


def _stage_jar_tasks(dag_bag: DagBag) -> list:
    dag = dag_bag.dags['unified_pipeline']
    return [t for t in dag.tasks if t.task_id.split('.')[-1].startswith('stage_jar_')]


def test_each_staged_jar_has_exactly_one_task(dag_bag: DagBag) -> None:
    """One staging task per destination object, shared by every cluster using it.

    Per-step staging produced ~60 tasks fetching the same ~629 MB jar, and
    per-cluster staging still produced two writers for the one object.
    """
    tasks = _stage_jar_tasks(dag_bag)
    destinations = [str(t.dst_uri) for t in tasks]
    assert destinations, 'expected the pts clusters to stage their Spark-NLP jar'
    assert len(destinations) == len(set(destinations)), (
        f'a destination is staged by more than one task: {sorted(destinations)}'
    )


def test_staged_jars_gate_every_step_of_the_clusters_that_use_them(dag_bag: DagBag) -> None:
    """A cluster must not be created before the jar it loads has been staged.

    Clusters that declare no ``spark.jars`` (pts_openfda, pts_association) stage
    nothing and are rightly ungated; but if any step of a cluster is gated, every
    step of that cluster must be — each step creates the cluster itself.
    """
    dag = dag_bag.dags['unified_pipeline']
    gated = {d for t in _stage_jar_tasks(dag_bag) for d in t.downstream_task_ids}
    assert gated, 'expected jar staging to gate some cluster creation'

    creates = [t.task_id for t in dag.tasks if t.task_id.split('.')[-1].startswith('create_cluster_')]
    staged_cluster_types = {t.split('create_cluster_')[-1] for t in gated}

    for task_id in creates:
        cluster_type = task_id.split('create_cluster_')[-1]
        if cluster_type in staged_cluster_types:
            assert task_id in gated, f'{task_id} skips the staging its cluster depends on'


def test_staged_jar_tasks_retry(dag_bag: DagBag) -> None:
    """Staging gates every PTS cluster, so it must not inherit retries=0.

    A single reset mid-transfer would otherwise fail every cluster creation.
    """
    for task in _stage_jar_tasks(dag_bag):
        assert task.retries > 0, f'{task.task_id} would take down the PTS stage on one hiccup'
