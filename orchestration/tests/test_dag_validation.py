"""Try to parse and validate the DAGs."""

from datetime import timedelta

from airflow.models import DagBag

from orchestration.utils import read_yaml_config


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


def test_decode_heavy_tasks_have_execution_timeout(dag_bag: DagBag) -> None:
    """The deCODE harmonisation and qc tasks must be time-bounded.

    A harmonisation run stalled for hours on 2026-07-02 and accrued ~£1k before
    anyone noticed. Airflow will not kill a task that is merely slow, so these
    tasks need an explicit ceiling. The downstream clumping and fine-mapping
    tasks operate on KiB-MiB and have never stalled, so they are left alone.
    """
    dag = dag_bag.dags['decode_ingestion']
    tasks = {t.task_id: t for t in dag.tasks}
    heavy = ['smp_harmonisation', 'smp_qc', 'raw_harmonisation', 'raw_qc']

    for task_id in heavy:
        assert task_id in tasks, f'{task_id} missing from the decode dag'
        timeout = tasks[task_id].execution_timeout
        assert timeout is not None, f'{task_id} has no execution_timeout'
        assert timeout == timedelta(hours=4), f'{task_id} timeout is {timeout}'


def _decode_config() -> dict:
    """Load the deCODE config the way the dag itself loads it."""
    from orchestration.utils import find_environment_vars

    path = 'src/orchestration/dags/config/decode_ingestion.yaml'
    raw = read_yaml_config(path)
    sentinels = find_environment_vars(raw['environment_specs'], raw['env'])
    return read_yaml_config(path, sentinels)


def test_decode_environments_pin_the_same_gentropy_ref() -> None:
    """Prod and Test must run identical gentropy code.

    The point of a Test run is to predict the Prod run. A ref that has drifted
    between the two makes the Test result meaningless.
    """
    raw = read_yaml_config('src/orchestration/dags/config/decode_ingestion.yaml')
    refs = {spec['name']: spec['vars']['gentropy_ref'] for spec in raw['environment_specs']}
    assert len(set(refs.values())) == 1, f'gentropy_ref differs between environments: {refs}'


def test_decode_keeps_sixteen_thousand_shuffle_partitions() -> None:
    """Shuffle partitions are deliberately not raised above 16000.

    Raising this was proposed, but the two dominant shuffles do not respond to
    it: the harmonisation clusters by studyId, whose cardinality (~4961) caps
    the number of non-empty partitions, and the write repartitions by the same
    key. A higher count only adds empty tasks. Locked here so the reasoning is
    not lost and the value is not raised by habit.
    """
    config = _decode_config()
    partitions = config['dataproc']['cluster_config']['properties']['spark:spark.sql.shuffle.partitions']
    assert partitions == '16000', f'shuffle partitions changed to {partitions}; see docstring'
