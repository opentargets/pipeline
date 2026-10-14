import pkgutil
from collections import Counter
from importlib import resources

from pps.core import main, task_packages


def test_main_is_importable():
    assert callable(main)


def test_task_packages_include_datasource_tasks():
    assert task_packages() == ['pps.tasks', 'pps.gwas_catalog.tasks']


def test_task_types_are_unique():
    """Otter keys tasks by module name, so a duplicate would silently override another task."""
    task_types = Counter(
        m.name
        for package in task_packages()
        for m in pkgutil.iter_modules([str(resources.files(package))])
        if not m.ispkg
    )
    assert [t for t, n in task_types.items() if n > 1] == []
