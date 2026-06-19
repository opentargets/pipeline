from pathlib import Path


def pytest_ignore_collect(collection_path: Path, config):
    return 'orchestration/assets' in str(collection_path)
