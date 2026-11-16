import subprocess
import sys


def test_registering_tasks_does_not_import_heavy_deps() -> None:
    script = (
        'import sys\n'
        'from otter.task.task_registry import TaskRegistry\n'
        "TaskRegistry(None, None).register('pts.tasks')\n"
        "heavy = {'openai', 'torch', 'transformers', 'mira', 'ontoma'}\n"
        "print(sorted({m.split('.')[0] for m in sys.modules} & heavy))\n"
    )
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '[]'
