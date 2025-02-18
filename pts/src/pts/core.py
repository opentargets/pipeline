from otter import Runner


def main() -> None:
    runner = Runner(name='pts')
    runner.start()
    runner.register_tasks('pts.tasks')
    runner.run()
