from box import Box

WIPE_DISKS = 'backend loads only into empty servers: wipe the ClickHouse and OpenSearch data disks, then rerun'


def get_config(config_file: str) -> Box:
    dataset_config = Box.from_yaml(filename=config_file)
    return Box(dataset_config, box_dots=True)
