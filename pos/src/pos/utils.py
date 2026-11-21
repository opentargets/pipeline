from box import Box

WIPE_DISKS = (
    "the backend steps load only into empty servers: wipe that server's data disk, "
    'or in Airflow trigger release_candidate again on fresh disks'
)


def get_config(config_file: str) -> Box:
    dataset_config = Box.from_yaml(filename=config_file)
    return Box(dataset_config, box_dots=True)
