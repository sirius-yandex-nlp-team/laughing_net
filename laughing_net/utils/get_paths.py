from typing import Tuple
from laughing_net.context import ctx
from laughing_net.config import params


def get_data_paths(task: str = 'task_1') -> Tuple[str]:
    paths = params.data[task]
    abs_paths = {k: str(ctx.root_dir / path) for k, path in paths.items()}
    return abs_paths
