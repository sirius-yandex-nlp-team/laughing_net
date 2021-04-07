import os
from pathlib import Path

from box import Box
from box.converters import yaml

from laughing_net.monitoring import report, shorten_path
from laughing_net.utils.fs import find_parent_containing

def _join_path(loader, node):
    seq = loader.construct_sequence(node)
    return Path().joinpath(*seq)

def _get_from_env(loader, node):
    return os.environ.get(node.value, None)

yaml.SafeLoader.add_constructor('!join_path', _join_path)
yaml.SafeLoader.add_constructor('!env', _get_from_env)

params = Box(box_dots=True)
config = Box(box_dots=True)

def load_configs():
    for name in ["params", "config"]:
        filename = find_parent_containing(f"{name}.yaml", return_parent=False)
        try:
            loaded = Box.from_yaml(
                filename=filename,
                box_dots=True,
                Loader=yaml.SafeLoader,
            )
            report("config", f"Loaded {name} from [!path]{shorten_path(filename)}[/]")
            globals()[name].merge_update(loaded)
        except Exception as e:
            report("config", f"Exception [!alert]{type(e).__name__}({e})[/] occured while parsing [!path]{shorten_path(filename)}[/]")

load_configs()
