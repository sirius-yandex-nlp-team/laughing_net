from pathlib import Path

def find_parent_containing(filename: str, return_parent: bool = True):
    parent = Path(".").resolve()
    while not parent == parent.root:
        file_path = parent / filename
        if file_path.exists():
            if return_parent:
                return parent
            return file_path
        parent = parent.parent
    return None
