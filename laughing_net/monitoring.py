from rich.console import Console
from rich.table import Table

_orange = "#F6BD4F"
_magenta = "#FD6CF9"
_red = "#FD6360"

_styles = {
    "!scope": f"bold {_orange}",
    "!path": f"bold {_magenta}",
    "!number": f"bold {_orange}",
    "!time": f"bold {_orange}",
    "!alert": f"bold {_red}",
}

_console = Console(highlight=False)

def _stylize(s: str):
    for shortcut, style in _styles.items():
        s = s.replace(shortcut, style)
    return s

def report(scope, message):
    scope = scope.upper().rjust(10)
    _console.print(_stylize(f"[!scope]{scope}[/]"), _stylize(message))

def report_table(name, table):
    title = f"[!table_title]{name.upper()}[/]"
    rich_table = Table(*table.columns, title=_stylize(title))
    for _, row in table.iterrows():
        rich_table.add_row(*map(str, row.values))
    _console.print(rich_table)

def shorten_path(path, len_limit=35, placeholder="[..]"):
    path = str(path)
    if len(path) > len_limit:
        split_point = 0
        for token in path.split("/")[::-1]:
            if split_point + len(token) + 1 < len_limit - len(placeholder):
                split_point += len(token) + 1
            else:
                break
        path = placeholder + path[-split_point:]
    return path
