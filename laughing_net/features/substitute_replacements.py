import re
from pathlib import Path

import click
import pandas as pd

from laughing_net.context import ctx
from laughing_net.config import params

def process_row(row):
    text = row["original"]
    replacement = row["edit"]
    return re.sub("<(.*)/>", replacement, text)

def substitute(df):
    result = df[[]].copy()
    result["text"] = df.apply(process_row, axis=1)
    result["target"] = df["meanGrade"]
    return result

@click.command()
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option("--output", "output_path", type=click.Path(dir_okay=False, resolve_path=True))
def main(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    df = pd.read_csv(input_path, index_col="id")
    result = substitute(df)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)

if __name__ == "__main__":
    main()
