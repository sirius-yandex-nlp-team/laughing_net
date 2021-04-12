import glob
import json
from pathlib import Path

import pandas as pd

SORT_ORDER_COLS = ['model', "transformer", "finetune", "train_extra", "feature", "task"]

def _create_name(row):
    name = str()
    if row["model"] == "cbow":
        name += "CBOW"
    else:
        if "roberta" in row["transformer"]:
            name += "RoBERTa"
        else:
            name += "BERT"
    if row["finetune"]:
        name += "+FT"
    if "orig" in row["feature"]:
        name += "+Orig"
    if row["train_extra"]:
        name += "+FunLines"
    return name

def _agg_metrics(d):
    result = pd.Series(dtype=float)
    result['Task 1 / RMSE'] = d[d["task"] == "task1"]["rmse"].values[0]
    result['Task 2 / Acc'] = d[d["task"] == "task2"]["accuracy"].values[0]
    result['Task 2 / Reward'] = d[d["task"] == "task2"]["reward"].values[0]
    for k in SORT_ORDER_COLS:
        result[k] = d[k].values[0]
    return result

def read_configs():
    entries = list()
    for path in glob.glob("external/table_repo/exp?/*"):
        path = Path(path)
        params = json.load(open(path / "params.json"))
        metrics = json.load(open(path / "metrics.json"))
        entries.append({**params, **metrics})
    df = pd.DataFrame.from_records(entries)
    return df

def format_table():
    df = read_configs()
    df["model_name"] = df.apply(_create_name, axis=1)
    (
        df
        .groupby("model_name")
        .apply(_agg_metrics)
        .sort_values("model")
        .sort_values(SORT_ORDER_COLS)
        .drop(SORT_ORDER_COLS, axis=1)
        .to_csv("outputs/table.csv")
    )

if __name__ == "__main__":
    format_table()
