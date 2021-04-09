import re
import pickle
from pathlib import Path

import click
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

from laughing_net.context import ctx
from laughing_net.config import params

def tfidf_fit_transform(text):
    tfidf = TfidfVectorizer(**params.features.tfidf)
    result = tfidf.fit_transform(text)
    return result, tfidf

def tfidf_transform(text, tfidf):
    result = tfidf.transform(text)
    return result

@click.command()
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option("--output", "output_path", type=click.Path(dir_okay=False, resolve_path=True))
@click.option("--fit", is_flag=True)
@click.option("--encoder", "encoder_path", type=click.Path(dir_okay=False, resolve_path=True))
def main(input_path, output_path, encoder_path, fit=False):
    assert fit or encoder_path is not None
    input_path = Path(input_path)
    output_path = Path(output_path)
    df = pd.read_csv(input_path, index_col="id")
    if fit:
        result, encoder = tfidf_fit_transform(df.text)
        pickle.dump(encoder, open(encoder_path, "wb"))
    else:
        encoder = pickle.load(open(encoder_path, "rb"))
        result = tfidf_transform(df.text, encoder)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output_path, result)

if __name__ == "__main__":
    main()
