import requests
import zipfile

from laughing_net.context import ctx
from laughing_net.config import params

def download(url, zip_path):
    response = requests.get(url, stream=True)

    zip_file = open(zip_path, "wb")
    for chunk in response.iter_content(chunk_size=1024):
        zip_file.write(chunk)

def unzip(zip_path, out_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out_path)

if __name__ == "__main__":
    zip_path = ctx.data_dir / "raw" / params.data.zip_file
    out_path = ctx.data_dir / "raw"
    download(params.data.url, zip_path)
    unzip(zip_path, out_path)
