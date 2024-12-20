import zipfile

import requests


def download_file(url: str, dst_path: str):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def unzip_archive(src_path: str, dst_path: str):
    with zipfile.ZipFile(src_path, "r") as zip_ref:
        zip_ref.extractall(dst_path)
