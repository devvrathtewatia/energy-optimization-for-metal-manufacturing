"""Fetch the REAL public dataset (CFAA milling tests, Zenodo 10.5281/zenodo.14445879, CC BY 4.0) and verify md5.
Already shipped in data/raw/; run this only to re-download.   python -m src.download_real"""
import hashlib
import urllib.request

from . import config as C

BASE = "https://zenodo.org/api/records/14445879/files/{}/content"
FILES = {"GMTK_dataset.csv": "f9bd66fb064a8c91b4a3131681ee0ef8", "IBARMIA_dataset.csv": "47c4c409c9b8670d411e6a4e2291376e"}


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    for name, want in FILES.items():
        path = C.DATA_RAW / name
        if not path.exists():
            urllib.request.urlretrieve(BASE.format(name), path)
        got = md5(path)
        print(f"{name}: md5 {'OK' if got == want else 'MISMATCH ' + got}")
        assert got == want


if __name__ == "__main__":
    main()
