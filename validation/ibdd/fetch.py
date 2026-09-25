"""Download the authors' IBDD code and datasets (https://sites.google.com/view/ibdd-paper).

    python validation/ibdd/fetch.py code        # code_site.zip (30 KB)  -> validation/ibdd/_external/
    python validation/ibdd/fetch.py datasets    # IBDD_Datasets.zip (374 MB) -> datasets/ibdd/

Neither is committed: the code has no stated license and the datasets are large.
"""
import argparse
import os
import shutil
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

SOURCES = {
    "code": ("1ls1A-0d9pqbt6_ewUcjiPTefBpTT7s_E", "code_site.zip", os.path.join(HERE, "_external")),
    "datasets": ("1b_YVDl4pq3kTeiKIUg7GsVFlOoOGd8Qo", "IBDD_Datasets.zip", os.path.join(ROOT, "datasets", "ibdd")),
}

# confirm=t skips Google Drive's "can't scan this file for viruses" page on large files.
URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"


def fetch(what):
    file_id, zip_name, target = SOURCES[what]
    os.makedirs(target, exist_ok=True)
    zip_path = os.path.join(target, zip_name)
    if not os.path.exists(zip_path):
        print(f"Downloading {zip_name} ...")
        with urllib.request.urlopen(URL.format(file_id)) as response, open(zip_path + ".part", "wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
        os.replace(zip_path + ".part", zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)
    print(f"{zip_name} extracted to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=sorted(SOURCES) + ["all"])
    args = parser.parse_args()
    for what in (sorted(SOURCES) if args.what == "all" else [args.what]):
        fetch(what)
