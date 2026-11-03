"""Download the NASA SMAP / MSL telemetry anomaly dataset into ./data.

Sources, tried in order:
  1. The original Telemanom S3 bucket (often returns 403 Access Denied nowadays).
  2. The Kaggle mirror  patrickfleith/nasa-anomaly-detection-dataset-smap-msl
     (needs the `kaggle` CLI and an API token in ~/.kaggle/kaggle.json).
  3. A zip you downloaded by hand:  python scripts/download_data.py --zip path/to/archive.zip

The labels file comes from the Telemanom GitHub repo.

Result:
    data/labeled_anomalies.csv
    data/train/*.npy
    data/test/*.npy
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S3_URL = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
KAGGLE_SLUG = "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"


def fetch(url: str, dest: Path) -> bool:
    try:
        print(f"downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "sad-downloader"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  failed: {e}")
        return False


def find_split_dirs(root: Path) -> tuple[Path, Path] | None:
    """Locate a train/ and test/ folder pair containing .npy files anywhere under root."""
    for train in sorted(root.rglob("train"), key=lambda p: len(p.parts)):
        test = train.parent / "test"
        if train.is_dir() and test.is_dir() and any(train.glob("*.npy")) and any(test.glob("*.npy")):
            return train, test
    return None


def install_from_zip(zip_path: Path, out: Path) -> bool:
    if not zipfile.is_zipfile(zip_path):
        print(f"  {zip_path} is not a valid zip")
        return False
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(zip_path) as z:
            members = [m for m in z.namelist() if m.endswith(".npy") or m.endswith(".csv")]
            z.extractall(tmp, members=members)
        pair = find_split_dirs(tmp)
        if pair is None:
            print("  could not find train/ and test/ folders with .npy files in the archive")
            return False
        for split, src in zip(("train", "test"), pair):
            dst = out / split
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        csvs = list(tmp.rglob("labeled_anomalies.csv"))
        if csvs and not (out / "labeled_anomalies.csv").exists():
            shutil.copy(csvs[0], out / "labeled_anomalies.csv")
    return True


def try_kaggle(tmp: Path) -> Path | None:
    exe = shutil.which("kaggle")
    if exe is None:
        print("  kaggle CLI not found (pip install kaggle, then put your API token in ~/.kaggle/kaggle.json)")
        return None
    print(f"downloading Kaggle dataset {KAGGLE_SLUG}")
    r = subprocess.run([exe, "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(tmp)])
    zips = list(tmp.glob("*.zip"))
    return zips[0] if r.returncode == 0 and zips else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--zip", help="install from an archive you already downloaded")
    ap.add_argument("--source", choices=["auto", "s3", "kaggle"], default="auto")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ok = False
    if a.zip:
        ok = install_from_zip(Path(a.zip), out)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            if a.source in ("auto", "s3"):
                z = tmp / "data.zip"
                ok = fetch(S3_URL, z) and install_from_zip(z, out)
            if not ok and a.source in ("auto", "kaggle"):
                z = try_kaggle(tmp)
                ok = z is not None and install_from_zip(z, out)

    if not fetch(LABELS_URL, out / "labeled_anomalies.csv") and not (out / "labeled_anomalies.csv").exists():
        print("could not download labeled_anomalies.csv")
        ok = False

    if not ok:
        print("\nAutomatic download failed. Download the zip manually from\n"
              f"  https://www.kaggle.com/datasets/{KAGGLE_SLUG}\n"
              "and run:  python scripts/download_data.py --zip ~/Downloads/archive.zip")
        return 1
    n_train = len(list((out / "train").glob("*.npy")))
    n_test = len(list((out / "test").glob("*.npy")))
    print(f"done: {n_train} train / {n_test} test channels in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
