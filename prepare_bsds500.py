"""Download and extract the official Berkeley BSDS500 image splits."""

from pathlib import Path
import tarfile
import urllib.request


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARCHIVE = DATA_DIR / "BSR_bsds500.tgz"
DATASET_DIR = DATA_DIR / "BSDS500"
URL = ("https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/"
       "grouping/BSR/BSR_bsds500.tgz")
IMAGE_PREFIX = "BSR/BSDS500/data/images/"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        print(f"Downloading BSDS500 from {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE)

    with tarfile.open(ARCHIVE, "r:gz") as archive:
        image_members = [
            member for member in archive.getmembers()
            if member.isfile() and member.name.startswith(IMAGE_PREFIX)
            and member.name.lower().endswith(".jpg")
        ]
        if len(image_members) != 500:
            raise RuntimeError(f"Expected 500 split images; found {len(image_members)}")
        for member in image_members:
            # Extract only the expected image tree, preventing unrelated archive
            # paths from being written to the workspace.
            destination = (DATASET_DIR / member.name).resolve()
            if not destination.is_relative_to(DATASET_DIR.resolve()):
                raise RuntimeError(f"Unsafe archive path: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read archive member: {member.name}")
            destination.write_bytes(source.read())

    image_root = DATASET_DIR / "BSR" / "BSDS500" / "data" / "images"
    counts = {split: len(list((image_root / split).glob("*.jpg")) )
              for split in ("train", "val", "test")}
    if counts != {"train": 200, "val": 100, "test": 200}:
        raise RuntimeError(f"Unexpected BSDS500 split sizes: {counts}")
    print(f"Ready: {image_root}; splits={counts}")


if __name__ == "__main__":
    main()
