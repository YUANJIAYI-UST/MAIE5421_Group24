"""Copy author-public V1 ZIPs to the authorized draft after exact archive/part checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests
from scripts.download_trans10k_v1 import FILES, download

REPOSITORY = "YUANJIAYI-UST/MAIE5421_Group24"
TAG = "trans10k-course-data-20261114"
RELEASE_ID = 445498999


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=FILES, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(Path("reports/member2_full_A_20261114/dataset_assets.json").read_text(encoding="utf-8"))
    expected = next(x for x in catalog["artifacts"] if x["archive"] == args.split + ".zip")
    info = download(args.split, FILES[args.split], output)
    if info["sha256"] != expected["sha256"] or info["bytes"] != expected["bytes"]:
        raise ValueError("Public archive differs from the frozen local archive; refusing upload")
    print(f"Author archive SHA256 matches frozen delivery: {args.split}", flush=True)

    session = requests.Session()
    session.headers.update({"Authorization": "Bearer " + os.environ["GH_TOKEN"],
        "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
    api = f"https://api.github.com/repos/{REPOSITORY}/releases/{RELEASE_ID}"

    def call(method, url, **kwargs):
        r = session.request(method, url, timeout=(30, 600), **kwargs)
        r.raise_for_status()
        return r.json() if r.content else None

    release = call("GET", api)
    if not release["draft"] or release["tag_name"] != TAG:
        raise ValueError("Expected the authorized, unpublished dataset release")
    upload_url = release["upload_url"].split("{")[0]
    receipts = []
    with (output / expected["archive"]).open("rb") as source:
        for part in expected["parts"]:
            chunk = source.read(part["bytes"])
            if len(chunk) != part["bytes"] or hashlib.sha256(chunk).hexdigest() != part["sha256"]:
                raise ValueError("Public archive part differs from frozen local part")
            path = (output / part["name"]).resolve()
            if not path.is_relative_to(output) or path.exists():
                raise ValueError("Unsafe/existing temporary part path")
            path.write_bytes(chunk)
            del chunk
            for attempt in range(4):
                assets = call("GET", release["assets_url"], params={"per_page": 100})
                asset = next((a for a in assets if a["name"] == part["name"]), None)
                if asset and asset["state"] == "uploaded":
                    if asset["size"] != part["bytes"] or asset.get("digest") != "sha256:" + part["sha256"]:
                        raise ValueError("Existing complete asset differs; refusing to replace it")
                    break
                if asset:
                    if asset["state"] != "starter" or asset.get("digest") is not None:
                        raise ValueError("Unexpected unfinished asset state")
                    # The local sender is stopped before this workflow starts.
                    call("DELETE", asset["url"])
                try:
                    with path.open("rb") as stream:
                        asset = call("POST", upload_url, params={"name": part["name"]}, data=stream,
                                     headers={"Content-Type": "application/octet-stream"})
                    if asset["size"] != part["bytes"] or asset.get("digest") != "sha256:" + part["sha256"]:
                        raise ValueError("Uploaded asset checksum mismatch")
                    break
                except requests.RequestException:
                    if attempt == 3:
                        raise
                    time.sleep(5 * (attempt + 1))
            receipts.append({"name": part["name"], "bytes": asset["size"], "digest": asset["digest"]})
            print(f"Verified public-to-release copy: {part['name']}", flush=True)
            path.unlink()
        if source.read(1):
            raise ValueError("Frozen part sizes do not cover archive")
    (output / f"{args.split}_receipt.json").write_text(json.dumps(receipts, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {args.split}: {len(receipts)} exact parts", flush=True)


if __name__ == "__main__":
    main()
