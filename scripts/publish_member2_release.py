"""Upload explicitly selected local assets to a draft GitHub Release, verify, then publish.

Authentication uses the existing Git Credential Manager entry in memory only.
No credential is printed, written to disk, or passed in a process argument.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import requests


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ProgressFile(io.BufferedReader):
    def __init__(self, path):
        super().__init__(io.FileIO(path, "r"))
        self.label, self.last = path.name, time.monotonic()

    def read(self, size=-1):
        chunk = super().read(size)
        if time.monotonic() - self.last > 30:
            print(f"Uploading {self.label}: {self.tell() / 2**20:.0f} MiB", flush=True)
            self.last = time.monotonic()
        return chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target", required=True, help="Pushed commit SHA")
    parser.add_argument("--title", required=True)
    parser.add_argument("--body-file", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--proxy")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 9))
    args = parser.parse_args()
    paths = sorted(p for p in Path(args.assets).iterdir() if p.is_file())
    if not paths or any(p.stat().st_size >= 2**31 for p in paths):
        raise ValueError("Release assets must be nonempty and each smaller than 2 GiB")
    records = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths]
    credential = subprocess.run(["git", "-c", "credential.helper=manager", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n", text=True, capture_output=True, check=True)
    fields = dict(line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line)
    token = fields.get("password")
    if not token:
        raise RuntimeError("No existing GitHub credential returned")
    session = requests.Session()
    # Use the explicitly requested route; do not silently inherit system proxies.
    session.trust_env = False
    session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                            "X-GitHub-Api-Version": "2022-11-28"})
    if args.proxy:
        session.proxies.update({"https": args.proxy, "http": args.proxy})
    api = f"https://api.github.com/repos/{args.repository}"
    thread_state = threading.local()

    def call(method, url, **kwargs):
        if not hasattr(thread_state, "session"):
            thread_state.session = requests.Session()
            thread_state.session.trust_env = session.trust_env
            thread_state.session.headers.update(session.headers)
            thread_state.session.proxies.update(session.proxies)
        response = thread_state.session.request(method, url, timeout=(30, 900), **kwargs)
        response.raise_for_status()
        return response.json() if response.content else None

    releases = call("GET", api + "/releases", params={"per_page": 100})
    release = next((r for r in releases if r["tag_name"] == args.tag), None)
    body = Path(args.body_file).read_text(encoding="utf-8")
    if release is None:
        release = call("POST", api + "/releases", json={"tag_name": args.tag,
            "target_commitish": args.target, "name": args.title, "body": body,
            "draft": True, "prerelease": True})
    elif not release["draft"]:
        raise ValueError("Existing release is published; refuse to mutate it")
    elif release["target_commitish"] != args.target:
        raise ValueError("Existing draft target differs")
    upload = release["upload_url"].split("{")[0]
    print(f"Draft Release {release['id']}", flush=True)

    def current_assets():
        result, page = [], 1
        while True:
            batch = call("GET", release["assets_url"], params={"per_page": 100, "page": page})
            result.extend(batch)
            if len(batch) < 100:
                return {a["name"]: a for a in result}
            page += 1

    def matches(asset, record):
        return (asset["state"] == "uploaded" and asset["size"] == record["bytes"]
                and asset.get("digest") == "sha256:" + record["sha256"])

    def upload_one(pair):
        path, record = pair
        for attempt in range(5):
            remote = current_assets().get(path.name)
            if remote:
                if matches(remote, record):
                    print(f"Verified {path.name}", flush=True)
                    break
                # GitHub records the declared Content-Length even for an aborted
                # upload. Only unfinished draft assets without a digest are retried.
                if remote["state"] == "starter" and remote.get("digest") is None:
                    call("DELETE", remote["url"])
                else:
                    raise ValueError(f"Existing asset differs: {path.name}")
            try:
                print(f"Starting {path.name} ({record['bytes'] / 2**20:.1f} MiB)", flush=True)
                with ProgressFile(path) as stream:
                    result = call("POST", upload, params={"name": path.name}, data=stream,
                                  headers={"Content-Type": "application/octet-stream"})
                if not matches(result, record):
                    raise ValueError(f"Uploaded asset digest mismatch: {path.name}")
                print(f"Verified {path.name}", flush=True)
                break
            except requests.RequestException as exc:
                print(f"Upload retry {attempt + 1}/5: {path.name} ({type(exc).__name__})", flush=True)
                if attempt == 4:
                    raise
                time.sleep(min(30, 2 ** attempt))
    # Every worker owns a distinct asset name. Each has its own HTTP session;
    # failed uploads can be retried without touching another worker's asset.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(upload_one, zip(paths, records)))
    remote = current_assets()
    if set(remote) != {r["name"] for r in records} or not all(matches(remote[r["name"]], r) for r in records):
        raise ValueError("Final release asset inventory mismatch")
    if args.publish:
        release = call("PATCH", release["url"], json={"draft": False, "body": body, "name": args.title})
    receipt = {"repository": args.repository, "tag": args.tag, "target": args.target,
        "url": release["html_url"], "draft": release["draft"], "assets": [
            {**record, "url": remote[record["name"]]["browser_download_url"],
             "github_digest": remote[record["name"]]["digest"]} for record in records]}
    Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"url": release["html_url"], "draft": release["draft"], "verified_assets": len(records)}))


if __name__ == "__main__":
    main()
