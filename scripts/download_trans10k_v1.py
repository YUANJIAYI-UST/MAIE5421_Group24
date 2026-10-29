"""Download the three author-linked Trans10K-v1 archives; never alter splits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import time
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests

SOURCE_PAGE = "https://xieenze.github.io/projects/TransLAB/TransLAB.html"
FILES = {
    "train": "1ZNTTbGjvPUHuLj3r3wEg8m12et7tDluz",
    "validation": "1I3BVEygFrCT3BmZL9byt2VFMt5ouCwRX",
    "test": "1A1mLgpOjioSQ1pzSUkU0Ce_3W_YozqgY",
}


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.action = None
        self.params = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.active = True
            self.action = attrs.get("action")
        elif self.active and tag == "input" and "name" in attrs:
            self.params[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


def archive_info(path, split, file_id):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if stream.read(4) != b"PK\x03\x04":
            raise ValueError(f"Not a ZIP archive: {path}")
        stream.seek(0)
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"split": split, "file": path.name, "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
            "source_url": f"https://drive.google.com/file/d/{file_id}/view"}


def download(split, file_id, output):
    path = output / f"{split}.zip"
    if path.exists():
        print(f"{split}: keeping existing archive", flush=True)
        return archive_info(path, split, file_id)
    partial = path.with_suffix(".zip.part")
    for attempt in range(3):
        try:
            with requests.Session() as session:
                response = session.get("https://drive.google.com/uc",
                    params={"export": "download", "id": file_id},
                    stream=True, timeout=(20, 60))
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", ""):
                    form = DownloadForm()
                    form.feed(response.text)
                    response.close()
                    if (not form.action or urlparse(form.action).scheme != "https"
                            or urlparse(form.action).hostname != "drive.usercontent.google.com"
                            or form.params.get("id") != file_id):
                        raise RuntimeError(f"{split}: no public download confirmation form; inspect access")
                    offset = partial.stat().st_size if partial.exists() else 0
                    response = session.get(form.action, params=form.params,
                        headers={"Range": f"bytes={offset}-"} if offset else {},
                        stream=True, timeout=(20, 60))
                else:
                    offset = 0
                with response:
                    response.raise_for_status()
                    if "text/html" in response.headers.get("Content-Type", ""):
                        raise RuntimeError(f"{split}: download returned HTML instead of ZIP")
                    append = offset > 0 and response.status_code == 206
                    if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Resume offset mismatch")
                    size = offset if append else 0
                    total = int(response.headers.get("Content-Length", 0)) + size
                    last_report = time.monotonic()
                    print(f"{split}: downloading; expected bytes={total or 'unknown'}", flush=True)
                    with partial.open("ab" if append else "wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            if not chunk:
                                continue
                            stream.write(chunk)
                            size += len(chunk)
                            if time.monotonic() - last_report >= 15:
                                print(f"{split}: {size / 2**20:.1f} MiB" +
                                      (f" / {total / 2**20:.1f} MiB" if total else ""), flush=True)
                                last_report = time.monotonic()
                    if total and size != total:
                        raise IOError(f"{split}: incomplete download: {size}/{total}")
                info = archive_info(partial, split, file_id)
                partial.rename(path)
                info["file"] = path.name
                print(f"{split}: complete, {info['bytes']} bytes, sha256={info['sha256']}", flush=True)
                return info
        except (requests.RequestException, OSError) as error:
            print(f"{split}: attempt {attempt + 1} failed: {type(error).__name__}", flush=True)
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/archives/Trans10K_v1")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    page = requests.get(SOURCE_PAGE, timeout=(20, 60))
    page.raise_for_status()
    for file_id in FILES.values():
        if file_id not in page.text:
            raise RuntimeError("Author page download links changed; review before downloading")
    (output / "author_source_page.html").write_bytes(page.content)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(download, split, file_id, output) for split, file_id in FILES.items()]
        archives = [future.result() for future in futures]
    record = {"recorded_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
              "source_page": SOURCE_PAGE, "archives": archives,
              "verification": "local SHA256 recorded; ZIP CRC and dataset audit are separate steps"}
    (output / "download_provenance.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
