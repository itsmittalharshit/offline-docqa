import hashlib
import json
import sys
from pathlib import Path

import requests

SOURCES = Path("data_sources.json")
RAW_DIR = Path("data/raw")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verify_only = "--verify-only" in sys.argv
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    problems = 0

    for src in json.loads(SOURCES.read_text(encoding="utf-8")):
        path = RAW_DIR / src["file"]
        if path.exists() and sha256(path) == src["sha256"]:
            print(f"{src['file']}: ok")
            continue

        if verify_only:
            print(f"{src['file']}: missing or checksum mismatch")
            problems += 1
            continue

        print(f"{src['file']}: downloading")
        response = requests.get(src["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)

        if sha256(path) != src["sha256"]:
            path.unlink()
            sys.exit(f"{src['file']}: checksum mismatch — the site returned a different file "
                     f"(possibly a bot-protection page). Download it in a browser from: {src['url']}")
        print(f"{src['file']}: ok")

    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()