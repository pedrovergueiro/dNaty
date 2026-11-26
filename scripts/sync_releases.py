"""Create the GitHub Release for every version tag that does not have one yet.

Notes come from the matching ``CHANGELOG.md`` section. Versions that never
reached PyPI get a notice saying so. Existing releases are left untouched.

Usage:
    python scripts/sync_releases.py --dry-run   # show what would be created
    python scripts/sync_releases.py             # needs the `gh` CLI and GH_TOKEN
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPI_URL = "https://pypi.org/pypi/dnaty/json"
TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
HEADER_RE = re.compile(r"^## \[(\d+\.\d+\.\d+)\](.*)$", re.MULTILINE)


def parse_changelog(text: str) -> dict[str, tuple[str, str]]:
    """Map version -> (release title, notes body)."""
    matches = list(HEADER_RE.finditer(text))
    sections = {}
    for i, match in enumerate(matches):
        version, rest = match.group(1), match.group(2)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end].strip().removesuffix("---").strip()
        _, dash, summary = rest.partition("—")
        title = f"v{version} — {summary.strip()}" if dash else f"v{version}"
        sections[version] = (title, body)
    return sections


def version_key(tag: str) -> tuple[int, ...]:
    return tuple(int(part) for part in TAG_RE.match(tag).groups())


def run(*cmd: str) -> str:
    return subprocess.run(
        cmd, cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def published_on_pypi() -> set[str]:
    with urllib.request.urlopen(PYPI_URL, timeout=30) as response:
        return set(json.load(response)["releases"])


def existing_releases() -> set[str]:
    out = run("gh", "release", "list", "--limit", "200", "--json", "tagName")
    return {item["tagName"] for item in json.loads(out)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without calling GitHub")
    args = parser.parse_args()

    tags = sorted((t for t in run("git", "tag").split() if TAG_RE.match(t)),
                  key=version_key)
    if not tags:
        print("No version tags found.")
        return 0

    sections = parse_changelog((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    on_pypi = published_on_pypi()
    done = set() if args.dry_run else existing_releases()

    for tag in tags:
        if tag in done:
            continue
        version = tag[1:]
        if version not in sections:
            print(f"skip {tag}: no CHANGELOG section")
            continue
        title, body = sections[version]
        if version not in on_pypi:
            body = (f"> **Not published to PyPI.** Its changes ship in later releases; "
                    f"install the current version with `pip install -U dnaty`.\n\n{body}")
        latest = "true" if tag == tags[-1] else "false"

        if args.dry_run:
            print(f"{tag}  latest={latest}  pypi={version in on_pypi}  "
                  f"notes={len(body)} chars  title={title!r}")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            notes = Path(tmp) / "notes.md"
            notes.write_text(body, encoding="utf-8")
            run("gh", "release", "create", tag, "--verify-tag", "--title", title,
                "--notes-file", str(notes), f"--latest={latest}")
        print(f"created {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
