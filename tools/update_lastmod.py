#!/usr/bin/env python3
"""Update the `lastmod` field in Markdown front matter to local current time."""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path


LASTMOD_PATTERN = re.compile(r"^(lastmod:\s*)(.+?)\s*$", re.MULTILINE)


def current_local_timestamp() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def update_lastmod(content: str, timestamp: str) -> tuple[str, bool]:
    match = LASTMOD_PATTERN.search(content)
    if not match:
        return content, False
    updated = LASTMOD_PATTERN.sub(
        lambda m: f"{m.group(1)}{timestamp}",
        content,
        count=1,
    )
    return updated, True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update front matter lastmod to local current time with timezone offset."
    )
    parser.add_argument("file", help="Path to markdown file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the new timestamp without writing changes",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}")
        return 1

    content = path.read_text(encoding="utf-8")
    timestamp = current_local_timestamp()
    new_content, changed = update_lastmod(content, timestamp)

    if not changed:
        print("Error: no `lastmod:` field found")
        return 1

    if args.dry_run:
        print(timestamp)
        return 0

    path.write_text(new_content, encoding="utf-8")
    print(f"Updated {path} -> lastmod: {timestamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
