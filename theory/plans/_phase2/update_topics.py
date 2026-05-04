#!/usr/bin/env python3
"""Update kb/index/topics.md status column based on actual note frontmatter.

For each leaf topic listed in topics.md, look up the note file and read
`status:` from its YAML frontmatter. If present and different, update the
status column. If file is missing, leave as `pending`.

Idempotent. Run from theory/ working directory.
"""
from __future__ import annotations
import re
from pathlib import Path

THEORY = Path(__file__).resolve().parents[2]
TOPICS = THEORY / "kb" / "index" / "topics.md"
NOTES_ROOT = THEORY / "kb" / "notes"


def read_status(note_path: Path) -> str | None:
    if not note_path.exists():
        return None
    text = note_path.read_text()
    if not text.startswith("---"):
        return None
    end = text.find("---", 3)
    if end == -1:
        return None
    fm = text[3:end]
    m = re.search(r"^status:\s*(\S+)", fm, re.MULTILINE)
    return m.group(1).strip() if m else None


ROW_RE = re.compile(
    r"^(\|\s*)(?P<topic>[a-z0-9-]+)(\s*\|\s*)(?P<status>[a-z]+)(\s*\|\s*`)(?P<path>kb/notes/[^`]+)(`\s*\|\s*)$"
)


def main() -> int:
    text = TOPICS.read_text()
    out_lines: list[str] = []
    changes = 0
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if not m:
            out_lines.append(line)
            continue
        topic = m.group("topic")
        old_status = m.group("status")
        rel = m.group("path")
        note = THEORY / rel
        actual = read_status(note)
        if actual is None:
            new_status = old_status  # leave pending if file missing
        else:
            new_status = actual
        if new_status != old_status:
            changes += 1
            print(f"  {topic}: {old_status} -> {new_status}")
            line = m.expand(
                r"\1\g<topic>\3" + new_status + r"\5\g<path>\7"
            )
        out_lines.append(line)
    TOPICS.write_text("\n".join(out_lines) + ("\n" if text.endswith("\n") else ""))
    print(f"\n{changes} status updates written to topics.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
