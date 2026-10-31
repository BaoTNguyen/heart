"""Rebuild the "In force" table in this folder's README.md from each decision's
front matter: status, date, and binds when a decision has it. `--check` fails
instead of writing, for CI.

    python3 docs/adr/index.py [--check]
"""
# ponytail: identical copies in vascular, heart and plexus (docs/adr/index.py),
# because each repo's CI stands alone; share one once a fourth repo needs it.
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
START, END = "<!-- adr-index:start -->", "<!-- adr-index:end -->"


def read(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    front = re.match(r"---\n(.*?)\n---\n", text, re.S)
    title = front and re.search(r"^# (.+)$", text[front.end():], re.M)
    if not title:
        sys.exit(f"{path.name}: needs front matter and a '# title'")
    fields = {}
    for line in front.group(1).splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.split(" #")[0].strip()
    for key in ("status", "date"):
        if not fields.get(key):
            sys.exit(f"{path.name}: front matter has no {key}")
    if not path.name.startswith(fields["date"] + "-"):
        sys.exit(f"{path.name}: name must start with its date, {fields['date']}")
    return fields, title.group(1)


def table() -> str:
    rows = [(f, title, p.name) for p in sorted(HERE.glob("2*.md"), reverse=True)
            for f, title in [read(p)] if f["status"] == "accepted"]
    if not rows:
        return "(none yet)"
    binds = any(f.get("binds") for f, _, _ in rows)
    lines = ["| Decision | Date |" + (" Binds |" if binds else ""),
             "|---|---|" + ("---|" if binds else "")]
    for f, title, name in rows:
        lines.append(f"| [{title}]({name}) | {f['date']} |"
                     + (f" {f.get('binds', '')} |" if binds else ""))
    return "\n".join(lines)


def main() -> None:
    readme = HERE / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        sys.exit(f"{readme}: missing {START} / {END}")
    head, rest = text.split(START, 1)
    new = f"{head}{START}\n{table()}\n{END}{rest.split(END, 1)[1]}"
    if new == text:
        return
    if "--check" in sys.argv:
        sys.exit(f"{readme} is out of date: run python3 docs/adr/index.py")
    readme.write_text(new, encoding="utf-8")


if __name__ == "__main__":
    main()
