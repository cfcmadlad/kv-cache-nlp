import re

from data_utils import REPO_ROOT


def main():
    parts = ["# Viva notes\n\nCollated from the `## Viva notes` section of each file in `docs/`. Regenerate with `python scripts/collate_viva.py`.\n"]
    for path in sorted((REPO_ROOT / "docs").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        m = re.search(r"^## Viva notes\n(.*?)(?=^## |\Z)", text, re.S | re.M)
        if m:
            title = text.splitlines()[0].lstrip("# ")
            parts.append(f"## {title} (`docs/{path.name}`)\n{m.group(1).rstrip()}\n")
    (REPO_ROOT / "docs" / "viva_notes.md").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
