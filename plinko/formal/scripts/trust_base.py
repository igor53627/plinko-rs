#!/usr/bin/env python3
"""Writes docs/TRUST_BASE.md from the Rocq sources and docs/trust_base_notes.toml.

The script lists every `Axiom`, `Parameter`, `Conjecture` and `Admitted` of the
files below. Each item must have a note (class and meaning) in the notes file,
and each note must belong to an item.

    python3 scripts/trust_base.py           # write docs/TRUST_BASE.md
    python3 scripts/trust_base.py --check   # exit 1 if the file is out of date

Run it from `plinko/formal`. It needs Python 3.11 or later (tomllib).

Limit: the script does not find a `Variable` or `Hypothesis` that is outside a
section. Rocq treats such a declaration as an axiom and prints a warning.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import tomllib
from dataclasses import dataclass

FORMAL_DIR = pathlib.Path(__file__).resolve().parent.parent
NOTES_PATH = FORMAL_DIR / "docs" / "trust_base_notes.toml"
OUTPUT_PATH = FORMAL_DIR / "docs" / "TRUST_BASE.md"

# (path relative to plinko/formal, how the files are checked)
SOURCES: list[tuple[str, str]] = [
    ("specs", "built by `make all`"),
    ("sims", "built by `make all`"),
    ("proofs", "built by `make all`"),
    ("linking", "not built by `make all` (needs RocqOfRust)"),
    ("../../docs/Plinko.v", "sketch, built by `make sketch`"),
]

CLASSES: dict[str, str] = {
    "symbol": "Uninterpreted symbol. It has no definition; a statement about it holds for every function of that type.",
    "math": "True statement that can be proven. It is not a statement about the code.",
    "trivial": "Statement of the form `x = x`. It adds nothing.",
    "link": "States that the Rust code equals the model. Nothing checks it.",
    "ideal": "False for the real primitive as stated. It holds only for an idealized function.",
    "interface": "Field of a `Module Type`. It is a requirement on a module, not a global assumption.",
    "open": "Statement with no proof (`Admitted`).",
}

# A keyword starts a command: it is not a part of a longer name or of a qualified name.
COMMAND_START = r"(?<![\w.'])"
IDENT = r"[A-Za-z_][\w']*"
DECLARATION = re.compile(COMMAND_START + r"(Axiom|Axioms|Parameter|Parameters|Conjecture|Conjectures)\s+(?=[A-Za-z_(])")
PROOF_HEAD = re.compile(
    COMMAND_START
    + r"(Lemma|Theorem|Corollary|Proposition|Fact|Remark|Example|Definition|Fixpoint|Instance)\s+("
    + IDENT
    + r")"
)
ADMITTED = re.compile(COMMAND_START + r"Admitted\s*\.(?=\s|$)")
SENTENCE_END = re.compile(r"\.(?=\s|$)")


@dataclass(frozen=True)
class Item:
    file: str
    kind: str
    name: str
    statement: str
    offset: int


def strip_comments(text: str) -> str:
    """Replaces Rocq comments (which nest) with spaces; keeps the offsets."""
    out: list[str] = []
    depth = 0
    i = 0
    while i < len(text):
        pair = text[i : i + 2]
        if pair == "(*":
            depth += 1
            out.append("  ")
            i += 2
        elif pair == "*)" and depth > 0:
            depth -= 1
            out.append("  ")
            i += 2
        else:
            char = text[i]
            out.append(char if depth == 0 or char == "\n" else " ")
            i += 1
    return "".join(out)


def tidy(statement: str) -> str:
    lines = [line.rstrip() for line in statement.strip().splitlines()]
    return "\n".join(lines)


def declared_names(body: str) -> list[str]:
    """Names of one declaration: `a b : T.` or `(a : T) (b c : U).`"""
    if body.lstrip().startswith("("):
        groups = re.findall(r"\(\s*((?:" + IDENT + r"\s*)+):", body)
        return [name for group in groups for name in group.split()]
    head = body.split(":", 1)[0]
    return re.findall(IDENT, head)


def items_of_file(path: pathlib.Path, label: str) -> list[Item]:
    code = strip_comments(path.read_text(encoding="utf-8"))
    items: list[Item] = []

    for match in DECLARATION.finditer(code):
        end = SENTENCE_END.search(code, match.end())
        stop = end.end() if end else len(code)
        statement = tidy(code[match.start() : stop])
        names = declared_names(code[match.end() : stop])
        if not names:
            raise SystemExit(f"{label}: cannot read the names of: {statement[:60]}")
        for name in names:
            items.append(Item(label, match.group(1), name, statement, match.start()))

    heads = list(PROOF_HEAD.finditer(code))
    for match in ADMITTED.finditer(code):
        before = [head for head in heads if head.start() < match.start()]
        if not before:
            raise SystemExit(f"{label}: `Admitted` with no statement before it")
        head = before[-1]
        proof = re.search(r"^[ \t]*Proof\b", code[head.start() : match.start()], re.M)
        stop = head.start() + proof.start() if proof else match.start()
        items.append(Item(label, "Admitted", head.group(2), tidy(code[head.start() : stop]), head.start()))

    return sorted(items, key=lambda item: item.offset)


def collect_items() -> list[tuple[str, str, list[Item]]]:
    """Returns (file label, check mode, items) for every source file, in a fixed order."""
    result: list[tuple[str, str, list[Item]]] = []
    for source, mode in SOURCES:
        path = (FORMAL_DIR / source).resolve()
        files = sorted(path.glob("*.v")) if path.is_dir() else [path]
        for file in files:
            if not file.is_file():
                raise SystemExit(f"missing source file: {file}")
            label = f"{source}/{file.name}" if path.is_dir() else "docs/Plinko.v"
            result.append((label, mode, items_of_file(file, label)))
    return result


def render(notes: dict, sources: list[tuple[str, str, list[Item]]]) -> tuple[str, list[str]]:
    errors: list[str] = []
    note_of: dict[tuple[str, str], dict] = {}
    for note in notes.get("item", []):
        key = (note["file"], note["name"])
        if key in note_of:
            errors.append(f"two notes for {key[0]}: {key[1]}")
        note_of[key] = note
        if note.get("class") not in CLASSES:
            errors.append(f"{key[0]}: {key[1]}: unknown class {note.get('class')!r}")

    seen: set[tuple[str, str]] = set()
    counts: dict[tuple[str, str], int] = {}
    body: list[str] = []

    for label, mode, items in sources:
        if not items:
            continue
        body.append(f"## {label}\n")
        body.append(f"Checked: {mode}.\n")
        for item in items:
            key = (item.file, item.name)
            seen.add(key)
            note = note_of.get(key)
            if note is None:
                errors.append(f"no note for {item.kind} `{item.name}` in {item.file}")
                continue
            counts[(mode, note["class"])] = counts.get((mode, note["class"]), 0) + 1
            body.append(f"### {item.name}\n")
            body.append(f"- Kind: `{item.kind}`")
            body.append(f"- Class: {note['class']}")
            body.append(f"- Meaning: {note['meaning'].strip()}")
            if note.get("model"):
                body.append(f"- Model that satisfies it: {note['model'].strip()}")
            body.append("")
            body.append("```coq")
            body.append(item.statement)
            body.append("```\n")

    for key in sorted(set(note_of) - seen):
        errors.append(f"note without an item: {key[0]}: {key[1]}")

    head: list[str] = [
        "# Trust Base Of The Formal Model\n",
        "<!-- Generated by scripts/trust_base.py from the .v files and docs/trust_base_notes.toml. Do not edit. -->\n",
        notes.get("intro", "").strip() + "\n",
        "## Classes\n",
    ]
    for name, text in CLASSES.items():
        head.append(f"- **{name}**: {text}")
    head.append("")
    head.append("## Counts\n")
    modes = [mode for _, mode in SOURCES]
    unique_modes = list(dict.fromkeys(modes))
    rows: list[list[str]] = [["Class", *unique_modes]]
    for name in CLASSES:
        rows.append([name, *(str(counts.get((mode, name), 0)) for mode in unique_modes)])
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    for index, row in enumerate(rows):
        head.append("| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |")
        if index == 0:
            head.append("|" + "|".join("-" * (width + 2) for width in widths) + "|")
    head.append("")

    return "\n".join(head + body).rstrip() + "\n", errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 if TRUST_BASE.md is out of date")
    args = parser.parse_args()

    notes = tomllib.loads(NOTES_PATH.read_text(encoding="utf-8"))
    text, errors = render(notes, collect_items())
    if errors:
        for error in errors:
            print(f"trust_base: {error}", file=sys.stderr)
        return 1

    if args.check:
        current = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
        if current != text:
            print("trust_base: docs/TRUST_BASE.md is out of date; run `make trust-base`", file=sys.stderr)
            return 1
        print("trust_base: docs/TRUST_BASE.md is up to date")
        return 0

    OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(f"trust_base: wrote {OUTPUT_PATH.relative_to(FORMAL_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
