"""Offline archive checks only: no imports of model code and no network calls."""
from pathlib import Path
import ast
import hashlib
import json
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
errors = []
manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
seen = set()
for item in manifest["files"]:
    rel = item["path"]
    p = ROOT / rel
    if rel in seen or not p.is_file() or p.is_symlink():
        errors.append(f"Missing, duplicate or symlinked source: {rel}")
        continue
    seen.add(rel)
    data = p.read_bytes()
    if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
        errors.append(f"Source digest mismatch: {rel}")

python_files = sorted(ROOT.rglob("*.py"))
for p in python_files:
    if ".git" in p.parts:
        continue
    try:
        ast.parse(p.read_text(), filename=str(p.relative_to(ROOT)))
    except (SyntaxError, UnicodeError) as exc:
        errors.append(f"Python parse failure: {p.relative_to(ROOT)}: {exc}")

links = 0
for p in ROOT.rglob("*.md"):
    if ".git" in p.parts:
        continue
    for raw in re.findall(r"\]\(([^)]+)\)", p.read_text()):
        target = raw.split("#", 1)[0].strip("<>")
        if not target or re.match(r"[a-zA-Z]+:", target):
            continue
        links += 1
        if not (p.parent / unquote(target)).exists():
            errors.append(f"Broken local link: {p.relative_to(ROOT)} -> {target}")

measurements = json.loads((ROOT / "results/measurements.json").read_text())
if measurements["status"] != "closed_without_target_achievement":
    errors.append("Archive status does not describe the recorded outcome")
for row in measurements["scores"]:
    for field in ["bpb", "core"]:
        if row.get(field) is not None and not isinstance(row[field], (int, float)):
            errors.append(f"Non-numeric metric: {row['id']} / {field}")

print(json.dumps({"source_files": len(seen), "python_files": len(python_files),
                  "local_links": links, "score_rows": len(measurements["scores"]),
                  "errors": errors}, indent=2))
sys.exit(bool(errors))
