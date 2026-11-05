"""Put one shader back to the module recorded in `.recompile-backup/`.

`recompile_shader.py --restore` only knows how to reset *all* four files, and it resets
recipes.json wholesale, which throws away every other shader's current entry. Reverting a
single shader by hand also fails in a specific way: once a shader has been recompiled its
recipe holds the new `raw_sha256`, so recompiling the pristine source with
`--decorations recorded` no longer reproduces the original module -- it either refuses or
installs a third variant. That happened to f01 on 2026-11-01 and invalidated a measurement.

So this tool reverts by *name*: the original .spv is still on disk, and `.recompile-backup/`
still holds the plan, manifest, allow-list and recipes as they were before the first
recompile. It repoints the four files back and splices the backup's recipe entry in verbatim
(line range, not a reformat, so later line-based patching keeps working).

    python tools/revert_shader.py --tag f01
    python tools/revert_shader.py --tag f13 --pick e8747d59afef
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recompile_shader as rs  # noqa: E402  (same directory, shared patch helpers)

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / ".recompile-backup"
PLAN = ROOT / "local/vk102/manifest/port_network_runtime.txt"
AUX = [ROOT / "local/vk102/manifest/port_network_runtime_1280x720.txt"]
MANIFEST = ROOT / "local/vk102/manifest/port_network_manifest.json"
FROZEN = ROOT / "local/vk102/generated/port_network_bridge_backend_frozen.inc"
RECIPES = ROOT / "source/network-fp16-20261027/recipes.json"

REF = re.compile(r'"\.\./assets/spv/([0-9a-f]{64})__([a-z0-9]+-[0-9a-f]+)\.spv"')


def pass_lines(path: Path) -> list[str]:
    return [l for l in path.read_text(encoding="utf-8").splitlines() if l.startswith("PASS ")]


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def recipe_block(path: Path, key: str) -> tuple[int, int]:
    """(first, last+1) line range of one entry, so it can be spliced without reformatting."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines) if l.strip().startswith(f'"{key}":')), None)
    if start is None:
        raise SystemExit(f"{path.name} has no entry {key!r}")
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("    }"))
    return start, end + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pick", default="", help="short hash after the tag, when it has variants")
    a = ap.parse_args()

    if not BACKUP.is_dir():
        raise SystemExit("no .recompile-backup; nothing to revert against")

    want = f"{a.tag}-{a.pick}.spv" if a.pick else None
    orig_lines = [l for l in pass_lines(BACKUP / "port_network_runtime.txt")
                  if f"__{a.tag}-" in l and (want is None or f"__{want}" in l)]
    if len({REF.search(l).group(0) for l in orig_lines}) != 1:
        raise SystemExit(f"backup has {len(orig_lines)} distinct modules for tag {a.tag}; pass --pick")

    index = pass_lines(BACKUP / "port_network_runtime.txt").index(orig_lines[0])
    orig_digest, orig_name = REF.search(orig_lines[0]).groups()
    orig_file = f"{orig_digest}__{orig_name}.spv"

    cur_digest, cur_name = REF.search(pass_lines(PLAN)[index]).groups()
    if cur_name == orig_name:
        print(f"{a.tag}: already at {orig_name}, nothing to do")
        return 0

    spv = ROOT / "local/vk102/assets/spv" / orig_file
    if not spv.is_file():
        raise SystemExit(f"original module {orig_file} is not on disk; cannot revert by name")

    cur_file = f"{cur_digest}__{cur_name}.spv"
    rs.patch_plan(PLAN, cur_file, orig_file)
    for aux in AUX:
        if aux.is_file():
            rs.patch_plan(aux, cur_file, orig_file)
    # patch_manifest keys on the manifest's own entry name, which carries the
    # `.spv` suffix (unlike the name inside a PASS line)
    rs.patch_manifest(MANIFEST, f"{cur_name}.spv", cur_digest, orig_digest, spv.stat().st_size)
    rs.patch_frozen(FROZEN, cur_file, orig_file, orig_digest,
                    sha_file(MANIFEST), sha_file(PLAN))

    # splice the backup's recipe entry in verbatim (same generator, same indentation)
    key = orig_name
    b_start, b_end = recipe_block(BACKUP / "recipes.json", key)
    c_start, c_end = recipe_block(RECIPES, key)
    backup_lines = (BACKUP / "recipes.json").read_text(encoding="utf-8").splitlines(keepends=True)
    cur_lines = RECIPES.read_text(encoding="utf-8").splitlines(keepends=True)
    RECIPES.write_text("".join(cur_lines[:c_start] + backup_lines[b_start:b_end] + cur_lines[c_end:]),
                       encoding="utf-8", newline="")

    print(f"{a.tag}: {cur_name} -> {orig_name}  ({spv.stat().st_size} bytes, recipe restored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
