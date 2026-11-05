"""Rebuild exactly one network shader and relink every place its hash is recorded.

Why this exists
---------------
`tools/build_assets.py` rebuilds *all* shaders *and* rewrites
`manifest/port_network_runtime.txt` from `source/<profile>/runtime.template.txt`.
That template is an old 118-dispatch plan; the plan on disk is the 232-dispatch one
that the 1280x720 baseline was measured with. So the documented asset path cannot
be used to change one kernel -- it would replace the plan we measure against.

This tool does the narrow thing instead: compile one shader, install it under its
new content hash, and patch the five places that record the old hash. Everything
else on disk is left byte-identical.

What gets patched
-----------------
1. `manifest/port_network_runtime.txt`   the `PASS` line's spv path
2. `manifest/port_network_manifest.json` `network_shaders` entry (key, bytes, sha)
3. `generated/..._frozen.inc`            the `FrozenFile` allow-list entry
4. `generated/..._frozen.inc`            `frozen_manifest_sha`  (hash of 2)
5. `generated/..._frozen.inc`            `frozen_runtime_sha`   (hash of 1)
6. `<profile>/recipes.json`              `raw_sha256`, `sha256`, `precision_decorations`

`port_network_bridge_backend.cpp` only checks those hashes (it never parses the
JSON), so this is the complete set.

About `precision_decorations`
-----------------------------
recipes records `raw_sha256`/`sha256` and a list of `OpDecorate ... NoContraction`
ops to insert at *instruction indices* of the raw module. glslang already emits
its own NoContraction ops for every `precise` expression (87 of them for the
f2c32 kernel), and the pipeline strips those and re-inserts a curated 69 at fixed
positions. Those positions are bound to the old module's instruction numbering, so
they cannot survive any source edit, and no tool in `tools/` can regenerate them.

So this tool does the simpler, stronger thing: it keeps glslang's own decorations
and records `precision_decorations: []`. Nothing is stripped, nothing is
position-bound, and every `precise` in the source is honoured. The result is a new,
self-consistent recipe -- not a byte-for-byte reproduction of the old one, which is
why the first thing to run after a no-op recompile is a control measurement: if the
unchanged source still produces the same picture, the decoration difference is
numerically inert and later deltas can be attributed to the actual change.

Usage
-----
    # 1) control: recompile with no source change, then re-measure the baseline
    python tools/recompile_shader.py --tag f19 --dry-run
    python tools/recompile_shader.py --tag f19 --glslang <glslang.exe>

    # 2) the real edit
    ...edit source/network-vk102/c32a_f2_transport/kernel.comp...
    python tools/recompile_shader.py --tag f19 --glslang <glslang.exe>

    # undo everything (restores the four files from .recompile-backup/)
    python tools/recompile_shader.py --tag f19 --restore
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / ".recompile-backup"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


# --------------------------------------------------------------------------------
# locate what to rebuild
# --------------------------------------------------------------------------------
def find_plan_line(plan_text: str, tag: str, pick: str = "") -> tuple[int, str, str, str]:
    """(line index, old full hash, old filename, the PASS line) for this tag."""
    # the file is named "<sha256>__<tag>-<sha256[:12]>.spv", so the tag is followed
    # by a short hash, not by ".spv"
    # Older assets in this tree predate the short-hash suffix and are simply
    # "<sha256>__<tag>.spv"; accept both spellings so a tag can be rebuilt before
    # and after it has been through this tool once.
    needles = (f"__{tag}-", f"__{tag}.spv")
    hits = []
    for i, line in enumerate(plan_text.splitlines()):
        if line.startswith("PASS ") and any(n in line for n in needles):
            m = re.search(r'"\.\./assets/spv/([0-9a-f]{64})__([^"]+)"', line)
            if m:
                hits.append((i, m.group(1), m.group(2), line))
    if pick:
        hits = [h for h in hits if h[2].endswith(f"-{pick}.spv")]
        if not hits:
            raise SystemExit(f"no PASS line references {tag}-{pick}.spv")
    if not hits:
        raise SystemExit(f"no PASS line references a spv tagged {tag!r}")
    if len(hits) > 1:
        names = {h[2] for h in hits}
        if len(names) > 1:
            raise SystemExit(f"tag {tag!r} is ambiguous: {sorted(names)}"
                             " -- pass --pick <short hash> to choose one")
    i, digest, name, line = hits[0]
    return i, digest, name, line


def find_recipe(recipes: dict, digest: str) -> tuple[str, dict]:
    for key, entry in recipes["shaders"].items():
        if entry.get("sha256") == digest:
            return key, entry
    raise SystemExit(f"no recipe has sha256 {digest}; is recipes.json in sync with the plan?")


# --------------------------------------------------------------------------------
# compile
# --------------------------------------------------------------------------------
def compile_one(compiler: Path, source: Path, defines: list[str], out: Path) -> bytes:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(compiler), "-V", "--target-env", "vulkan1.3", *defines,
           "-I" + str(source.parent), "-o", str(out), str(source)]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return out.read_bytes()


def apply_recorded_decorations(spv: bytes, decorations: list[dict]) -> bytes:
    """Reproduce build_assets.py's transformation: strip every NoContraction, then
    insert the recorded ones at their recorded instruction indices.

    This is only valid for the *unmodified* source, because the indices are bound to
    that module's instruction numbering. It exists so the tool can prove it still
    reproduces the deployed bytes before any source is touched.
    """
    words = struct.unpack("<%dI" % (len(spv) // 4), spv)
    header, ops, pos = list(words[:5]), [], 5
    while pos < len(words):
        n = words[pos] >> 16
        if not n or pos + n > len(words):
            raise SystemExit("invalid SPIR-V while stripping decorations")
        ops.append(tuple(words[pos:pos + n]))
        pos += n
    ops = [op for op in ops if not (op[0] & 0xFFFF == 71 and len(op) == 3 and op[2] == 42)]
    for entry in decorations:
        ops.insert(entry["position"], tuple(entry["words"]))
    flat = list(header) + [w for op in ops for w in op]
    return struct.pack("<%dI" % len(flat), *flat)


def count_decorations(spv: bytes) -> tuple[int, int]:
    """(ops, NoContraction decorations already present from glslang)."""
    words = struct.unpack("<%dI" % (len(spv) // 4), spv)
    pos, ops, nc = 5, 0, 0
    while pos < len(words):
        n = words[pos] >> 16
        if not n or pos + n > len(words):
            raise SystemExit("invalid SPIR-V while counting decorations")
        if (words[pos] & 0xFFFF) == 71 and n == 3 and words[pos + 2] == 42:
            nc += 1
        ops += 1
        pos += n
    return ops, nc


# --------------------------------------------------------------------------------
# line surgery -- all of these files have hand-stable formatting, so never
# re-serialise them; a json round-trip changes unrelated bytes and every hash.
# --------------------------------------------------------------------------------
def patch_plan(path: Path, old_file: str, new_file: str) -> None:
    text = path.read_text(encoding="utf-8")
    old_ref = f'"../assets/spv/{old_file}"'
    new_ref = f'"../assets/spv/{new_file}"'
    # one shader can serve many layers (f13 feeds all eight vit_*_ffn_expand layers),
    # so replace every occurrence -- but insist on at least one.
    n = text.count(old_ref)
    if n == 0:
        raise SystemExit(f"no reference to {old_file} in {path.name}")
    path.write_text(text.replace(old_ref, new_ref), encoding="utf-8", newline="\n")
    print(f"  {path.name}: {n} reference(s) repointed")


def patch_manifest(path: Path, old_key: str, old_digest: str, new_digest: str,
                   new_bytes: int, new_name: str) -> None:
    """Rewrite one `network_shaders` entry: key name, byte count, sha256, path.

    The entry is found by its `sha256`, not by its key. The key is rewritten to
    "<tag>-<short>.spv" on every rebuild, so after a shader has been rebuilt once its
    key no longer equals the plan's spv name; matching on the key therefore only works
    for the first rebuild and silently leaves the manifest stale for all later ones
    (that is how three f07 entries ended up with a 116-character sha and a duplicated
    digest in their path).
    """
    tag = old_key.split("-", 1)[0]
    # The manifest keys are the short spelling "<tag>-<short>.spv"; the plan names the same file
    # as "<sha64>__<tag>-<short>.spv". Passing the plan name here made the rewritten key contain
    # the digest prefix, which then failed the tag guard below -- so derive the key, not the name.
    new_key = f"{tag}-{new_digest[:12]}.spv"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    starts = [i for i, l in enumerate(lines) if f'"sha256": "{old_digest}"' in l]
    if len(starts) != 1:
        raise SystemExit(f'manifest has {len(starts)} network_shaders entries with sha256 '
                         f'{old_digest[:12]}, expected 1')
    sha_line = starts[0]
    # walk back to the entry's opening brace line
    start = next(i for i in range(sha_line, -1, -1) if lines[i].rstrip("\n").endswith("{"))
    indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
    lines[start] = f'{indent}"{new_key}": {{\n'

    touched = 0
    # scan from the entry's opening brace, not from the sha256 line: this manifest writes
    # "bytes" *before* "sha256", so starting at the sha line left `bytes` stale and the
    # `touched != 3` guard then aborted after the plan had already been repointed.
    for i in range(start + 1, min(start + 8, len(lines))):
        if '"bytes"' in lines[i]:
            lines[i] = re.sub(r'"bytes":\s*\d+', f'"bytes": {new_bytes}', lines[i])
            touched += 1
        elif '"sha256"' in lines[i]:
            lines[i] = re.sub(r'"[0-9a-f]{64}"', f'"{new_digest}"', lines[i])
            touched += 1
        elif '"path"' in lines[i]:
            lines[i] = re.sub(r'"\.\./assets/spv/[^"]*"', f'"../assets/spv/{new_name}"', lines[i])
            touched += 1
        elif lines[i].strip() in ("}", "},"):
            break
    if touched != 3:
        raise SystemExit(f"expected bytes/sha256/path lines, patched {touched}")
    if tag and not new_key.startswith(tag):
        raise SystemExit(f"rewritten key {new_key!r} does not start with tag {tag!r}")
    path.write_text("".join(lines), encoding="utf-8", newline="")


def patch_frozen(path: Path, old_file: str, new_file: str, new_digest: str,
                 manifest_sha: str, runtime_sha: str) -> None:
    """Swap the spv allow-list entry and re-pin the two file hashes.

    The allow-list stores the *installed filename* (hash + "__" + tag + short hash),
    not the manifest key, so it has to be matched on `old_file`.
    """
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    hits = [i for i, l in enumerate(lines) if f'../assets/spv/{old_file}"' in l]
    if len(hits) != 1:
        # the allow-list can still carry an older filename for the same bytes (the plan is
        # repointed before this runs), so fall back to the digest both names start with
        digest = old_file.split("__")[0]
        hits = [i for i, l in enumerate(lines) if f'../assets/spv/{digest}__' in l]
    if len(hits) != 1:
        raise SystemExit(f"expected 1 allow-list entry for {old_file}, found {len(hits)}")
    i = hits[0]
    lines[i] = lines[i].replace(
        f'../assets/spv/{old_file}"', f'../assets/spv/{new_file}"').replace(
        old_file.split("__")[0], new_digest)

    replaced = 0
    for j, l in enumerate(lines):
        if l.startswith("static constexpr const char* frozen_manifest_sha="):
            lines[j] = f'static constexpr const char* frozen_manifest_sha="{manifest_sha}";\n'
            replaced += 1
        elif l.startswith("static constexpr const char* frozen_runtime_sha="):
            lines[j] = f'static constexpr const char* frozen_runtime_sha="{runtime_sha}";\n'
            replaced += 1
    if replaced != 2:
        raise SystemExit(f"expected manifest+runtime sha lines, found {replaced}")
    path.write_text("".join(lines), encoding="utf-8", newline="")


def patch_recipes(path: Path, recipe_key: str, raw_sha: str, final_sha: str) -> str:
    """Rewrite one entry's three hash fields; returns a one-line description."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    key_line = next((i for i, l in enumerate(lines)
                     if l.strip().startswith(f'"{recipe_key}":')), None)
    if key_line is None:
        raise SystemExit(f"recipes.json has no entry {recipe_key!r}")

    end = next((i for i in range(key_line + 1, len(lines))
                if lines[i].startswith("    }")), len(lines))
    dropped = 0
    out = []
    i = key_line + 1
    while i < end:
        l = lines[i]
        if l.lstrip().startswith('"sha256"'):
            out.append(re.sub(r'"[0-9a-f]{64}"', f'"{final_sha}"', l))
        elif l.lstrip().startswith('"raw_sha256"'):
            out.append(re.sub(r'"[0-9a-f]{64}"', f'"{raw_sha}"', l))
        elif l.lstrip().startswith('"precision_decorations"'):
            # swallow the whole array, however many lines it spans
            depth = l.count("[") - l.count("]")
            dropped += 1
            i += 1
            while depth > 0 and i < len(lines):
                depth += lines[i].count("[") - lines[i].count("]")
                dropped += 1
                i += 1
            indent = " " * (len(l) - len(l.lstrip()))
            out.append(f'{indent}"precision_decorations": []\n')
            continue
        else:
            out.append(l)
        i += 1
    lines[key_line + 1:end] = out
    # Guard against a bad `end` span: once, every entry in this file came back holding the
    # same hash because the block ran to EOF and the regex rewrote all of them. Exactly one
    # entry may change; anything else means the span was wrong, so put the bytes back.
    pristine = path.read_text(encoding="utf-8")
    before = {k: v.get("sha256") for k, v in json.loads(pristine)["shaders"].items()}
    path.write_text("".join(lines), encoding="utf-8", newline="")
    after = json.loads(path.read_text(encoding="utf-8"))["shaders"]
    touched = [k for k in before if before[k] != after[k].get("sha256")]
    if touched != [recipe_key]:
        path.write_text(pristine, encoding="utf-8", newline="")
        raise SystemExit(f"patch_recipes would touch {len(touched)} entries, not 1 -- recipes.json left "
                         f"unchanged (span was {key_line + 1}..{end})")
    return f"{recipe_key}: decorations array collapsed ({dropped} lines -> [])"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="spv tag as it appears in the plan, e.g. f19")
    ap.add_argument("--pick", default="",
                    help="disambiguate when one tag has several variants in the plan, e.g. "
                         "f13 feeds both the ffn expand and the ffn contract; pass the short "
                         "hash that follows the tag in the filename")
    ap.add_argument("--model", default="local/vk102")
    ap.add_argument("--glslang", default="", help="path to glslang.exe (required unless --restore)")
    ap.add_argument("--source", default="", help="override the source path from recipes.json")
    ap.add_argument("--also-plan", action="append", default=[], metavar="PATH",
                    help="another plan that references the same spv and must be repointed too, "
                         "e.g. the copy a measurement harness was pinned to. Not hash-checked, "
                         "so nothing else has to be re-pinned for it.")
    ap.add_argument("--decorations", choices=("glslang", "recorded"), default="glslang",
                    help="glslang: keep the compiler's own NoContraction (survives a source edit). "
                         "recorded: reproduce recipes.json's exact layout (only valid for an "
                         "unmodified source; this is the tool's self-check).")
    ap.add_argument("--dry-run", action="store_true", help="compile and report, patch nothing")
    ap.add_argument("--restore", action="store_true", help="restore the four files from the backup")
    a = ap.parse_args()

    model = (ROOT / a.model).resolve()
    manifest_dir = model / "manifest"
    generated = model / "generated"
    plan = manifest_dir / "port_network_runtime.txt"
    manifest = manifest_dir / "port_network_manifest.json"
    frozen = generated / "port_network_bridge_backend_frozen.inc"

    profile = json.loads((model / "model/extraction-receipt.json").read_text(encoding="utf-8"))["profile"]
    profile_dir = ROOT / json.loads((ROOT / "tools/model_profiles.json").read_text(encoding="utf-8"))[profile]["network_dir"]
    recipes_path = profile_dir / "recipes.json"

    for p in (plan, manifest, frozen, recipes_path):
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    if a.restore:
        if not BACKUP.is_dir():
            raise SystemExit(f"no backup at {BACKUP}")
        for p in (plan, manifest, frozen, recipes_path):
            b = BACKUP / p.name
            if b.is_file():
                shutil.copyfile(b, p)
                print(f"restored {p.relative_to(ROOT)}")
        return 0

    if not a.glslang:
        raise SystemExit("--glslang is required unless --restore")

    plan_text = plan.read_text(encoding="utf-8")
    line_no, old_digest, old_name, _ = find_plan_line(plan_text, a.tag, a.pick)
    print(f"plan line {line_no + 1}: {old_name}  ({old_digest[:12]})")

    recipes = json.loads(recipes_path.read_text(encoding="utf-8"))
    recipe_key, entry = find_recipe(recipes, old_digest)
    source = ROOT / (a.source or entry["source"])
    if not source.is_file():
        raise SystemExit(f"source not found: {source}")
    # `--source` may legitimately point outside ROOT (an experiment tree), where relative_to
    # raises; the path is only informational, so fall back to the absolute form.
    try:
        shown = source.relative_to(ROOT)
    except ValueError:
        shown = source
    print(f"recipe {recipe_key}  source {shown}")

    out = ROOT / ".tmpbuild" / f"{a.tag}.raw.spv"
    spv = compile_one(Path(a.glslang).resolve(), source, entry.get("defines", []), out)
    raw_digest = sha_bytes(spv)
    if a.decorations == "recorded":
        if raw_digest != entry.get("raw_sha256"):
            raise SystemExit(
                "the source no longer compiles to recipes.json's raw_sha256, so the recorded "
                "decoration indices no longer line up; use --decorations glslang for an edit")
        spv = apply_recorded_decorations(spv, entry.get("precision_decorations", []))
        if sha_bytes(spv) != entry.get("sha256"):
            raise SystemExit("reproduced the decorations but not recipes.json's sha256")
        print("recorded layout reproduced exactly (self-check passed)")
    new_digest = sha_bytes(spv)
    new_name = f"{new_digest}__{a.tag}-{new_digest[:12]}.spv"
    ops, nc = count_decorations(spv)
    print(f"compiled: {len(spv)} bytes, {ops} ops, {nc} NoContraction kept from glslang")
    print(f"new name: {new_name}")

    if new_digest == old_digest:
        print("\nnothing changed -- the source compiles to the bytes already installed")
        return 0
    if nc == 0:
        print("\nWARNING: the module carries no NoContraction at all; `precise` may be missing "
              "and the compiler is free to contract the rounding away", file=sys.stderr)

    if a.dry_run:
        print("\ndry run: no files touched")
        return 0

    # snapshot once: a second run must not overwrite the pristine copy with an
    # already-patched file, or --restore would restore the edit
    BACKUP.mkdir(exist_ok=True)
    for p in (plan, manifest, frozen, recipes_path):
        dest_backup = BACKUP / p.name
        if not dest_backup.exists():
            shutil.copyfile(p, dest_backup)

    # install under the new hash first: if a later patch fails, the plan still points
    # at the old file and the build stays valid.
    dest = model / "assets/spv" / new_name
    dest.write_bytes(spv)
    print(f"installed {dest.relative_to(ROOT)}")

    old_file = f"{old_digest}__{old_name}"
    patch_plan(plan, old_file, new_name)
    for extra in a.also_plan:
        patch_plan(ROOT / extra, old_file, new_name)
    patch_manifest(manifest, old_name, old_digest, new_digest, len(spv), new_name)
    # the two sha lines must be recomputed from the bytes actually on disk now
    patch_frozen(frozen, f"{old_digest}__{old_name}", new_name, new_digest,
                 sha_file(manifest), sha_file(plan))
    note = patch_recipes(recipes_path, recipe_key, new_digest, new_digest)

    print(f"\npatched: plan, manifest, frozen.inc (3 lines), recipes.json")
    print(f"  {note}")
    print("\nnext: rebuild (tools\\build-backend-local.cmd vk102) and re-measure")
    print("      the old .spv is left on disk; `--restore` reverts the four text files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
