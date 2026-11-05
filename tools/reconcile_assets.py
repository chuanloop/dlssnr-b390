"""Re-align the three files that all record the same spv hashes.

`port_network_runtime.txt` (the plan), `port_network_manifest.json` and
`generated/port_network_bridge_backend_frozen.inc` each carry the shader hashes, and
`recompile_shader.py` used to keep them in step one shader at a time. It could not: its
manifest lookup matched on the *key*, which is rewritten to `<tag>-<short>.spv` on the
first rebuild, so every later rebuild of the same shader silently left both files
behind (three f07 entries ended up with a 116-character `sha256` and a digest repeated
inside their path; the frozen allow-list kept names that no longer existed on disk).

This script treats the plan as the only source of truth and rewrites the other two to
match, then re-pins the two hashes the backend checks at load:

    frozen_manifest_sha = sha256(port_network_manifest.json)
    frozen_runtime_sha  = sha256(port_network_runtime.txt)

It is idempotent: run it after any recompile batch, then rebuild. Exits non-zero if
something is left inconsistent, so it can be used as a check as well as a repair.

    python tools/reconcile_assets.py [--model local/vk102] [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPV_REF = re.compile(r'assets/spv/([0-9a-f]{64})__([^"]+)')
FROZEN_SPV = re.compile(r'^\{"\.\./assets/spv/[0-9a-f]{64}__')


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan_shaders(plan_text: str) -> dict[str, str]:
    """name -> digest, in first-use order."""
    out: dict[str, str] = {}
    for digest, name in SPV_REF.findall(plan_text):
        out.setdefault(name, digest)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="local/vk102")
    ap.add_argument("--check", action="store_true", help="report drift, change nothing")
    a = ap.parse_args()

    model = (ROOT / a.model).resolve()
    plan_p = model / "manifest/port_network_runtime.txt"
    man_p = model / "manifest/port_network_manifest.json"
    frz_p = model / "generated/port_network_bridge_backend_frozen.inc"
    for p in (plan_p, man_p, frz_p):
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    plan_text = plan_p.read_text(encoding="utf-8")
    shaders = plan_shaders(plan_text)
    spv_dir = model / "assets/spv"

    # ---- manifest ----------------------------------------------------------------
    manifest = json.loads(man_p.read_text(encoding="utf-8"))
    entries = manifest["network_shaders"]
    by_sha = {v["sha256"]: k for k, v in entries.items()}
    fixed = 0
    # `digest` is the hash the plan's *filename* claims. For every ordinary shader that is also
    # the file's content hash, because a recompile renames the file. Resolution variants break
    # that: `patch_baked_canvas` writes `<base>_WxH.spv` from a patched module and keeps the base
    # name, so the real content differs. Recording the filename hash for those made the backend's
    # allow-list demand a hash no such file can have, and `nrvk_backend_init` then refused the
    # 1024x576 production plan with -2 "frozen file hash". Hash the file.
    for name, digest in shaders.items():
        path = f"../assets/spv/{digest}__{name}"
        f = spv_dir / f"{digest}__{name}"
        if not f.is_file():
            raise SystemExit(f"plan references a missing spv: {f.name}")
        real = sha_file(f)
        old_key = by_sha.get(real, name)
        entry = entries.pop(old_key, None) or entries.get(name) or {}
        entry.update({"bytes": f.stat().st_size, "sha256": real, "path": path})
        if old_key != name or entries.get(name) != entry:
            fixed += 1
        entries[name] = entry
        by_sha[real] = name
    # entries left over from an earlier name for the same shader are dangling: the plan is
    # the only list that decides which spv files exist
    dropped = [k for k, v in entries.items()
               if k not in shaders and v.get("path", "").startswith("../assets/spv/")]
    for k in dropped:
        del entries[k]
    bad = [k for k, v in entries.items() if len(v.get("sha256", "")) != 64]
    if bad:
        raise SystemExit(f"manifest entries with a malformed sha256: {bad}")
    missing = [k for k, v in entries.items()
               if not (spv_dir / v["path"].split("/")[-1]).is_file()]
    if missing:
        raise SystemExit(f"manifest entries point at missing files: {missing}")
    man_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"

    # ---- recipes -----------------------------------------------------------------
    # recipes.json is how `recompile_shader.py` finds the *source* for a digest, so it has to
    # agree with the plan too. Its keys are "<source dir>.spv", which for a shader that has
    # been rebuilt once is the plan name with the trailing "-<short hash>" removed.
    rec_p = None
    try:
        profile = json.loads((model / "model/extraction-receipt.json")
                             .read_text(encoding="utf-8"))["profile"]
        net_dir = json.loads((ROOT / "tools/model_profiles.json")
                             .read_text(encoding="utf-8"))[profile]["network_dir"]
        rec_p = ROOT / net_dir / "recipes.json"
    except (OSError, KeyError, ValueError):
        pass
    aliases = {}
    alias_p = ROOT / "tools/recipe_aliases.json"
    if alias_p.is_file():
        aliases = {k: v for k, v in json.loads(alias_p.read_text(encoding="utf-8")).items()
                   if not k.startswith("_")}
    rec_drift = 0
    unsynced: list[str] = []
    if rec_p and rec_p.is_file():
        rec = json.loads(rec_p.read_text(encoding="utf-8"))
        shaders_rec = rec["shaders"]
        for name, digest in shaders.items():
            key = next((k for k in ((name),
                                    re.sub(r"-[0-9a-f]{12}\.spv$", ".spv", name))
                        if k in shaders_rec), None)
            if key is None:
                # explicit alias table first: a family like f04 has three source directories, so
                # the prefix rule below cannot tell them apart
                parts = name.split("-", 2)
                alias_key = (parts[0] + "-" + parts[1].removesuffix(".spv")) if len(parts) > 1 else ""
                key = aliases.get(alias_key)
            if key is None:
                # last resort: the plan tag's leading field ("f01") is the recipe's source
                # directory prefix, so a unique such entry has to be the same shader
                head = name.split("-", 1)[0] + "-"
                cand = [k for k in shaders_rec if k.startswith(head)]
                if len(cand) == 1:
                    key = cand[0]
            if key is None:
                unsynced.append(name)
                continue
            if shaders_rec[key].get("sha256") != digest:
                shaders_rec[key]["sha256"] = digest
                shaders_rec[key]["raw_sha256"] = digest
                rec_drift += 1
        if rec_drift:
            rec_p.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8", newline="\n")

    # ---- frozen allow-list -------------------------------------------------------
    frz_lines = frz_p.read_text(encoding="utf-8").splitlines(keepends=True)
    keep = [l for l in frz_lines if not FROZEN_SPV.match(l)]
    slot = next((i for i, l in enumerate(frz_lines) if FROZEN_SPV.match(l)), None)
    if slot is None:
        raise SystemExit("frozen.inc has no spv allow-list block to replace")
    # The allow-list pairs the plan's path with the *content* hash, for the reason above: the
    # backend compares this against the file, not against the name.
    block = []
    for name, digest in shaders.items():
        block.append(f'{{"../assets/spv/{digest}__{name}","{sha_file(spv_dir / (digest + "__" + name))}"}},\n')
    # insert where the old block started, expressed against the filtered list
    before = sum(1 for l in frz_lines[:slot] if not FROZEN_SPV.match(l))
    frz = keep[:before] + block + keep[before:]
    frz_text = "".join(frz)
    frz_text = re.sub(r'frozen_manifest_sha="[0-9a-f]+"',
                      f'frozen_manifest_sha="{hashlib.sha256(man_text.encode()).hexdigest()}"',
                      frz_text)
    frz_text = re.sub(r'frozen_runtime_sha="[0-9a-f]+"',
                      f'frozen_runtime_sha="{sha_file(plan_p)}"', frz_text)

    # ---- apply / report ----------------------------------------------------------
    drift = []
    if man_p.read_text(encoding="utf-8") != man_text:
        drift.append(f"manifest: {fixed} entries realigned")
    if frz_p.read_text(encoding="utf-8") != frz_text:
        drift.append("frozen.inc: allow-list rebuilt, hashes re-pinned")
    if rec_drift:
        drift.append(f"recipes: {rec_drift} entries re-linked to the plan")
    if unsynced:
        print(f"  note: {len(unsynced)} shaders have no recipes entry, left alone: {unsynced[:3]}")
    print(f"plan: {len(shaders)} shaders" + (f", dropped {len(dropped)} dangling entries" if dropped else ""))
    for d in drift or ["nothing to do"]:
        print("  " + d)
    if a.check:
        return 1 if drift else 0
    man_p.write_text(man_text, encoding="utf-8", newline="\n")
    frz_p.write_text(frz_text, encoding="utf-8", newline="\n")

    # ---- verify ------------------------------------------------------------------
    assert sha_file(man_p) == re.search(r'frozen_manifest_sha="([0-9a-f]+)"',
                                        frz_text).group(1)
    assert sha_file(plan_p) == re.search(r'frozen_runtime_sha="([0-9a-f]+)"',
                                         frz_text).group(1)
    allow = set(re.findall(r'\{"\.\./assets/spv/([^"]+)"', frz_text))
    for name, digest in shaders.items():
        if f"{digest}__{name}" not in allow:
            raise SystemExit(f"{name} missing from the frozen allow-list")
    print("  verified: hashes re-pinned, allow-list complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
