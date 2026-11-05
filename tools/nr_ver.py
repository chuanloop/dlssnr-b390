#!/usr/bin/env python3
"""Version bookkeeping for the Small network.

Rules this tool enforces (decided 2026-11-02 after a bad shader swap):

  * v1 is frozen. It is the version that measured 32.28/32.78 ms in The Witcher 3
    and 66.46 dB offline. Nothing overwrites it, ever.
  * Every later attempt gets its own number and its own directory, even if it is
    rejected. A rejected version stays on disk with its measurement, because the
    reason it failed is worth more than the disk space.
  * A version is a *recipe*: the three manifest files plus the 42 spv binaries.
    Source edits alone are not a version; if you cannot reproduce the spv you
    cannot reproduce the number.

Usage
-----
    python tools/nr_ver.py snap v2 --note "ffn_contract split into 64-token groups"
    python tools/nr_ver.py list
    python tools/nr_ver.py diff v2            # current working tree vs v2
    python tools/nr_ver.py checkout v1        # restore manifest+spv of a version
    python tools/nr_ver.py promote v2         # mark v2 as the one the game runs

`checkout` only restores the recipe on disk under local/small/. It does NOT
rebuild dlssnr_core.dll: the frozen hashes live in the core, so after a checkout
you must rebuild before deploying to a game.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMALL = os.path.join(ROOT, "local", "small")
VERS = os.path.join(SMALL, "versions")
MANIFEST = os.path.join(SMALL, "manifest")
SPV = os.path.join(SMALL, "assets", "spv")
MAN_FILES = ["port_network_runtime.txt", "port_network_manifest.json", "cold_resources_index.json"]
INDEX = os.path.join(VERS, "INDEX.md")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hashes(manifest: str, spv_dir: str) -> dict[str, str]:
    out = {}
    for name in MAN_FILES:
        p = os.path.join(manifest, name)
        if os.path.exists(p):
            out[f"manifest/{name}"] = sha256(p)
    if os.path.isdir(spv_dir):
        for name in sorted(os.listdir(spv_dir)):
            if name.endswith(".spv"):
                out[f"assets/spv/{name}"] = sha256(os.path.join(spv_dir, name))
    # The ladder decides which render sizes the installed plan can serve; a snapshot without it
    # cannot be reproduced (37 of 38 plans live here).
    ladder = os.path.join(manifest, "ladder")
    if os.path.isdir(ladder):
        for name in sorted(os.listdir(ladder)):
            if name.endswith(".txt"):
                out[f"manifest/ladder/{name}"] = sha256(os.path.join(ladder, name))
    return out


def u32(v: int) -> str:
    return "0x%08x" % (v & 0xFFFFFFFF)


def load_index() -> dict[str, dict]:
    """Very small table reader: one `| name | ... |` row per version."""
    rows = {}
    if not os.path.exists(INDEX):
        return rows
    for line in open(INDEX, encoding="utf-8"):
        if not line.startswith("| v") or line.startswith("| 版本"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2:
            rows[cells[0]] = {"status": cells[1], "note": cells[2] if len(cells) > 2 else ""}
    return rows


def write_index(version: str, status: str, note: str, extra: str = "") -> None:
    os.makedirs(VERS, exist_ok=True)
    header = (
        "# Small 版本台账\n\n"
        "| 版本 | 状态 | 改动 | 离线单支 | 游戏内 | 结论 |\n"
        "|---|---|---|---|---|---|\n"
    )
    lines = []
    if os.path.exists(INDEX):
        body = open(INDEX, encoding="utf-8").read().splitlines()
        for line in body:
            if line.startswith("| v"):
                lines.append(line)
    lines = [l for l in lines if not l.startswith(f"| {version} |")]
    lines.append(f"| {version} | {status} | {note} | {extra} |")
    with open(INDEX, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines) + "\n")


def cmd_snap(args) -> int:
    dest = os.path.join(VERS, args.name)
    if os.path.exists(dest) and not args.force:
        raise SystemExit(f"{dest} exists; pick a new number or pass --force")
    os.makedirs(os.path.join(dest, "manifest"), exist_ok=True)
    os.makedirs(os.path.join(dest, "spv"), exist_ok=True)
    for name in MAN_FILES:
        src = os.path.join(MANIFEST, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, "manifest", name))
    for name in sorted(os.listdir(SPV)):
        if name.endswith(".spv"):
            shutil.copy2(os.path.join(SPV, name), os.path.join(dest, "spv", name))
    ladder = os.path.join(MANIFEST, "ladder")
    if os.path.isdir(ladder):
        os.makedirs(os.path.join(dest, "manifest", "ladder"), exist_ok=True)
        for name in sorted(os.listdir(ladder)):
            if name.endswith(".txt"):
                shutil.copy2(os.path.join(ladder, name),
                             os.path.join(dest, "manifest", "ladder", name))
    h = hashes(os.path.join(dest, "manifest"), os.path.join(dest, "spv"))
    with open(os.path.join(dest, "hashes.txt"), "w", encoding="utf-8") as f:
        for k, v in h.items():
            f.write(f"{v}  {k}\n")
    rt = h.get("manifest/port_network_runtime.txt", "")
    mf = h.get("manifest/port_network_manifest.json", "")
    with open(os.path.join(dest, "VERSION.md"), "w", encoding="utf-8") as f:
        f.write(f"# {args.name}\n\n")
        f.write(f"{args.note}\n\n")
        f.write("| 项 | 值 |\n|---|---|\n")
        f.write(f"| runtime sha | `{rt[:16]}` |\n| manifest sha | `{mf[:16]}` |\n")
        f.write(f"| spv 数量 | {sum(1 for k in h if k.startswith('assets/spv'))} |\n")
        f.write(f"| 梯级档位 | {sum(1 for k in h if k.startswith('manifest/ladder/'))} |\n")
    write_index(args.name, args.status, args.note, args.extra)
    print(f"snapshot {args.name}: runtime {rt[:12]} manifest {mf[:12]} "
          f"({sum(1 for k in h if k.startswith('assets/spv'))} spv)")
    return 0


def cmd_list(args) -> int:
    if not os.path.exists(INDEX):
        print("no versions yet")
        return 0
    print(open(INDEX, encoding="utf-8").read())
    return 0


def cmd_diff(args) -> int:
    dest = os.path.join(VERS, args.name)
    if not os.path.isdir(dest):
        raise SystemExit(f"no such version: {args.name}")
    want = {}
    for line in open(os.path.join(dest, "hashes.txt"), encoding="utf-8"):
        h, name = line.split()
        want[name] = h
    cur = hashes(MANIFEST, SPV)
    only_cur = sorted(set(cur) - set(want))
    only_ref = sorted(set(want) - set(cur))
    changed = sorted(k for k in set(cur) & set(want) if cur[k] != want[k])
    if not (only_cur or only_ref or changed):
        print(f"identical to {args.name}")
        return 0
    for k in changed:
        print(f"  changed  {k}")
    for k in only_cur:
        print(f"  added    {k}")
    for k in only_ref:
        print(f"  missing  {k}")
    return 1


def cmd_checkout(args) -> int:
    dest = os.path.join(VERS, args.name)
    if not os.path.isdir(dest):
        raise SystemExit(f"no such version: {args.name}")
    for name in MAN_FILES:
        src = os.path.join(dest, "manifest", name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(MANIFEST, name))
    if os.path.isdir(SPV):
        for name in os.listdir(SPV):
            if name.endswith(".spv"):
                os.remove(os.path.join(SPV, name))
    for name in os.listdir(os.path.join(dest, "spv")):
        shutil.copy2(os.path.join(dest, "spv", name), os.path.join(SPV, name))
    ref_ladder = os.path.join(dest, "manifest", "ladder")
    if os.path.isdir(ref_ladder):
        cur_ladder = os.path.join(MANIFEST, "ladder")
        os.makedirs(cur_ladder, exist_ok=True)
        for name in os.listdir(cur_ladder):
            if name.endswith(".txt"):
                os.remove(os.path.join(cur_ladder, name))
        for name in os.listdir(ref_ladder):
            shutil.copy2(os.path.join(ref_ladder, name), os.path.join(cur_ladder, name))
    print(f"checked out {args.name}; REBUILD dlssnr_core.dll before deploying "
          f"(frozen hashes changed)")
    return 0


def cmd_promote(args) -> int:
    dest = os.path.join(VERS, args.name)
    if not os.path.isdir(dest):
        raise SystemExit(f"no such version: {args.name}")
    vm = os.path.join(dest, "VERSION.md")
    meta = {}
    for line in open(vm, encoding="utf-8"):
        if line.startswith("| ") and "|" in line[2:]:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 2:
                meta[cells[0]] = cells[1]
    with open(os.path.join(dest, "PROMOTED"), "w", encoding="utf-8") as f:
        f.write(args.note or "promoted")
    print(f"{args.name} promoted. Record the game measurement in its VERSION.md.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snap")
    s.add_argument("name")
    s.add_argument("--note", default="")
    s.add_argument("--status", default="候选")
    s.add_argument("--extra", default="—")
    s.add_argument("--force", action="store_true")
    sub.add_parser("list")
    d = sub.add_parser("diff")
    d.add_argument("name")
    c = sub.add_parser("checkout")
    c.add_argument("name")
    p = sub.add_parser("promote")
    p.add_argument("name")
    p.add_argument("--note", default="")
    args = ap.parse_args()
    return {"snap": cmd_snap, "list": cmd_list, "diff": cmd_diff,
            "checkout": cmd_checkout, "promote": cmd_promote}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
