"""Repair the two C256 swin weight packs of the Small profile.

NOTE: this post-install patch is now folded into the packer.  ``pack_swin`` in
``pack_b580_weights.py`` writes qkv as plain rows, keeps ``attn_out_proj``
tiled (plain for block 8, which has no ``*_finaltile`` asset), and rotates the
residual_scale lead-in, so a fresh re-extraction matches the frozen assets byte
for byte.  Keep this script for repairing already-installed ``assets\\cold``.

The accepted Small assets (and the upstream ``cold-hashes.json`` they match) put
two matrices in the wrong tiling for the kernels this profile actually runs:

* ``qkv`` (``wq``) is stored as 16x16 row-major tiles, but the C256 attention
  kernel loads it as plain rows (``coopMatLoad(..., stride 256, LCOL)`` over a
  contiguous 256-wide row).  Result: Q/K come out uncorrelated with
  ``Y @ Wq^T`` (measured: -0.002 / +0.052).
* ``attn_out_proj`` (``wo``) is stored as plain rows, but ``body_4`` loads it as
  16x16 tiles.  Result: the block output stays at ~0.66 even when the attention
  context is right.

Both were measured against the PyTorch reference on block 7/8; with the two
regions swapped the block outputs become corr 1.0000 (bit-identical).

A third repair lives in the same blob: ``residual_scale`` is preceded by an
8-half zero lead-in, and the plan's ``rs`` points at the lead-in, so every
channel read the scale of channel c-8 (first eight read 0).  Rotating the
16 padding halfs behind the table puts the table exactly at ``rs``.

Usage
-----
    python tools/patch_small_c256.py --model local/small          # report only
    python tools/patch_small_c256.py --model local/small --apply  # rewrite assets
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def read_plan(plan: Path):
    """[(buffer name, buffer path, {wo,wq,rs} of the body_1 pass that binds it)]."""
    buffers, passes = [], []
    current = None
    for line in plan.read_text(encoding="utf-8").splitlines():
        if line.startswith("BUFFER "):
            m = re.match(r'BUFFER "([^"]+)" (\d+) "([^"]*)"', line)
            buffers.append((m.group(1), m.group(3)))
        elif line.startswith("PASS "):
            current = {"name": line.split()[2].strip('"'), "binds": [], "push": None}
            passes.append(current)
        elif line.startswith("BIND ") and current is not None:
            current["binds"].append(int(line.split()[2]))
        elif line.startswith("PUSH ") and current is not None:
            current["push"] = [int(x) for x in line.split()[1:]]
    out = []
    for p in passes:
        if not p["name"].endswith("swin_body_1") or p["push"] is None:
            continue
        name, path = buffers[p["binds"][0]]
        if "weights" not in name:
            continue
        # p3 = (we, wm, wc, wq); p4 = (wo, rs, ars, bias)
        out.append((name, path, {"wq": p["push"][15], "wo": p["push"][16], "rs": p["push"][17]}))
    return out


def untile(words: np.ndarray, offset: int, n: int, k: int) -> None:
    """16x16 row-major tiles -> the plain (n, k) row-major matrix."""
    tile = words[offset:offset + n * k].reshape(n // 16, k // 16, 16, 16)
    words[offset:offset + n * k] = np.transpose(tile, (0, 2, 1, 3)).reshape(-1)


def tile(words: np.ndarray, offset: int, n: int, k: int) -> None:
    """The plain (n, k) row-major matrix -> 16x16 row-major tiles."""
    tile = words[offset:offset + n * k].reshape(n // 16, k // 16, 16, 16)
    words[offset:offset + n * k] = np.transpose(tile, (0, 2, 1, 3)).reshape(-1)


def place_scale(words: np.ndarray, offset: int, channels: int) -> bool:
    """Move the `channels` residual scales in front of their 8-half lead-in."""
    head = words[offset:offset + 8]
    if not np.all(head == 0):
        return False
    table = words[offset + 8:offset + 8 + channels].copy()
    words[offset:offset + channels] = table
    words[offset + channels:offset + channels + 16] = 0
    return True


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--apply", action="store_true", help="rewrite the assets and the plan")
    ap.add_argument("--force", action="store_true", help="apply even if the marker says it is done")
    args = ap.parse_args()

    model = args.model.resolve()
    manifest = model / "manifest"
    plan_path = manifest / "port_network_runtime.txt"
    index_path = manifest / "cold_resources_index.json"
    # The three repairs are not idempotent (a second pass would undo the first),
    # so record what was applied and refuse to run twice on the same assets.
    marker = manifest / "c256-patch.json"
    if marker.is_file() and not args.force:
        done = json.loads(marker.read_text(encoding="utf-8"))
        print("already patched: " + ", ".join(f"{k} -> {v[:12]}" for k, v in done.items()))
        print("pass --force to apply again (this would undo the repairs)")
        return 0
    index = json.loads(index_path.read_text(encoding="utf-8"))
    plan_text = plan_path.read_text(encoding="utf-8")

    changed = 0
    applied = {}
    for logical, rel, push in read_plan(plan_path):
        record = index[logical]
        file = model / "assets/cold" / Path(record["path"]).name
        if not file.is_file():
            file = model / record["path"]
        words = np.frombuffer(file.read_bytes(), dtype="<u2").copy()
        # Only the 256-channel swin packs carry the C256 layout (768x256 qkv,
        # 256x256 out_proj, 256 scales); the 32/64/128-channel blocks that share
        # the same body_1 pass name already match the reference to corr 1.0.
        if push["wq"] + 768 * 256 > words.size or push["rs"] + 264 > words.size:
            continue
        before = words.tobytes()

        # qkv: tiled -> plain rows.  The blob is 768 rows of 256; the last 256 of
        # the 768 are V, which the kernel also reads as rows, so all of it moves.
        untile(words, push["wq"], 768, 256)
        # attn_out_proj: plain rows -> 16x16 tiles (body_4's MAT_B loads).
        if push["wo"] == 0:
            tile(words, 0, 256, 256)
        # residual scale: drop the 8-half lead-in (the plan's `rs` points at the table).
        placed = place_scale(words, push["rs"], 256)
        after = words.tobytes()
        if after == before:
            continue
        changed += 1
        applied[logical] = digest(after)
        name = Path(record["path"]).name.split("__", 1)[1]
        new_name = f"{digest(after)}__{name}"
        print(f"{logical}: qkv rows, out_proj tiles, scale at rs={placed} -> {new_name}")
        if args.apply:
            (model / "assets/cold" / new_name).write_bytes(after)
            old = model / "assets/cold" / Path(record["path"]).name
            if old.exists() and old.name != new_name:
                old.unlink()
            index[logical] = {"logical_name": logical, "path": "../assets/cold/" + new_name,
                              "bytes": len(after), "sha256": digest(after)}
            plan_text = plan_text.replace("../assets/cold/" + Path(record["path"]).name,
                                          "../assets/cold/" + new_name)

    print(f"{changed} C256 pack(s) repaired")
    applied = {}
    if args.apply and changed:
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        plan_path.write_text(plan_text, encoding="utf-8")
        print("index and plan updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
