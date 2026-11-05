#!/usr/bin/env python3
"""Push local/small's manifest and shaders into a game's dlssnr-amd directory.

Why this exists: `recompile_shader.py` rewrites local/small/manifest and
local/small/assets/spv, and the built core freezes those hashes. The copy inside
a game tree is a *separate* copy (small 档在巫师3 是 `dlssnr-amd-small/`, RE 是
`dlssnr-amd/`), so a rebuild that skips this step makes the game refuse to start
the network:

    port: init failed: requires exact build-frozen manifest/runtime

Usage:
    python tools/sync_small_to_game.py "D:\\SteamLibrary\\...\\bin\\x64_dx12"
    python tools/sync_small_to_game.py "<bin>" --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]          # .../intel-dlss5nr-cyberpunk2077-main
SMALL = HERE / "local" / "small"
MANIFEST_NAMES = ("port_network_runtime.txt", "port_network_manifest.json",
                  "cold_resources_index.json")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_hashes() -> tuple[str, str]:
    inc = (SMALL / "generated" / "port_network_bridge_backend_frozen.inc").read_text(encoding="utf-8")
    runtime = manifest = ""
    for line in inc.splitlines():
        if "frozen_runtime_sha" in line:
            runtime = line.split('"')[1]
        elif "frozen_manifest_sha" in line:
            manifest = line.split('"')[1]
    return runtime, manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game", help="the directory holding the game .exe (dlssnr-amd lives there)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    game = Path(a.game)
    target = game / "dlssnr-amd"
    if not (target / "manifest").is_dir():
        sys.exit(f"no dlssnr-amd/manifest under {game}; is that the bin directory?")

    copied = 0
    for name in MANIFEST_NAMES:
        src, dst = SMALL / "manifest" / name, target / "manifest" / name
        if not src.is_file():
            continue
        if dst.is_file() and sha(dst) == sha(src):
            continue
        print(f"  manifest/{name}")
        if not a.dry_run:
            target.joinpath("manifest").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied += 1

    # The ladder is what lets the backend serve a different render size than the installed plan.
    # It lives beside the runtime plan, so a package without it can only ever run one extent.
    lad_src = SMALL / "manifest" / "ladder"
    if lad_src.is_dir():
        lad_dst = target / "manifest" / "ladder"
        lad_dst.mkdir(parents=True, exist_ok=True)
        for src in sorted(lad_src.glob("*.txt")):
            dst = lad_dst / src.name
            if dst.is_file() and sha(dst) == sha(src):
                continue
            print(f"  manifest/ladder/{src.name}")
            if not a.dry_run:
                shutil.copy2(src, dst)
            copied += 1

    spv_src = SMALL / "assets" / "spv"
    spv_dst = target / "assets" / "spv"
    if spv_src.is_dir():
        spv_dst.mkdir(parents=True, exist_ok=True)
        for src in spv_src.glob("*.spv"):
            dst = spv_dst / src.name
            if dst.is_file() and sha(dst) == sha(src):
                continue
            print(f"  assets/spv/{src.name}")
            if not a.dry_run:
                shutil.copy2(src, dst)
            copied += 1

    rt, mf = frozen_hashes()
    real_rt = sha(target / "manifest" / "port_network_runtime.txt")
    real_mf = sha(target / "manifest" / "port_network_manifest.json")
    ok = (rt == real_rt) and (mf == real_mf)
    print(f"\n{'would copy' if a.dry_run else 'copied'} {copied} file(s)")
    print(f"frozen runtime  {rt[:16]} vs {real_rt[:16]}  {'OK' if rt == real_rt else 'MISMATCH'}")
    print(f"frozen manifest {mf[:16]} vs {real_mf[:16]}  {'OK' if mf == real_mf else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
