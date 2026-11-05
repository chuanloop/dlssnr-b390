"""One-shot deploy: game backend DLL + the three bridge shaders, with the three-way hash check.

What the game actually reads (verified against nr_bridge.ini in bin/x64):

    bin/x64/port_network_bridge_backend.dll     the backend, loaded by _nvngx.dll
    <package>/manifest                         the frozen plan + runtime, pointed at by nr_bridge.ini
    <package>/assets/spv                       the three bridge shaders, relative to the manifest

`frozen_files` inside the DLL pins the plan, the cold weights *and* the three .spv, so a shader
that is newer than the DLL is not a downgrade - it is rc=-2 and a silently dead network. This
script refuses to leave the three out of step, and refuses to touch a running game.

    python tools/deploy_delta.py           # archive + deploy + verify
    python tools/deploy_delta.py --check   # verify only
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
GAME = Path(r"D:\SteamLibrary\steamapps\common\Cyberpunk 2077\bin\x64")
PACKAGE = WORKSPACE / "NR-B390-vk102-package" / "1-nr-game-package-vk102"
MODEL = ROOT / "local/vk102"
BUILD = ROOT / "build/vulkan/Release"
ARCHIVE = WORKSPACE / "NR-versions" / "20261031-delta-完整分辨率合成"
STAMP = ".bak-preDelta"
NAMES = ("gather", "scatter", "scatter32")
PLAN = "port_network_runtime.txt"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def game_running():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Cyberpunk2077.exe"],
                         capture_output=True).stdout.decode("gbk", errors="replace")
    return "Cyberpunk2077.exe" in out


def put(src, dst, backup=True):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if backup and dst.is_file() and not Path(str(dst) + STAMP).is_file():
        shutil.copy2(dst, str(dst) + STAMP)
    shutil.copy2(src, dst)
    return sha(dst)


def verify():
    dll = GAME / "port_network_bridge_backend.dll"
    want = sha(dll)
    problems = []
    for name in NAMES:
        pkg = PACKAGE / "assets/spv" / f"port_network_bridge_{name}.spv"
        model = MODEL / "assets/spv" / f"port_network_bridge_{name}.spv"
        if sha(pkg) != sha(model):
            problems.append(f"{name}: package {sha(pkg)[:16]} != model {sha(model)[:16]}")
    # The plan is part of the identity too: the DLL pins its hash, and a plan left behind from
    # another scale is the same failure as a stale shader - rc=-2 and a silently dead network.
    model_plan = MODEL / "manifest" / PLAN
    pkg_plan = PACKAGE / "manifest" / PLAN
    if sha(pkg_plan) != sha(model_plan):
        problems.append(f"plan: package {sha(pkg_plan)[:16]} != model {sha(model_plan)[:16]}")
    # The ladder is what `scale=` switches to at runtime; a half-copied ladder only fails later.
    model_ladder = MODEL / "manifest" / "ladder"
    pkg_ladder = PACKAGE / "manifest" / "ladder"
    for src in sorted(model_ladder.glob("plan-s*.txt")):
        dst = pkg_ladder / src.name
        if not dst.is_file() or sha(dst) != sha(src):
            problems.append(f"ladder: {src.name} missing or differs in the package")
    # The DLL's own pin list has to name exactly what is on disk next to the manifest.
    text = (MODEL / "generated/port_network_bridge_backend_frozen.inc").read_text(encoding="utf-8")
    for name in NAMES:
        pkg = PACKAGE / "assets/spv" / f"port_network_bridge_{name}.spv"
        if sha(pkg) not in text:
            problems.append(f"{name}: package hash absent from the pin list compiled into the DLL")
    if sha(model_plan) not in text:
        problems.append("plan: model hash absent from the pin list compiled into the DLL")
    return want, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if a.check:
        digest, problems = verify()
        print(f"deployed backend {digest[:16]}")
        for p in problems:
            print("  MISMATCH", p)
        return 1 if problems else 0

    if game_running():
        raise SystemExit("game is running; exit it first (the DLL is locked and mid-session swaps corrupt the swapchain)")

    # Archive the build that is about to go live; the files it replaces stay next to the originals
    # as `*.bak-preDelta`, so a rollback is one copy and never depends on this directory.
    if not ARCHIVE.exists():
        (ARCHIVE / "game-files").mkdir(parents=True)
        (ARCHIVE / "assets-spv").mkdir(parents=True)
        (ARCHIVE / "manifest").mkdir(parents=True)
        shutil.copy2(BUILD / "port_network_bridge_backend.dll", ARCHIVE / "game-files")
        for f in ("cold_resources_index.json", "port_network_manifest.json", "port_network_runtime.txt"):
            shutil.copy2(PACKAGE / "manifest" / f, ARCHIVE / "manifest" / f)
        for name in NAMES:
            shutil.copy2(MODEL / "assets/spv" / f"port_network_bridge_{name}.spv", ARCHIVE / "assets-spv")
        print(f"archived this build -> {ARCHIVE.relative_to(WORKSPACE)}")

    print("[1/4] backend DLL")
    print("  game   ", put(BUILD / "port_network_bridge_backend.dll",
                           GAME / "port_network_bridge_backend.dll"))
    print("[2/4] bridge shaders -> package assets/spv")
    for name in NAMES:
        src = MODEL / "assets/spv" / f"port_network_bridge_{name}.spv"
        print(f"  {name:9s}", put(src, PACKAGE / "assets/spv" / src.name))
    print("[3/4] runtime plan (the network's canvas lives here)")
    print(f"  plan     ", put(MODEL / "manifest" / PLAN, PACKAGE / "manifest" / PLAN))
    print("[4/4] scale ladder (what `scale=` switches to, without a rebuild)")
    n = 0
    for src in sorted((MODEL / "manifest" / "ladder").glob("plan-s*.txt")):
        put(src, PACKAGE / "manifest" / "ladder" / src.name)
        n += 1
    print(f"  ladder   {n} plans")

    digest, problems = verify()
    print(f"\ndeployed backend {digest[:16]}, spv pins {'OK' if not problems else 'MISMATCH'}")
    for p in problems:
        print("  ", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
