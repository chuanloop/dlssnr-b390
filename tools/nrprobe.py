#!/usr/bin/env python3
"""Install / inspect / remove the dlssnr probe in Cyberpunk 2077.

Round 1 of M1 only needs one answer: does OptiScaler load and call a provider we
built, and with which arguments. So this script swaps in the OptiScaler build
that has the DLSS-NR channel, puts our probe where the reference proxy was, and
parks the existing bridge so nothing competes for the same NGX entry points.

Every file it touches is backed up with a .bak-nrprobe suffix and restored by
`uninstall`. The probe never renders: if the picture changes at all, something
is wrong.

    python tools/nrprobe.py install
    python tools/nrprobe.py status
    python tools/nrprobe.py log
    python tools/nrprobe.py uninstall
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GAME = Path(r"D:\SteamLibrary\steamapps\common\Cyberpunk 2077\bin\x64")
OPTISCALER_SRC = Path(
    r"D:\game\opti\【密码：juij.fun】[Master 特殊构建 - 优化 FSR & 添加 cyboman 版 DLL]"
    r" [2026-10-22] OptiScaler - 跨 GPU 游戏画质增强工具 [Juij 汉化]\OptiScaler"
)
PROBE = REPO / "build" / "dlssnr-provider" / "Release" / "nvngx.dll_dlssnr.dll"
LOG = GAME / "dlssnr_probe.log"
BACKUP_SUFFIX = ".bak-nrprobe"

# file in the game dir -> where install takes it from
DEPLOY = {
    "dxgi.dll": OPTISCALER_SRC / "dxgi.dll",
    "OptiScaler.dll": OPTISCALER_SRC / "OptiScaler.dll",
    "OptiScaler.ini": OPTISCALER_SRC / "OptiScaler.ini",
    "nvngx_dlssnr.dll": OPTISCALER_SRC / "nvngx_dlssnr.dll",
    "nvngx.dll_dlssnr.dll": PROBE,
    # NGX core stand-in: OptiScaler refuses to start the NR channel without one.
    # Deployed as _nvngx.dll, the name it tries first and games do not load.
    "_nvngx.dll": PROBE.parent / "_nvngx.dll",
}

# Files this script creates in the game dir from scratch (no original to restore).
OURS = {"nvngx.dll_dlssnr.dll", "_nvngx.dll"}
ASI = GAME / "plugins" / "CyberpunkNRBridge.asi"
ASI_PARKED = GAME / "plugins" / "CyberpunkNRBridge.asi.off-nrprobe"


def sha16(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def game_running() -> bool:
    out = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq Cyberpunk2077.exe"],
        capture_output=True,
    ).stdout.decode("gbk", "replace")
    return "Cyberpunk2077.exe" in out


def enable_nr_section(ini: Path) -> None:
    """Flip [DlssNr] Enabled to true without disturbing the rest of the file."""
    lines = ini.read_text(encoding="utf-8", errors="replace").splitlines()
    section = None
    changed = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            continue
        if section == "DlssNr" and stripped.lower().startswith("enabled"):
            lines[i] = "Enabled = true"
            changed = True
            break
    if not changed:
        lines.append("[DlssNr]")
        lines.append("Enabled = true")
    ini.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_ini(ini: Path) -> None:
    """NR on, plus OptiScaler logging - without it a failing channel is silent."""
    enable_nr_section(ini)
    lines = ini.read_text(encoding="utf-8", errors="replace").splitlines()
    section = None
    wanted = {
        "logtofile": "LogToFile = true",
        "loglevel": "LogLevel = 0",
        "singlefile": "SingleFile = false",
    }
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            continue
        if section == "Log":
            replacement = wanted.get(stripped.lower().split("=")[0].strip())
            if replacement:
                lines[i] = replacement
    ini.write_text("\n".join(lines) + "\n", encoding="utf-8")


def install() -> int:
    if game_running():
        print("game is running - close it first")
        return 1
    missing = [str(p) for p in list(DEPLOY.values()) if not p.is_file()]
    if missing:
        print("missing inputs:")
        for m in missing:
            print("  ", m)
        return 1

    for name, src in DEPLOY.items():
        dst = GAME / name
        backup = GAME / (name + BACKUP_SUFFIX)
        if name in OURS and backup.is_file():
            # A previous run backed up our own binary as if it were the original.
            backup.unlink()
            print(f"  drop stale backup {backup.name} (it was our own file)")
        if dst.is_file() and name not in OURS and not backup.is_file():
            shutil.copy2(dst, backup)
            print(f"  backup {name} -> {backup.name} ({sha16(backup)})")
        shutil.copy2(src, dst)
        print(f"  deploy {name:22s} {sha16(dst)}  <- {src.parent.name}")

    configure_ini(GAME / "OptiScaler.ini")
    print("  ini: [DlssNr] Enabled = true, [Log] LogToFile = true, LogLevel = 0")

    if ASI.is_file():
        ASI.replace(ASI_PARKED)
        print(f"  parked {ASI.name} -> {ASI_PARKED.name}")

    LOG.unlink(missing_ok=True)
    print("\ninstalled. launch the game, let it render a few seconds, then:")
    print("  python tools/nrprobe.py log")
    return 0


def uninstall() -> int:
    if game_running():
        print("game is running - close it first")
        return 1
    restored = 0
    for name in DEPLOY:
        backup = GAME / (name + BACKUP_SUFFIX)
        dst = GAME / name
        if backup.is_file():
            shutil.copy2(backup, dst)
            backup.unlink()
            restored += 1
            print(f"  restore {name}")
        elif dst.is_file() and name in OURS:
            dst.unlink()
            print(f"  remove {name} (no original existed)")
    if ASI_PARKED.is_file():
        ASI_PARKED.replace(ASI)
        print(f"  restore {ASI.name}")
    print(f"\nuninstalled, {restored} file(s) restored")
    return 0


def status() -> int:
    print(f"game running: {game_running()}")
    for name in DEPLOY:
        dst = GAME / name
        backup = GAME / (name + BACKUP_SUFFIX)
        state = sha16(dst) if dst.is_file() else "absent"
        print(f"  {name:22s} {state:16s} backup={'yes' if backup.is_file() else 'no'}")
    print(f"  asi                    {'parked' if ASI_PARKED.is_file() else 'active'}")
    if LOG.is_file():
        print(f"\nlog: {LOG.stat().st_size} bytes")
    return 0


def show_log() -> int:
    if not LOG.is_file():
        print(f"no log at {LOG}\nOptiScaler did not load the probe, or the game never started NR.")
        return 1
    text = LOG.read_text(encoding="utf-8", errors="replace")
    calls: dict[str, int] = {}
    for line in text.splitlines():
        if line.startswith("[") and "]" in line:
            rest = line.split("]", 1)[1].strip().split()
            if rest:
                calls[rest[0]] = calls.get(rest[0], 0) + 1
    print(f"{LOG.stat().st_size} bytes, {sum(calls.values())} call(s)")
    for name, count in sorted(calls.items(), key=lambda kv: -kv[1]):
        print(f"  {count:6d}  {name}")
    print("\n--- first 40 lines ---")
    print("\n".join(text.splitlines()[:40]))
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    actions = {
        "install": install,
        "uninstall": uninstall,
        "status": status,
        "log": show_log,
    }
    if cmd not in actions:
        print(__doc__)
        return 2
    return actions[cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
