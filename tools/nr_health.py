"""Is the deployed pair actually the one running, and is the network keeping up?

Answers the three questions that otherwise cost a game session to tell apart:

  1. identity  - the DLL in bin/x64 and the three .spv the manifest points at were built together
  2. liveness  - `delta=` in nr_tune_read.txt proves the *new* backend parsed the tuning file
                 (older builds have no such key, so this line alone dates the running binary)
  3. throughput- `frames N: vk ran=` from nr_intel.log, plus any refused / rc= lines

    python tools/nr_health.py
"""
import ctypes
import hashlib
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAME = Path(r"D:\SteamLibrary\steamapps\common\Cyberpunk 2077\bin\x64")
PACKAGE = ROOT.parent / "NR-B390-vk102-package" / "1-nr-game-package-vk102"
MODEL = ROOT / "local/vk102"
BUILD = ROOT / "build/vulkan/Release/port_network_bridge_backend.dll"
NAMES = ("gather", "scatter", "scatter32")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def last_session(path):
    """nr_intel.log is appended across launches; only the last '=== ... attached ===' block counts."""
    if not Path(path).is_file():
        return []
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("=== nr-intel core attached"):
            start = i
    return lines[start:]


def tail(path, n=1, pattern=None):
    if not Path(path).is_file():
        return []
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if pattern:
        lines = [x for x in lines if re.search(pattern, x)]
    return lines[-n:]


def main():
    print("identity")
    live = GAME / "port_network_bridge_backend.dll"
    print(f"  backend in game  {sha(live)[:16]}  {live.stat().st_size} B")
    print(f"  backend in build {sha(BUILD)[:16]}  {BUILD.stat().st_size} B"
          f"{'' if sha(live) == sha(BUILD) else '   <- differs (build is newer or game is stale)'}")
    for name in NAMES:
        pkg = PACKAGE / "assets/spv" / f"port_network_bridge_{name}.spv"
        mod = MODEL / "assets/spv" / f"port_network_bridge_{name}.spv"
        print(f"  {name:9s} package {sha(pkg)[:16]}  model {sha(mod)[:16]}"
              f"{'' if sha(pkg) == sha(mod) else '   <- MISMATCH'}")

    lib = ctypes.CDLL(str(BUILD))
    lib.nrvk_backend_probe.argtypes = [ctypes.c_char_p]
    lib.nrvk_backend_probe.restype = ctypes.c_int
    lib.port_network_bridge_backend_last_error.restype = ctypes.c_char_p
    rc = lib.nrvk_backend_probe(str(PACKAGE / "manifest").replace("\\", "/").encode())
    print(f"\nprobe on the package manifest: rc={rc}"
          f"{'' if rc == 0 else '  ' + lib.port_network_bridge_backend_last_error().decode('utf-8', 'replace')}")

    print("\nliveness (the running backend's own reading of nr_tune.txt)")
    for line in tail(PACKAGE / "nr_tune_read.txt", 2):
        print("  " + line.strip())
    tune = PACKAGE / "nr_tune.txt"
    if tune.is_file():
        print("  nr_tune.txt: " + " | ".join(tune.read_text(encoding="utf-8").split()))

    print("\nthroughput (bin/x64/nr_intel.log, last session only)")
    log = GAME / "nr_intel.log"
    if log.is_file():
        print(f"  mtime {__import__('time').strftime('%H:%M:%S', __import__('time').localtime(log.stat().st_mtime))}")
    session = last_session(log)
    if session:
        print("  " + session[0].strip())
    ready = [x for x in session if "net: backend ready" in x]
    print(f"  backend: {'ready' if ready else 'NOT ready (network never started)'}")
    for line in [x for x in session if "frame submissions" in x or re.search(r"frames \d+: vk ran=", x)][-2:]:
        print("  " + line.strip())
    if not [x for x in session if re.search(r"frames \d+: vk ran=", x)]:
        print("  no 'frames N: vk ran=' line yet (session too short, or the network never ran)")
    # A first `bridge attach rc=-2` that is followed by `net: backend ready` is normal: the
    # processor is retried once the D3D12 device exists. Only hash/pipeline failures are fatal.
    fails = [x for x in session if re.search(r"frozen file hash|requires exact build-frozen|DEVICE_LOST|backend failed|refused", x)]
    for line in fails[-3:]:
        print("  ! " + line.strip())
    if not fails:
        print("  no hash/pipeline failures")
    notes = [x for x in session if "attach rc=" in x]
    if notes and ready:
        print("  i " + notes[-1].strip() + "   (first attach attempt, recovered)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
