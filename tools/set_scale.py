"""Pick the network's working scale, rebuild, and put it live.

The model resolution is baked into the plan (`ioport_block70_2`'s frame words and the ioport
plane offsets), and the backend compiles those numbers in, so today a scale change is a
rebuild. This script does that whole chain in one command:

    freeze_plan.py --plan ladder/plan-s<scale>-<w>x<h>.txt --install   (plan + frame.inc + hash)
    cmake --build ...                                                  (the DLL that carries it)
    deploy_delta.py                                                    (DLL + shaders + plan)
    nrvk_backend_probe                                                 (rc=0 or it is not live)

    python tools/set_scale.py --list
    python tools/set_scale.py --scale 0.5        # 640x360 at a 720p render
    python tools/set_scale.py --scale 0.8        # back to the shipped baseline

The ladder plans are generated for a 1280x720 render. At another render size the same file is
still usable - the backend fits it by aspect - but the *scale* is then relative to 1280x720,
not to the render size. Regenerating the ladder per render size is a separate work item.
"""
import argparse
import ctypes
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "local/vk102"
LADDER = MODEL / "manifest" / "ladder"
BUILD = ROOT / "build/vulkan/Release/port_network_bridge_backend.dll"
PY = sys.executable


def run(cmd, **kw):
    r = subprocess.run([str(x) for x in cmd], capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.stdout.write(r.stdout[-2000:])
        sys.stderr.write(r.stderr[-2000:])
        raise SystemExit(f"failed: {' '.join(str(x) for x in cmd)}")
    return r.stdout


def ladder():
    entries = []
    for p in sorted(LADDER.glob("plan-s*.txt")):
        m = re.match(r"plan-s([0-9.]+)-(\d+)x(\d+)", p.stem)
        if not m:
            continue
        entries.append((float(m.group(1)), int(m.group(2)), int(m.group(3)), p))
    return entries


def show():
    print(f"{'scale':>6s} {'frame':>10s} {'file':>34s}")
    for scale, w, h, p in ladder():
        print(f"{scale:6.2f} {f'{w}x{h}':>10s} {p.name:>34s}")


def pick(scale):
    entries = ladder()
    if not entries:
        raise SystemExit(f"no ladder plans in {LADDER}")
    best = min(entries, key=lambda e: abs(e[0] - scale))
    if abs(best[0] - scale) > 0.02:
        raise SystemExit(f"no plan near scale {scale}; available: "
                         + ", ".join(f"{e[0]:.2f}" for e in entries))
    return best


def probe():
    dll = BUILD
    lib = ctypes.CDLL(str(dll))
    lib.nrvk_backend_probe.argtypes = [ctypes.c_char_p]
    lib.nrvk_backend_probe.restype = ctypes.c_int
    lib.port_network_bridge_backend_last_error.restype = ctypes.c_char_p
    rc = lib.nrvk_backend_probe(str(MODEL / "manifest").replace("\\", "/").encode())
    err = "" if rc == 0 else lib.port_network_bridge_backend_last_error().decode("utf-8", "replace")
    return rc, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--scale", type=float)
    a = ap.parse_args()

    if a.list or a.scale is None:
        show()
        return 0

    scale, w, h, plan = pick(a.scale)
    print(f"scale {scale:.2f} -> {plan.name} (frame {w}x{h})")

    print("[1/3] pin the plan")
    run([PY, ROOT / "tools/freeze_plan.py", "--plan", plan, "--install"])

    print("[2/3] build")
    env_note = " (unset https_proxy/http_proxy first if MSBuild reports MSB6001)"
    try:
        run(["cmake", "--build", "build/vulkan", "--config", "Release",
             "--target", "port_network_bridge_backend"], cwd=ROOT)
    except SystemExit:
        raise SystemExit(f"build failed{env_note}")

    rc, err = probe()
    if rc != 0:
        raise SystemExit(f"probe rc={rc} after build: {err}\n(the plan and the DLL disagree; nothing was deployed)")

    print("[3/3] deploy")
    print(run([PY, ROOT / "tools/deploy_delta.py"]).strip().splitlines()[-1])
    print(f"\nnext launch: nr_tune_read.txt gets a line like "
          f"`geom render=1280x720 active={w}x{h} canvas=... delta=1`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
