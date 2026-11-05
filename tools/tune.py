"""Live-tune the package's nr_tune.txt - the backend re-reads it twice a second, so no restart.

    python tools/tune.py                          # print the live values
    python tools/tune.py tone=1.0 texture=0       # strip every high frequency from the edit
    python tools/tune.py tone=0 texture=1         # keep only the high frequency (the "texture" half)
    python tools/tune.py tone=1.0 texture=2.0     # push the texture half to the ceiling
    python tools/tune.py --default                # back to the shipped 1/1 identity pair

Every key the backend parses is accepted; unknown keys are rejected instead of appended, because a
typo that silently does nothing is worse than a refusal.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUNE = ROOT.parent / "NR-B390-vk102-package" / "1-nr-game-package-vk102" / "nr_tune.txt"

# Mirrors the key table in port_network_bridge_backend.cpp ("intensity=" ... "texture=").
KEYS = ("detail", "colour", "guard", "acutance", "delta", "scale", "hist", "intensity",
        "blend", "easu", "fill", "dump", "tone", "texture")
DEFAULT = {"detail": "1.0", "colour": "1.0", "guard": "2", "acutance": "0", "delta": "1",
           "scale": "0.8", "intensity": "1.0", "blend": "1", "easu": "1", "dump": "0",
           "tone": "1.0", "texture": "1.0"}


def read():
    out = {}
    for line in TUNE.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def show():
    live = read()
    for k in KEYS:
        if k in live:
            print(f"{k:>10s} = {live[k]}")


def main(argv):
    if not TUNE.is_file():
        raise SystemExit(f"no tune file at {TUNE}")
    if not argv:
        show()
        return 0

    updates = dict(DEFAULT) if "--default" in argv else {}
    for arg in argv:
        if arg.startswith("-"):
            continue
        if "=" not in arg:
            raise SystemExit(f"expected key=value, got {arg!r}")
        k, v = arg.split("=", 1)
        k = k.strip()
        if k not in KEYS:
            raise SystemExit(f"unknown key {k!r}; the backend parses: {', '.join(KEYS)}")
        updates[k] = v.strip()

    lines, seen = [], set()
    for line in TUNE.read_text(encoding="utf-8").splitlines():
        head = line.split("#", 1)[0].strip()
        if "=" in head:
            k = head.split("=", 1)[0].strip()
            if k in updates:
                lines.append(f"{k}={updates[k]}")
                seen.add(k)
                continue
        lines.append(line)
    for k, v in updates.items():
        if k not in seen:
            lines.append(f"{k}={v}")
    TUNE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    live = read()
    print(f"wrote {TUNE}")
    for k in updates:
        print(f"  {k:>10s} = {live.get(k)}")
    print("live within half a second - no restart")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
