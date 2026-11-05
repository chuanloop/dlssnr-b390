"""Can the frozen plan's PUSH constants be explained by layer geometry?

For every pass family, each of the 32-or-fewer push words is fitted as a linear
combination of (1, w, h, w*h) taken from that pass's layer. A word that fits with
zero residual is geometry-derived and can be regenerated for another frame size;
a word that does not is a constant the generator must carry over verbatim.

Run: python tools/analyze_push.py
"""
import collections
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "source/network-fp16-20261027/runtime.template.txt"

spec = importlib.util.spec_from_file_location("nr_geometry", ROOT / "tools/nr_geometry.py")
geom = importlib.util.module_from_spec(spec)
sys.modules["nr_geometry"] = geom
spec.loader.exec_module(geom)


def read_plan():
    passes, current = [], None
    for line in PLAN.read_text(encoding="utf-8").splitlines():
        if line.startswith("PASS "):
            parts = line.split()
            current = {"row": int(parts[1]), "name": parts[2].strip('"'),
                       "npush": int(parts[7]), "push": None}
            passes.append(current)
        elif line.startswith("PUSH ") and current is not None:
            current["push"] = [int(x) for x in line.split()[1:]]
    return passes


def main():
    layers = geom.build_plan(1280, 720)
    families = collections.defaultdict(list)
    for p in read_plan():
        if not p["push"]:
            continue
        families[(re.sub(r"\d+", "N", p["name"]), p["npush"])].append(p)

    print(f"{'family (npush)':<34}{'n':>4}{'geom':>7}{'const':>7}{'words':>7}")
    total = fit_total = 0
    for (name, npush), passes in sorted(families.items(), key=lambda kv: -len(kv[1])):
        if len(passes) < 5:
            continue
        x = np.array([[1.0, layers[p["row"]].w, layers[p["row"]].h,
                       layers[p["row"]].w * layers[p["row"]].h] for p in passes])
        y = np.array([p["push"] for p in passes], dtype=float)
        fit = 0
        for k in range(y.shape[1]):
            coef, _, _, _ = np.linalg.lstsq(x, y[:, k], rcond=None)
            if np.max(np.abs(x @ coef - y[:, k])) < 0.5:
                fit += 1
        total += y.shape[1]
        fit_total += fit
        print(f"{name[:18] + f'({npush})':<34}{len(passes):>4}{fit:>7}"
              f"{y.shape[1] - fit:>7}{y.shape[1]:>7}")
    print(f"\ngeometry-derived words: {fit_total}/{total} "
          f"({100 * fit_total / max(1, total):.0f}%)")


if __name__ == "__main__":
    main()
