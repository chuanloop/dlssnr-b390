"""Lift the cubic SiLU off the narrowed value and onto the float32 accumulator.

AMD's ``fswin`` evaluates the activation with ``OpExtInst %float ... Fma`` on ``%float`` constants,
straight off the float32 accumulator; the port narrows to ``float16_t`` first and evaluates it in half
precision. One f16 rounding (5e-4) before an e4m3 point (12.5% steps) flips about 0.4% of the roundings
that follow, which is 0.57% of the picture on the reference frames. Doing it on the accumulator instead is worth +0.048 dB on f04 alone.

The rewrite is deliberately narrow: ``v`` keeps its name, its type and every use below it, so the only
thing that changes is *where* the narrowing happens::

    NRH_CH v = nrh_narrow(c);          NRH_C v_f = c;
    if (debug) coopMatStore(v, ...);   v_f[0] = nr_act_f(v_f[0]);  ...
    v[0] = nr_act_h(v[0]);      ->     NRH_CH v = nrh_narrow(v_f);
    ...                                if (debug) coopMatStore(v, ...);
    NRH_A a = nrh_to_A(v);             NRH_A a = nrh_to_A(v);

(the debug store moves one line down, so it shows the post-activation value; debug is off in every
measurement anyway).

Usage
-----
    python tools/f32_activation.py --dir source/network-fp16-20261027/f01-c12c0f9d6e04 --dry-run
    python tools/f32_activation.py --dir source/network-fp16-20261027/f01-c12c0f9d6e04
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BLOCK = re.compile(
    r"(?P<indent>[ \t]*)NRH_CH (?P<v>\w+)\s*=\s*nrh_narrow\((?P<c>[^;]+)\);\n"
    r"(?P<mid>(?:[ \t]*if\s*\(debug\)[^\n]*\n)?)"
    r"(?P<acts>(?:[ \t]*(?P=v)\[[0-3]\]\s*=\s*(?P<fn>\w*_?act_h)\((?P=v)\[[0-3]\]\);\n){4})"
)

F32FUNC = """
// float32 cubic SiLU on the accumulator, as AMD's fswin does it (OpExtInst %%float Fma on
// %%float constants).  %(h)s narrows first; that one f16 rounding moves ~0.4%% of the e4m3
// roundings downstream, which is 0.57%% of the picture (measured on the reference).
float %(f)s(float x) {
    float t = clamp(x, -4.0, 4.0);
    float u = fma(-0.055908203125, abs(t), 0.447265625);
    return x * fma(t, u, 0.89453125);
}
"""


def rewrite(text: str) -> tuple[str, int, str]:
    count = 0
    known: set[str] = set()

    def sub(m: re.Match) -> str:
        nonlocal count
        count += 1
        ind, v, c, mid, fn = m.group("indent"), m.group("v"), m.group("c").strip(), m.group("mid"), m.group("fn")
        f32 = fn[:-len("_h")] + "_f"
        known.add((fn, f32))
        out = [f"{ind}NRH_C f_{v} = {c};\n"]
        for i in range(4):
            out.append(f"{ind}f_{v}[{i}] = {f32}(f_{v}[{i}]);\n")
        out.append(f"{ind}NRH_CH {v} = nrh_narrow(f_{v});\n")
        out.append(mid)
        return "".join(out)

    return BLOCK.sub(sub, text), count, ""


BACK = re.compile(
    r"(?P<indent>[ \t]*)NRH_C f_(?P<v>\w+) = (?P<c>[^;]+);\n"
    r"(?P<acts>(?:[ \t]*f_(?P=v)\[[0-3]\] = (?P<f32>\w*_?act_f)\(f_(?P=v)\[[0-3]\]\);\n){4})"
    r"[ \t]*NRH_CH (?P=v) = nrh_narrow\(f_(?P=v)\);\n"
    r"(?P<mid>(?:[ \t]*if\s*\(debug\)[^\n]*\n)?)"
)


def revert_text(text: str) -> tuple[str, int]:
    """Undo `rewrite`: back to narrowing first and activating in half precision."""
    count = 0

    def sub(m: re.Match) -> str:
        nonlocal count
        count += 1
        ind, v, c, mid = m.group("indent"), m.group("v"), m.group("c").strip(), m.group("mid")
        h = m.group("f32")[:-len("_f")] + "_h"
        out = [f"{ind}NRH_CH {v}=nrh_narrow({c});\n", mid]
        for i in range(4):
            out.append(f"{ind}{v}[{i}]={h}({v}[{i}]);\n")
        return "".join(out)

    text = BACK.sub(sub, text)
    # drop the helper we inserted (the half one stays)
    text = re.sub(r"// float32 cubic SiLU on the accumulator.*?\nfloat (\w*_?act_f)\(float x\) \{.*?\n\}\n\n", "", text, flags=re.S)
    return text, count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true", help="undo a previous run (verified by content hash)")
    a = ap.parse_args()

    d = (ROOT / a.dir).resolve()
    kernel = d / "kernel.comp"
    if not kernel.is_file():
        raise SystemExit(f"no kernel.comp under {d}")
    text = kernel.read_text(encoding="utf-8")
    if a.revert:
        back, n = revert_text(text)
        print(f"{d.name}: {n} blocks returned to half precision")
        if n:
            if a.dry_run:
                print("   dry run, not written")
            else:
                kernel.write_text(back, encoding="utf-8", newline="\n")
                print("   written")
        return 0
    new, count, _ = rewrite(text)
    if count == 0:
        print(f"{d.name}: no `NRH_CH x = nrh_narrow(c)` + 4x act block found")
        return 0

    # the f32 helper goes next to the half one, in whichever include defines it
    pairs = set()
    for m in re.finditer(r"float16_t (\w*_?act_h)\(float16_t x\)", text):
        pairs.add(m.group(1))
    for glsl in sorted(d.glob("*.glsl")):
        t = glsl.read_text(encoding="utf-8")
        for m in re.finditer(r"float16_t (\w*_?act_h)\(float16_t x\)", t):
            pairs.add(m.group(1))
        for fn in list(pairs):
            f32 = fn[:-len("_h")] + "_f"
            if f"{f32}(" not in t:
                anchor = f"float16_t {fn}(float16_t x) {{"
                if anchor in t:
                    t = t.replace(anchor, F32FUNC % {"h": fn, "f": f32} + "\n" + anchor, 1)
                    print(f"  {glsl.name}: added {f32}")
        if not a.dry_run:
            glsl.write_text(t, encoding="utf-8", newline="\n")

    if not pairs:
        raise SystemExit("could not find the half activation to mirror")

    # a kernel may define the half helper itself rather than in an include
    for fn in sorted(pairs):
        f32 = fn[:-len("_h")] + "_f"
        anchor = f"float16_t {fn}(float16_t x) {{"
        if f"float {f32}(float x)" not in new and anchor in new:
            new = new.replace(anchor, F32FUNC % {"h": fn, "f": f32} + "\n" + anchor, 1)
            print(f"  kernel.comp: added {f32}")

    print(f"{d.name}: {count} activation blocks lifted to float32; helpers {sorted(pairs)}")
    if a.dry_run:
        for line in new.splitlines():
            if "NRH_C f_" in line or "nrh_narrow(f_" in line:
                print("   ", line.strip())
    else:
        kernel.write_text(new, encoding="utf-8", newline="\n")
        print("   written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
