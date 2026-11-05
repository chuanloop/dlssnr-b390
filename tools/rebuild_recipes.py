"""Rebuild source/network-fp16-20261027/recipes.json from the backup plan + the current plan.

`recompile_shader.py` patches recipes by line range, which silently rewrites neighbouring
entries when the file has been written by anything else (a json round-trip is enough). The
authoritative record of "which module is deployed" is the plan: every PASS line names
`<sha256>__<tag>-<short>.spv`. So rebuild the table from that, taking each module's source
and defines from the pre-recompile backup, paired by PASS line index.
"""
import json, re, pathlib

REF = re.compile(r'"\.\./assets/spv/([0-9a-f]{64})__([a-z0-9]+-[0-9a-f]+)\.spv"')
ROOT = pathlib.Path(__file__).resolve().parents[1]
PLAN = ROOT / "local/vk102/manifest/port_network_runtime_1280x720.txt"
BAK_PLAN = ROOT / ".recompile-backup/port_network_runtime.txt"
BAK_RECIPES = ROOT / ".recompile-backup/recipes.json"
OUT = ROOT / "source/network-fp16-20261027/recipes.json"


def passes(p):
    return [l for l in p.read_text(encoding="utf-8").splitlines() if l.startswith("PASS ")]


old, new = passes(BAK_PLAN), passes(PLAN)
bak = json.load(open(BAK_RECIPES))["shaders"]
out, unmapped = {}, []
for lo, ln in zip(old, new):
    mn = REF.search(ln)
    if not mn:
        continue
    digest, name = mn.groups()
    mo = REF.search(lo)
    entry = bak.get(mo.group(2) + ".spv") if mo else None
    entry = entry or bak.get(name + ".spv")
    if entry is None:
        unmapped.append((mo and mo.group(2), name))
        continue
    out[name + ".spv"] = {"source": entry["source"], "defines": entry.get("defines", []),
                          "raw_sha256": digest, "sha256": digest, "precision_decorations": []}
missing = [k for k, v in out.items() if not (ROOT / v["source"]).exists()]
OUT.write_text(json.dumps({"shaders": dict(sorted(out.items()))}, indent=1) + "\n",
               encoding="utf-8", newline="\n")
print(f"rebuilt {len(out)} entries; unmapped {len(unmapped)}; missing sources {len(missing)}")
if unmapped or missing:
    print("  ", unmapped[:3], missing[:3])
    raise SystemExit(1)
