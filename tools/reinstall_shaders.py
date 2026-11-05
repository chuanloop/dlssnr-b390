"""Install freshly compiled bridge shaders and re-pin their hashes.

The backend refuses to run unless every file in `frozen_files` hashes to the value compiled
into the DLL, and two of those entries are the gather/scatter pairs themselves. Rebuilding a
shader therefore means three coordinated writes: the .spv in the model directory, the hash in
`generated/port_network_bridge_backend_frozen.inc`, and finally the DLL that carries it.

    python tools/reinstall_shaders.py                # copy local/up_*.spv? no: compile from source
    python tools/reinstall_shaders.py --spv-dir DIR  # use pre-built .spv files

Without --spv-dir the three shaders are compiled from source/vulkan with glslang, so the
binary always matches the source that is on disk.
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "local/vk102"
SPV = MODEL / "assets/spv"
FROZEN = MODEL / "generated" / "port_network_bridge_backend_frozen.inc"
GLSLANG = ROOT.parent / "_deps/glslang1650/bin/glslang.exe"
NAMES = ("gather", "scatter", "scatter32")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_one(name, out_dir):
    src = ROOT / "source/vulkan" / f"port_network_bridge_{name}.comp"
    out = Path(out_dir) / f"port_network_bridge_{name}.spv"
    subprocess.run([str(GLSLANG), "-V", "--target-env", "vulkan1.3",
                    "-I" + str(src.parent), "-o", str(out), str(src)], check=True)
    return out


def repin(text, name, digest):
    needle = f'"../assets/spv/port_network_bridge_{name}.spv","'
    hits = [i for i, line in enumerate(text.splitlines()) if needle in line]
    if len(hits) != 1:
        raise SystemExit(f"expected one frozen entry for {name}, found {len(hits)}")
    lines = text.splitlines()
    lines[hits[0]] = '{"../assets/spv/port_network_bridge_%s.spv","%s"},' % (name, digest)
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spv-dir", type=Path, help="directory holding the three .spv (default: compile)")
    a = ap.parse_args()

    SPV.mkdir(parents=True, exist_ok=True)
    text = FROZEN.read_text(encoding="utf-8")
    changed = []
    for name in NAMES:
        if a.spv_dir:
            built = a.spv_dir / f"port_network_bridge_{name}.spv"
        else:
            built = compile_one(name, SPV)
        target = SPV / f"port_network_bridge_{name}.spv"
        if built.resolve() != target.resolve():
            target.write_bytes(built.read_bytes())
        digest = sha(target)
        before = sha(target)
        text = repin(text, name, digest)
        changed.append((name, digest[:16], target.stat().st_size))
        assert before == digest
    FROZEN.write_text(text, encoding="utf-8", newline="\n")
    for name, digest, size in changed:
        print(f"  {name:9s} {digest} {size} B")
    print(f"re-pinned {FROZEN.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
