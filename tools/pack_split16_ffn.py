"""Rebuild a split16 block's grouped feed-forward weights in the layout the kernel reads.

The vendor asset for a split16 block keeps the three FFN matrices in a fragment order that
does not match `f08` (SplitFfwd). Measured with single-word pulses on block 9 (eight pulses,
one per group, each response matched the expected input column at corr 1.0000):

    ffwd_a (expand_in, [in, out])       off 0        262144 words
    ffwd_b (expand_hidden + contract)   off 262144   262144 words
    ffwd_c (contract)                   off 393216   131072 words

    stage 1: blob[g*16384 + 1024*(o//16) + 256*(k//16) + 16*(o%16) + k%16] = expand_in[k, g*64+o]
    stage 2: blob[262144 + g*16384 + 1024*(o//16) + 256*(k//16) + 16*(o%16) + k%16] = Q0[g, k, o]
    stage 3: blob[393216 + g*16384 + (k//16)*256 + 4096*(o//16) + 16*(o%16) + k%16] = Q2[g, k, o]

with the three reference tensors coming straight out of the PyTorch reference:

    expand_in      (512, 512)     [in, group*64 + out]
    expand_hidden  (8, 64, 256)   [group, in, out]
    contract       (8, 256, 64)   [group, in, out]

Rewriting those three regions this way takes block 9's SplitFfwd output from
corr 0.0058 (block9 F) to corr 1.0000 -- same bytes in, only the order changes.

Usage:
    python tools/pack_split16_ffn.py --model local/small --block 9
    python tools/pack_split16_ffn.py --model local/small --block 9 --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CROP = ROOT.parent.parent / "50-SCRATCH" / "weights-small.safetensors"


def reference_tensors(weights: Path, block: int):
    """(expand_in, expand_hidden, contract, qkv or None) for `block`, from the reference model."""
    sys.path.insert(0, str(ROOT.parent.parent / "20-AMD" / "dlssnr-src" / "dlssnr-pytorch"))
    from dlssnr.model import NRNet
    from dlssnr.weights import Weights

    net = NRNet(Weights.load(str(weights)), 1280, 720)
    ffwd = next(l for spec, l in zip(net.plan, net.layers)
                if spec.block == block and getattr(l, "expand_in", None) is not None)
    attn = next((l for spec, l in zip(net.plan, net.layers)
                 if spec.block == block and getattr(l, "qkv", None) is not None), None)
    proj = next((l for spec, l in zip(net.plan, net.layers)
                 if spec.block == block and spec.kind == "SplitProjPool"), None)
    head = next((l for spec, l in zip(net.plan, net.layers)
                 if spec.block == block and spec.kind == "SplitFinalHead"), None)
    return (
        ffwd.expand_in.detach().float().numpy(),
        ffwd.expand_hidden.detach().float().numpy(),
        ffwd.contract.detach().float().numpy(),
        None if attn is None else attn.qkv.detach().float().numpy(),
        None if proj is None else proj.weight.detach().float().numpy(),
        None if head is None else head.weight.detach().float().numpy(),
    )


def read_plan_offsets(plan: Path, block: int) -> tuple[int, int, int, int, int, int, str]:
    """(we, wm, wc, wq, buffer name) out of the plan's own push blocks."""
    push = {}
    current = None
    for line in plan.read_text(encoding="utf-8").splitlines():
        if line.startswith("PASS "):
            current = line.split()[2].strip('"')
        elif line.startswith("PUSH ") and current is not None:
            push[current] = [int(x) for x in line.split()[1:]]
    # pass names carry the *row* prefix (r13 for block 10), so match on the suffix
    l0 = next((n for n in push if n.endswith(f"_split16_b{block}_l0")), None)
    if l0 is None:
        raise SystemExit(f"plan has no *_split16_b{block}_l0 push")
    ffwd = push[l0]
    # every pass of the layer carries the same 24-word push block, so p3.y is the qkv offset
    # even on the l0 pass (f08 itself ignores p3).
    wq = ffwd[13]
    return ffwd[8], ffwd[9], ffwd[10], wq, ffwd[14], ffwd[18], l0


def slot(g: int, o: int, k: int, outs_per_group: int, ins: int) -> int:
    """Word index of (group, out, in) inside one of the three FFN matrices.

    One rule covers all three stages (verified with single-word pulses on block 9):
    the 16x16 tiles are stored out-major, the input-block stride is 256 words, the
    out-block stride is (ins/16)*256, and a group is outs_per_group*ins words.
    """
    return (g * outs_per_group * ins
            + (o // 16) * ((ins // 16) * 256)
            + (k // 16) * 256
            + 16 * (o % 16) + (k % 16))


def rebuild(words: np.ndarray, block: int, we: int, wm: int, wc: int, wq: int, wp: int, wh: int, ref) -> int:
    expand_in, q0, q2, qkv, proj, head = ref
    touched = 0
    for g in range(8):
        for o in range(64):
            for k in range(512):
                words[we + slot(g, o, k, 64, 512)] = np.float16(expand_in[k, g * 64 + o]).view(np.uint16)
                touched += 1
        for o in range(256):
            for k in range(64):
                words[wm + slot(g, o, k, 256, 64)] = np.float16(q0[g, k, o]).view(np.uint16)
                touched += 1
        for o in range(64):
            for k in range(256):
                words[wc + slot(g, o, k, 64, 256)] = np.float16(q2[g, k, o]).view(np.uint16)
                touched += 1
    if proj is not None and wp:
        # KIND=1/2 projections read blob[out*in_dim + in], i.e. the reference weight
        # ([in][out]) transposed, row-major.
        for o in range(proj.shape[1]):
            for k in range(proj.shape[0]):
                words[wp + o * proj.shape[0] + k] = np.float16(proj[k, o]).view(np.uint16)
                touched += 1
    if head is not None and wh:
        for o in range(head.shape[1]):
            for k in range(head.shape[0]):
                words[wh + o * head.shape[0] + k] = np.float16(head[k, o]).view(np.uint16)
                touched += 1
    if qkv is not None and wq:
        # attention: tile (out//16, in//16) at ((out//16)*32 + in//16)*256, out-major inside
        outs = qkv.shape[1]
        for out in range(outs):
            for k in range(qkv.shape[0]):
                idx = wq + ((out // 16) * 32 + k // 16) * 256 + 16 * (out % 16) + (k % 16)
                words[idx] = np.float16(qkv[k, out]).view(np.uint16)
                touched += 1
    return touched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=ROOT / "local/small")
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--weights", type=Path, default=CROP, help="reference safetensors")
    ap.add_argument("--apply", action="store_true", help="rewrite the asset and the plan")
    args = ap.parse_args()

    model = args.model.resolve()
    plan = model / "manifest" / "port_network_runtime.txt"
    index_path = model / "manifest" / "cold_resources_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))

    logical = f"split16_b{args.block}_weights"
    record = index[logical]
    asset = model / "assets/cold" / Path(record["path"]).name
    if not asset.is_file():
        asset = model / record["path"]

    we, wm, wc, wq, wp, wh, _ = read_plan_offsets(plan, args.block)
    ref = reference_tensors(args.weights.resolve(), args.block)
    words = np.frombuffer(asset.read_bytes(), dtype="<u2").copy()
    touched = rebuild(words, args.block, we, wm, wc, wq, wp, wh, ref)

    print(f"block {args.block}: we={we} wm={wm} wc={wc} wq={wq} wp={wp} wh={wh}, {touched} words from {asset.name}")
    print(f"  expand_in {ref[0].shape}  expand_hidden {ref[1].shape}  contract {ref[2].shape}"
          f"  qkv {None if ref[3] is None else ref[3].shape}")
    if not args.apply:
        print("  (dry run -- pass --apply to write)")
        return 0

    import hashlib
    import re

    data = words.tobytes()
    digest = hashlib.sha256(data).hexdigest()
    out = asset.with_name(f"{digest}__{logical}.fp16.bin")
    out.write_bytes(data)
    record["path"] = str(out)
    record["sha256"] = digest
    index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    text = plan.read_text(encoding="utf-8")
    text = re.sub(r'(BUFFER "' + re.escape(logical) + r'" \d+ ")[^"]*(")',
                  lambda m: m.group(1) + "../assets/cold/" + out.name + m.group(2), text)
    plan.write_text(text, encoding="utf-8")
    print(f"  wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
