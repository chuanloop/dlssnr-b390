#!/usr/bin/env python3
"""Extract and verify the supported NR model from a user-supplied DLL.

Offline only. The DLL is parsed as data, never loaded or executed. No DLL,
CUDA kernels or model bytes are downloaded or distributed by this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_TOOLS = ROOT / "third_party" / "dlssnr_amd_model_tools"
PROFILES = json.loads((ROOT / "tools/model_profiles.json").read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def run_tool(name: str, *arguments: object) -> None:
    subprocess.run([sys.executable, "-B", str(MODEL_TOOLS / name),
                    *(str(x) for x in arguments)], check=True)


def extract(dll: Path, output: Path, requested_profile: str = "auto") -> dict:
    sys.path.insert(0, str(MODEL_TOOLS))
    from inspect_nr import PE, checked, load_input, parse_weights

    data = load_input(dll)
    digest = hashlib.sha256(data).hexdigest()
    # alt_dll_sha256 accepts an alternative container DLL (e.g. a different
    # driver bundle shipping the same model resource); the extracted tensors
    # still have to pass the full pinned tensor-hash chain below.
    matching = [key for key, value in PROFILES.items()
                if value["dll_sha256"] == digest or digest in value.get("alt_dll_sha256", ())]
    if not matching or requested_profile not in ("auto", matching[0]):
        raise ValueError(f"Unsupported model DLL/profile: SHA256 {digest}; supported: Small 310.9.16.0, VK102 310.8.0.0")
    profile_key = matching[0]
    profile = PROFILES[profile_key]
    resource_name = profile["resource"]
    pe = PE(data)
    resources = [x for x in pe.resources() if resource_name in x["path"]]
    if len(resources) != 1:
        raise ValueError(f"Expected one {resource_name} resource, found {len(resources)}")
    resource = resources[0]
    weights = checked(data, resource["file_offset"], resource["size"])
    records = parse_weights(weights)
    blocks = sorted({int(m.group(1)) for r in records
                     if (m := re.match(r"block(\d+)\.", r["name"]))})
    if len(records) != profile["records"] or blocks != list(range(profile["blocks"])):
        raise ValueError("Model record/block layout differs from this release")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be a new or empty directory: {output}")

    extracted = output / "extracted"
    inventory = extracted / "inventory"
    target = inventory / "weights"
    target.mkdir(parents=True)
    # The upstream Swin unpacker also reads the original model envelope.
    (inventory / "weights_ht.bin").write_bytes(weights)
    for record in records:
        name = record["name"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
            raise ValueError(f"Unsafe record name: {name!r}")
        (target / f"{name}.bin").write_bytes(
            checked(weights, record["payload_offset"], record["payload_size"]))
    (inventory / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "source_name": dll.name, "dll_sha256": digest,
        "weight_record_count": len(records), "block_ids": blocks,
        "weights": records,
    }, indent=2) + "\n", encoding="utf-8")
    graph = extracted / "graph"
    graph.mkdir()
    shutil.copyfile(MODEL_TOOLS / profile["descriptor"], graph / "descriptor.json")
    # Parse only the model resource: do not emit the input DLL or embedded CUDA code.
    del weights, data

    run_tool("unpack_swin_family.py", "--artifacts", extracted,
             "--output", extracted / "unpacked", "--max-c", 256)
    run_tool("unpack_splitswin.py", "--artifacts", extracted,
             "--output", extracted / "unpacked-splitswin")
    run_tool("unpack_vit.py", "--artifacts", extracted,
             "--output", extracted / "unpacked-vit")
    run_tool("unpack_preblock.py", target / "block0.layer0.layer.bin",
             extracted / "unpacked-preblock")
    run_tool("unpack_postblock.py", target / f"block{profile['post_block']}.layer0.layer.bin",
             extracted / "unpacked-postblock")

    checksums = {}
    for line in (MODEL_TOOLS / profile["tensor_hashes"]).read_text(encoding="utf-8").splitlines():
        if line.strip():
            expected, name = line.split(None, 1)
            name = name.strip()
            actual = file_hash(extracted / name)
            if actual != expected:
                raise ValueError(f"Extracted tensor hash mismatch: {name}")
            checksums[name] = actual
    if len(checksums) != profile["tensor_files"]:
        raise ValueError(f"Expected {profile['tensor_files']} checked files, found {len(checksums)}")
    packed = output / profile["packed_model"]
    run_tool("pack_model.py", "--root", extracted,
             "--list", MODEL_TOOLS / profile["tensor_list"], "--out", packed,
             "--verify", MODEL_TOOLS / profile["tensor_hashes"])
    receipt = {
        "model": profile["name"], "profile": profile_key, "dll_sha256": digest,
        "resource": resource_name, "blocks": profile["blocks"], "records": len(records),
        "verified_tensor_files": len(checksums), "tensor_sha256": checksums,
        "packed_model": {"file": packed.name, "bytes": packed.stat().st_size,
                         "sha256": file_hash(packed)},
        "dll_executed": False, "network_downloads": False,
        "scope": "raw extracted tensors and NRMODEL1 only; not Vulkan-ready assets or a game acceptance gate",
    }
    (output / "extraction-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["auto", *PROFILES], default="auto")
    parser.add_argument("--dll", type=Path, required=True,
                        help="Local nvngx_dlssnr.dll or ZIP containing it")
    parser.add_argument("--out", type=Path, required=True,
                        help="New/empty local output directory; generated files must not be committed")
    parser.add_argument("--raw-only", action="store_true",
                        help="Only verified raw tensors/NRMODEL1; skip NumPy-based B580 packing")
    parser.add_argument("--full-model", type=Path, help="Optimized Full model directory for the Small shared blend scalar")
    args = parser.parse_args()
    output = args.out.resolve()
    if not args.raw_only and importlib.util.find_spec("numpy") is None:
        parser.error("Install model-tool requirements in your venv: python -m pip install -r requirements-model.txt")
    if output.exists() and any(output.iterdir()):
        parser.error("--out must be a new or empty directory")
    try:
        receipt = extract(args.dll.resolve(), output if args.raw_only else output / "model", args.profile)
        if not args.raw_only:
            subprocess.run([
                sys.executable, "-B", str(Path(__file__).with_name("pack_b580_weights.py")),
                "--profile", receipt["profile"],
                "--extracted", str(output / "model" / "extracted"),
                "--out", str(output / "assets" / "cold"),
                "--manifest", str(output / "manifest"),
                *(["--full-model", str(args.full_model.resolve())] if args.full_model else []),
            ], check=True)
            cold = json.loads((output / "manifest" / "cold_resources_index.json").read_text(encoding="utf-8"))
            summary = {
                "status": "MODEL_WEIGHTS_PREPARED", "profile": receipt["profile"], "verified_tensors": receipt["verified_tensor_files"],
                "dll_sha256": receipt["dll_sha256"], "cold_resources": len(cold),
                "cold_bytes": sum(x["bytes"] for x in cold.values()),
                "generated_weights_are_local_only": True,
                "not_included": ["developer image fixtures", "compiled network shaders", "runtime execution plan"],
                "scope": "model weights ready; runtime shader/plan preparation and installation are separate",
            }
            (output / "model-preparation.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(summary, indent=2))
            return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in receipt.items() if k != "tensor_sha256"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
