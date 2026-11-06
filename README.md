# Intel DLSS-NR for The Witcher 3 — New User Install Guide (v1.3 / `dlssnr-b390`)

> **Download**: this repository holds sources and docs only, and the
> [Releases page → v1.3](https://github.com/chuanloop/dlssnr-b390/releases/tag/v1.3) attaches source archives only.
> **No NVIDIA artifact is redistributed** — not `nvngx_dlssnr.dll`, not the 39 `__small-*.spv` derived from it,
> not the 70 cold weights.

You need: The Witcher 3, an Intel Arc B390 (Xe3 iGPU) with a working driver, and **four things you supply yourself** —
NVIDIA's official `nvngx_dlssnr.dll` runtime, an `OptiScaler` build, your own weight files (`assets\cold`), and the
shaders reproduced from your own copy of that runtime. This repository ships the NR core, the manifest, the dispatch
plans, and install/verify scripts. It contains **no NVIDIA runtime, no shaders derived from it, no weights and no model
data**. Installing needs no compilation; reproducing the shaders once does (`tools\build_assets.py`, Python 3 + glslang,
see section 5).

[中文说明](README.zh-CN.md) · [Release notes](RELEASE-NOTES.md) · [What is bundled](ASSETS.md) · [Source and build](INSTALL.md)

## 1. Before you start

- Windows 10/11 x64. Verified on **Intel Arc B390 (Xe3 iGPU)** with an Intel/OEM release driver.
- The installer defaults to **The Witcher 3 next-gen**: `...\The Witcher 3\bin\x64_dx12\witcher3.exe`, launched in **DX12**. Tested on The Witcher 3 and Black Myth: Wukong (DX12); other games are for you to test.
- Quit the game. Extract the whole archive to a normal folder; **do not run scripts from inside the zip**, and do not pull `install.ps1` out on its own.
- **You must supply (not included):**

  | Item | Requirement | Where |
  |---|---|---|
  | `nvngx_dlssnr.dll` | official 310.9.16.0, exactly **276,162,560 bytes**, SHA-256 `CDFA6C1E…ABF3ED` | `bin\x64_dx12\` |
  | `OptiScaler` | any recent upstream build (no binary bundled here) | `bin\x64_dx12\` |
  | weights `assets\cold` | **70** `.bin` files, each must match the corresponding `frozen-files.sha256` cold entry | `<x64_dx12>\dlssnr-b390\assets\cold\` |
  | shaders `assets\spv` | **42** `.spv` files (39 `__small-*` + 3 `port_network_bridge_*`), each must match its `frozen-files.sha256` spv entry | `<x64_dx12>\dlssnr-b390\assets\spv\` |

  The NVIDIA runtime is copyrighted, and both the weights and those 39 `__small-*.spv` shaders are derived from it, so
  none of them is redistributed — restore them from your own game folder or backup and reproduce the shaders with your
  own runtime. **A ~5.3 MB file with that name is a harmful legacy forwarder, not the official runtime**; `install.ps1` removes it.

- **About the weights** (deliberately not bundled, to avoid redistribution): the core validates every entry of the frozen list at start-up and fails `init failed` on a missing or mismatched weight, which looks exactly like "NR does nothing".
  - Take them from a copy you already own: an older release (e.g. v3.7's `dlssnr-amd\assets\cold`), your previous `dlssnr-b390\assets\cold`, or your own backup.
  - After copying, run `verify.ps1` and require `cold 70 / 0`.
  - **Re-extraction now matches the frozen list**: the full chain (`extract_model.py` + `pack_b580_weights.py`) reproduces all 70 cold entries byte-for-byte (`--full-model`, or `--blend-scalar` pointing at the 4-byte `ioport_blend_scale` resource, is required). Earlier builds differed on 8 entries (`swin_7/22/23` `*_finaltile`, `swin_8`, `split16_b9/b10/b20/b21`); the packer now writes the consumer layouts directly. See `ASSETS.md`.
- **About the shaders** (also deliberately not bundled): the 39 `__small-*.spv` files are derivatives of shaders from
  NVIDIA's runtime. Reproduce them locally with `tools\build_assets.py` (Python 3 + glslang 16.5.0) against your own
  `nvngx_dlssnr.dll`; `verify.ps1` then compares each file with its `frozen-files.sha256` spv entry. The core only
  checks hashes, so identical bytes from your own copy are enough.
- Leftovers from older naming (`dlssnr-amd\`, `intel-b390-dlssnr\`, `dlssnr-amd.ini`) are cleaned up automatically.
- Do not delete `amd_fidelityfx_loader_dx12.dll`, `amd_fidelityfx_framegeneration_dx12.dll` or `amd_fidelityfx_upscaler_dx12.dll` — the game and the FSR path use them too.

## 2. Install

1. **Right-click `install.ps1` → Run with PowerShell**, or:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\install.ps1
   ```

2. The script scans Steam libraries for `bin\x64_dx12` and prints the target for you to confirm. Override it if needed:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\install.ps1 -GameDir "D:\SteamLibrary\steamapps\common\The Witcher 3\bin\x64_dx12"
   ```

   `-DryRun` previews without touching files; `-SkipVerify` skips the post-install hash pass.

3. Six steps: locate → validate package (core / manifest / 37 ladder tiers; unbundled `assets\cold` and `assets\spv` are reported explicitly) → copy → clean legacy names → run `verify.ps1` → report `OptiScaler.ini` and official-DLL status. `安装完成` at the end means done.
4. No write permission? Run the same command in an **administrator** PowerShell. The script never touches drivers, the registry, TDR or saves.
5. Manual install — copy three things:

   | From | To |
   |---|---|
   | `game-files\dlssnr_core.dll` | `<x64_dx12>\dlssnr_core.dll` |
   | `game-files\dlssnr-b390.ini` | `<x64_dx12>\dlssnr-b390.ini` |
   | `game-files\dlssnr-b390\` | `<x64_dx12>\dlssnr-b390\` |

6. Point OptiScaler at this core. In `<x64_dx12>\OptiScaler.ini`:

   ```ini
   NvngxPath=.\dlssnr_core.dll
   ```

## 3. First launch

- Start from Steam and pick **DX12**.
- Enable **DLSS / upscaling** in the graphics menu. NR processes at the render resolution the game feeds it ("native" = render resolution, not display resolution).
- Press **Insert** for the OptiScaler panel; confirm NR is enabled and routed to `dlssnr_core.dll`.
- After a minute, verify with two files:

  ```
  <x64_dx12>\dlssnr-b390.log             → port: plan read, 222 buffers and 108 passes
                                           then network built at <W>x<H>
  <x64_dx12>\dlssnr-b390\nr_tune_read.txt → created, contains ini=ok tune=ok detail=1.000000 …
  ```

- **This build ships the Small network only** (108 dispatches / 70 cold resources / 222 buffers). There is no Full-232 profile to switch to.
- Tune by editing `dlssnr-b390\nr_tune.txt` — saving applies live, no restart. `nr_tune_read.txt` is what the core actually read back.
- **Ctrl+F11 toggles the int4 mixed path, it is not the NR on/off switch.** The image changes when you toggle it; that does not mean NR was off.
- No web panel and no custom filter editor: settings live in the ini and `nr_tune.txt`.

## 4. Uninstall / update

- Quit the game and delete four items (nothing is installed outside the game folder):

  ```
  <x64_dx12>\dlssnr_core.dll
  <x64_dx12>\dlssnr-b390.ini
  <x64_dx12>\dlssnr-b390.log
  <x64_dx12>\dlssnr-b390\
  ```

  Then restore `NvngxPath` in `OptiScaler.ini` to the official value.
- This package keeps **no install manifest and no automatic backup** (`install.ps1` overwrites same-named files). Back up `dlssnr_core.dll` and `dlssnr-b390\` before upgrading.
- To update, uninstall as above, then install the new version. Never stack two versions.
- `nvngx_dlssnr.dll` and OptiScaler are yours — do not delete them when removing this package, or official DLSS breaks too.

## 5. Troubleshooting

Read the log first; do not stack versions and hope:

| Symptom | Meaning |
|---|---|
| `dlssnr-b390.log` empty | Core never loaded: wrong folder, `NvngxPath` not pointing at `dlssnr_core.dll`, or a 5.3 MB fake `nvngx_dlssnr.dll` squatting on the name |
| `port: init failed … cannot read …__small-<hash>-<hash>.spv` | A frozen shader entry is missing — either the v1.0 stale-entry defect (use the v1.3 core) or you have not reproduced `assets\spv` yet |
| `port: init failed … requires exact build-frozen manifest/runtime` | Data was regenerated or replaced (e.g. weights re-extracted with newer tools); restore the shipped `dlssnr-b390\manifest` and `assets\cold` |
| `plan read` but the image is unchanged | OptiScaler is not routing here: check `NvngxPath`, the in-game DLSS preset, and other proxy DLLs |
| Stutter or NR pauses | Save the log and quit; do not mask it with a larger TDR, overclocking or backend flipping |

Offline integrity check:

```powershell
powershell -ExecutionPolicy Bypass -File .\verify.ps1 -Path "<x64_dx12>"
```

Expect `cold 70/0`, `spv 42/0`, both identities `✔`, and a final `✔ 全部文件与 core 内置身份一致`.
If the weights or shaders are not in place yet, those two lines read `通过 0 / 失败 0（缺失 N，未随包分发）`;
that is the expected state for this repository, not a corrupted install. Re-run once you bolted them on.

## 6. What is in v1.3

- `dlssnr_core.dll` 5,437,700 bytes, carrying the first-frame-cut crash fix; embedded identity manifest `83ded71e…` / runtime `2a0ed57d…`, 112 frozen entries (70 cold + 42 spv), 37 dispatch ladder tiers (up to 1280x800).
- Fully renamed: `dlssnr-b390\`, `dlssnr-b390.ini`, `dlssnr-b390.log`; `dlssnr-b390` ×29 and the old name ×0 inside the binary.
- v1.3 fixes "first frame cut → `0xC0000005` / device removed": replaying the recorded command-list state no longer replays `SetDescriptorHeaps`. That moves the working range from "the few scenarios that never cut frames" to every DX12 game that does. Details in [RELEASE-NOTES.md](RELEASE-NOTES.md).
- Image baseline is still V3.7's Small network (108 dispatches): **neither the 70 weights nor the 42 shaders are bundled**; hashes are in `frozen-files.sha256`. Since v1.3 nothing distributed here comes from NVIDIA.
- "New user" means no developer environment required, not universal compatibility with every Intel GPU or game.

## Credits and licensing

- **mochizuki0323 / DLSSNR-AMD** (MIT, copyright 2026 mochizuki0323): the Vulkan NR backend, producer/consumer scheduling and generic input hookup draw heavily on that project and its OptiScaler integration. Upstream notices are retained in `LICENSE`, `LICENSES.md` and `third_party/dlssnr_amd_model_tools/LICENSE`.
- Community project **dlssnr on intel**: reference for porting and integrating DLSS-NR on Intel platforms.
- **灰葉白鸠 (Bilibili)**: for technical support.
- Not redistributed: NVIDIA `nvngx_dlssnr.dll`, **the weights and model data extracted from it (`assets\cold`)**, **the shaders derived from it (the 39 `__small-*.spv` under `assets\spv`)**, OptiScaler binaries, FSR XMX / XeSS FG components, The Witcher 3 game files. Any public release must follow each component's own distribution terms.
- Release guard: `tools\check_no_nv.ps1 -Path .` (tree) or `-Path xxx.zip` (archive); it exits 1 on any of the above.
- VC++ redistributable (only if your system lacks it): https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
