# 包内容与来源 / What is bundled

## 随包提供

| 路径 | 内容 | 说明 |
|---|---|---|
| `game-files\dlssnr_core.dll` | 5,431,431 B<br>`d7a21d612548953b7b623dd79edea6365b2d25f9bfdeee428d0e7fbddae85877` | NR 内核，内置冻结身份 `bf6c087a…` / `c168a078…` |
| `game-files\dlssnr-b390.ini` | 17,619 B | 后端配置（分辨率档位、int4、日志级别等），游戏内改即生效 |
| `game-files\dlssnr-b390\assets\cold\` | **空目录** | 权重不分发，需自行准备（见下节） |
| `game-files\dlssnr-b390\assets\spv\` | 42 个 `.spv` | 网络着色器（含 gather/scatter/scatter32）。其中 **39 个 `__small-*.spv` 来源为官方运行时的离线抽取，仅供参考**；其余 3 个 `port_network_bridge_*.spv` 为本项目编译产物 |
| `game-files\dlssnr-b390\manifest\` | `port_network_manifest.json`、`port_network_runtime.txt`、`nr_tune.txt` 共 3 个 | 派发计划与运行参数 |
| `game-files\dlssnr-b390\nr_tune.txt` | 运行参数模板 | 可直接改，游戏内热生效 |
| `game-files\dlssnr-b390\nr_present_probe.exe` | 探针工具 | 排查用，非必需 |
| `generated\port_network_bridge_backend_frozen.inc` | 110 条冻结条目 | **已修正版**，重编时必须使用这一份 |
| `frozen-files.sha256` | 110 行 | 与内核内置校验完全一致，供 `verify.ps1` 使用；cold 70 条即权重应满足的哈希 |
| `frozen-identity.txt` | 2 行 | manifest / runtime 身份 |
| `source\` | 336 个文件 / 4.1 MB | 改名后的 V3.7 源码（已剔除 `research-archive`） |

包体合计约 **41 MB**（不含权重；`game-files` 约 9.3 MB）。

## 不随包分发（需你自备）

| 项目 | 原因 |
|---|---|
| `nvngx_dlssnr.dll` 310.9.16.0（276,162,560 B，SHA-256 `CDFA6C1E…`） | NVIDIA 版权运行时，须自行从游戏原版/备份取回 |
| **`assets\cold` 的 70 个权重 `.bin`** | 由上述 DLL 离线抽取得到，属 NVIDIA 模型数据，不再分发 |
| OptiScaler 及其依赖 DLL | 第三方作品，从上游获取 |
| FSR XMX / XeSS FG 组件 | 第三方作品 |
| 巫师3 游戏文件（`witcher3.exe`、`dx12filelist.txt` 等） | 商业游戏内容 |
| `source\research-archive` | 仅研究用归档，构建不需要 |

## 权重从哪来

`assets\cold` 由官方 `nvngx_dlssnr.dll`（310.9.16.0）**离线抽取**得到，运行时以数据形式读取，不加载、不执行 NVIDIA 的任何代码。
由于它属于 NVIDIA 的模型数据，本包**不再分发**，需要你从自己已有的那份里取：

1. **旧版本目录**：v3.7 的 `dlssnr-amd\assets\cold`、v1.0 的 `dlssnr-b390\assets\cold`，或你之前装到游戏里的那份；
2. 复制到 `<x64_dx12>\dlssnr-b390\assets\cold\`（70 个 `.bin`）；
3. 跑 `verify.ps1`，`cold 70 / 失败 0` 才算对，哈希以 `frozen-files.sha256` 的 cold 条目为准。

**可以从 DLL 端到端复现**（v1.2 起，逐条实测）：

- 只跑 `extract_model.py --raw-only`：得到的是原始张量布局，与冻结清单只有 8 条能逐字节命中（旧用法，别再当基准）；
- 跑完整链路（`extract_model.py` 去掉 `--raw-only` + 内置的 `pack_b580_weights.py`，需 numpy）：
  产出与 `frozen-files.sha256` 的 cold 条目 **70/70 逐字节一致**，工具内部也逐条 assert 通过。

早期版本有 8 条与冻结清单不同，原因是打包时按"仓库基准"排布、而不是按内核消费排布。现已全部在打包阶段直出：

| 条目 | 原先的问题 | 修正（`pack_b580_weights.py`） |
|---|---|---|
| `split16_b9 / b10 / b20 / b21` | L0 的 we/wm/wc 用 `ffwd_maps` 位交织后再 TILE16，与消费端读法不符 | `split16_l0()`：raw → 逻辑张量（`ffwd_maps`）→ 按内核 16×16 out-major 分块写入（组步长 `outs*ins`，in 块步长 256，out 块步长 `(ins/16)*256`）；与 `pack_split16_ffn.py` 里用单字脉冲标定过的 `slot()` 一致 |
| `split16_b10`（补充） | split_last 的 L3/L4 被多 tile 一次 | small 的 split_last 不再 tile L3，L4 保持 plain `blob[o*in+k]` |
| `swin_7 / 22 / 23` 的 `*_finaltile`、`swin_8` | qkv 被当成 tiles（应为 plain rows）；`residual_scale` 的表被 8 个 half 的零前导顶偏，每个通道读到 `c-8` 的标量 | `pack_swin()`：qkv 不再 tile、`attn_out_proj` 仍 tile；`residual_scale` 前移 8 个 half。`swin_8` 没有 finaltile 配方，其 out_proj 保持 plain |

- `weight-transforms.json` 里 3 条 `*_finaltile` 配方已删除——它们原本只是"把 out_proj tile 一次再让补丁 tile 回来"的中间步骤，现在改名直接由打包侧完成。
- `patch_small_c256.py` 保留为历史工具（对已安装 `assets\cold` 的事后修补），逻辑与上面一致，不再需要跑。

关于 blend scalar（`--full-model` / `--blend-scalar` 为什么必需）：

- DLL 内 small 的 `block30.layer0.blend_scale` 是 `23 3a`（≈0.767），而内核冻结的 `ioport_30_ioport_blend_scale`
  是 `eb 39`（≈0.740，与 full 的 `ioport_70_…` 同值），两者差 3.7%；
- 内核 init 时按 `frozen-files.sha256` 逐字节校验每个 cold 资源，small 的这份标量必须是 `b0b8e401…` 那个 4 字节文件；
  直接用 DLL 的 `372f374e…` 会被判 `init failed`。small 是从 full 裁出来的子网，这个跨 profile 共享的标量取的是 full 的值，
  所以打包必须 `--full-model`（或 `--blend-scalar` 指向那份 4 字节资源）。

## 着色器从哪来（重要说明）

`assets\spv` 的 42 个文件分两类，版权性质不同：

| 文件 | 数量 | 来源 |
|---|---|---|
| `port_network_bridge_*.spv` | 3 | 本项目自建（由 `source\` 中的 bridge 着色器源码编译） |
| `__small-<hash>-<hash>.spv` | 39 | **来源为官方运行时 `nvngx_dlssnr.dll`（310.9.16.0）的离线抽取，仅供参考** |

关于这 39 个文件：

- 它们是 NVIDIA 运行时的 SPIR-V 产物，**本包保留它们仅为让内核能直接跑通，不主张任何权利，仅供参考**；
- 与权重不同，着色器**没有经过 TILE 重排**，因此理论上可由同版本 DLL 用 `tools\build_assets.py` 重新编译出来；
- 若你打算公开再分发，或对这批文件的权利归属有顾虑，请**删除 `assets\spv` 下的 39 个 `__small-*.spv`，自行从你的官方 DLL 构建**，并把结果放回同一目录；内核会按 `frozen-files.sha256` 的 spv 条目校验，哈希一致即可运行；
- 只保留 3 个 bridge 着色器时，内核会因为 spv 条目缺失而 `init failed`，这是预期行为，不是安装错误。

换言之：**权重必须来自你自己已有的那份，着色器可以按需要自行重建或替换。**

## 完整性校验

```powershell
powershell -ExecutionPolicy Bypass -File .\verify.ps1 -Path "<...>\bin\x64_dx12"
powershell -ExecutionPolicy Bypass -File .\verify.ps1            # 校验包内 game-files
```

## 致谢与版权

| 来源 | 关系 |
|---|---|
| **mochizuki0323 / DLSSNR-AMD**（MIT） | Vulkan NR 后端、调度与输入接入的主要参考；许可保留在 `LICENSE` / `LICENSES.md` / `third_party/dlssnr_amd_model_tools/LICENSE` |
| 社区项目 **dlssnr on intel** | Intel 平台 DLSS-NR 移植与集成思路参考 |
| **RenoDX**（MIT，© 2025 Carlos Lopez Jr.） | HDR 通路的来源血统；通知保留在 `THIRD-PARTY\renodx-LICENSE.txt` |
| **B 站 UP 主 灰葉白鸠** | 提供技术支持 |

`LICENSES.md` 逐项列出随包分发的上游代码、派生关系、外部依赖与不复分发的资产；通知原文在 `THIRD-PARTY\`、`third_party\dlssnr_amd_model_tools\LICENSE`、`source\network-*\LICENSE`。
本包不含 NVIDIA `nvngx_dlssnr.dll`、由其抽取的 `assets\cold` 权重、OptiScaler 二进制、FSR XMX / XeSS FG 组件与游戏文件。
`assets\spv` 中的 39 个 `__small-*.spv` 系官方运行时离线抽取的着色器，**来源为官方运行时的离线抽取，仅供参考**，权利归原厂商；如需自行公开再分发，请移除并用 `tools\build_assets.py` 从你自己的 DLL 重建。
