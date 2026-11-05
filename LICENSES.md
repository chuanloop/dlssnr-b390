# Licenses, provenance and acknowledgements

本包 `NR-INTEL / dlssnr-b390` 以 MIT 发布，根 `LICENSE` 载有两位版权人：`mochizuki0323`（上游）与 `Intel DLSS5 NR contributors`（本项目）。下面逐项列出随包分发的第三方、派生关系、外部依赖与不复分发的资产。

## 1. 随包分发的第三方代码

| 路径 | 内容 | 上游 | 许可 | 许可原文 |
|---|---|---|---|---|
| `third_party/dlssnr_amd_model_tools/` | 模型解包/检视/打包工具 `inspect_nr.py`、`unpack_{vit,preblock,postblock,splitswin,swin_family}.py`、`pack_model.py`，描述符与张量清单 `descriptor-{small,vk102}.json`、`model-files-{small,vk102}.{txt,sha256}`，以及 `PROVENANCE.json` | [mochizuki0323/DLSSNR-AMD](https://github.com/mochizuki0323/DLSSNR-AMD) | MIT，© 2026 mochizuki0323 | `third_party/dlssnr_amd_model_tools/LICENSE`；副本见 [THIRD-PARTY/mochizuki-DLSSNR-AMD-LICENSE.txt](THIRD-PARTY/mochizuki-DLSSNR-AMD-LICENSE.txt) |
| `source/network-fp16-20261027/`、`source/network-small-20261028/` | 已采纳的 FP16 网络源码与还原配方 | 同上 | MIT，© 2026 mochizuki0323 | 两个目录各自的 `LICENSE` |
| `source/network/`、`source/network-vk102/`、`source/network-small-v2/`、`source/vulkan/` | 本项目在 Xe2/XMX 上的 Vulkan 实现 | 由上述上游网络的父级 shader 逐级改写；每个 kernel 头部保留父级路径、部署 SPV 哈希与复编一致性结论 | MIT（同上） | 同根 `LICENSE` |
| `source/bridge/nr_hdr_proxy.hlsl`、`source/bridge/src/nr_hdr_proxy.*` | 场景线性 R11G11B10 ↔ SDR NR ↔ R11G11B10 通路 | 沿用 AMD 实现的色彩约定；该实现（`windows/src/core/nr_renodx.cpp`）承自 RenoDX | MIT，© 2025 Carlos Lopez Jr. | [THIRD-PARTY/renodx-LICENSE.txt](THIRD-PARTY/renodx-LICENSE.txt) |

`PROVENANCE.json` 记录上游工具快照 `939b6072b…` 的逐文件 SHA256 与全部本地改动；工具快照 commit 不代表后续每项 shader 研究都出自该版本。

## 2. 运行时抽取产物（权利不属本项目）

`game-files/dlssnr-b390/assets/spv/` 下的 39 个 `__small-*.spv` 是 NVIDIA 官方运行时 `nvngx_dlssnr.dll`（310.9.16.0）的 SPIR-V 产物，随包提供**仅为让内核能直接运行**，本项目不主张任何权利，权利仍归 NVIDIA。它们未经张量重排，可由同版本 DLL 配合 `tools/build_assets.py` 重新产出；若你要公开再分发或对权利归属有疑虑，删除这 39 个文件、用你自己那份 DLL 重建即可——内核按 `frozen-files.sha256` 逐条校验，哈希一致即可运行。

同目录另 3 个 `port_network_bridge_*.spv` 由本仓库 `source/` 编译产出，属本项目 MIT 范围。

本包不含权重：`game-files/dlssnr-b390/assets/cold/` 为空目录，70 个 `.bin` 属 NVIDIA 模型数据，需自备，`verify.ps1` 按 `frozen-files.sha256` 的 cold 条目校验。

## 3. 外部依赖（不随包分发）

| 依赖 | 用途 | 许可与说明 |
|---|---|---|
| OptiScaler v0.9.4（commit `7534ad0`）源码检出 | `source/bridge/CMakeLists.txt` 要求 `OPTISCALER_SOURCE` 指向它，取 `external/nvngx_dlss_sdk` 头文件与 `OptiScaler/library/detours/detours.lib` | 适用其自身条款；本包不复制其源码、二进制或 SDK 头。它自带的 NVIDIA NGX SDK、Microsoft Detours、MinHook、Dear ImGui 由其自身分发，本包不含，故不在此重复其通知 |
| NVIDIA DLSS-NR SDK 头文件、`nvngx_dlssnr.dll`、模型权重 | 编译期声明、运行期由用户自备 | NVIDIA 条款，禁止再分发；`.gitignore` 已排除 `nvngx*.dll` |
| Khronos Vulkan-Headers（Apache-2.0）、glslang 16.5.0（BSD-3-Clause 类） | 构建期编译着色器 | 不含二进制；源码注释记录了编译版本 |
| Windows SDK（D3D12 / DXGI / D3DCompiler）、MSVC 运行时 | 构建与运行 | Microsoft 条款；VC++ 可再发行包由用户自行安装 |
| Python 3 标准库、numpy ≥ 2.0（`requirements-model.txt`） | `panel/` 与 `tools/` | PSF-2.0 / BSD-3-Clause |

## 4. 权利与背书边界

No vendor endorsement or model redistribution right is asserted. Game and third-party visual rights are retained by their respective owners. Runtime fixtures, test outputs, model data, native binaries and compiled shaders are excluded unless listed above. This is a community effort; it is not an NVIDIA, AMD or Intel SDK, certification or endorsement.

## Optimized Vulkan backend attribution (2026-10-29)

The current Vulkan backend substantially references mochizuki0323 / DLSSNR-AMD (望月), including network organization and producer/consumer layouts, beyond extraction tools. The boundaries of that migration are described below. Both new shader directories retain the upstream MIT notice.

## 望月 / DLSSNR-AMD 来源与迁移边界

本 Vulkan 后端大量参考 [mochizuki0323/DLSSNR-AMD](https://github.com/mochizuki0323/DLSSNR-AMD)（中文讨论中称“望月的 AMD 项目”）。这是社区实现，不代表 AMD 官方贡献、认证或 SDK。

来源包括模型提取/解包及网络结构理解，也包括 shader 中 Swin、attention、FFN、DS/UPS、尾部折叠的组织方式，以及让生产者按下个消费者所需布局输出、减少中间张量落地的思路。仓库 `third_party/dlssnr_amd_model_tools/PROVENANCE.json` 记录了工具快照；该工具快照 commit 不代表后续每项 shader 研究都来自同一个版本。公开代码中保留了原有来源注释，根 LICENSE 与第三方 LICENSE 保留 Copyright (c) 2026 mochizuki0323。

Intel 专项工作是把这些思路落实为 Xe2/XMX 的具体 cooperative-matrix 映射：SG32 的 owner、A/B fragment 方向、权重返回寄存器、SLM 发布/消费、舍入点与驱动调度。AMD 的 wave/MFMA/FP8 packing 不能直接复制到 XMX；C32 的寄存器方案也不能无条件扩展到 C128/C256/C512。

我们没有把 AMD FP8 的近似数学当成此 FP16 包的质量保证。已采纳 FP16 路线保留对应父臂的运算顺序和精度约束，以具体输入下的数值门和同批计时验收。模型版权与 MIT 源码许可分开：用户自行提供支持的模型 DLL，仓库不分发权重或 NVIDIA DLL。

## 致谢

| 来源 | 关系 |
|---|---|
| **mochizuki0323 / DLSSNR-AMD（望月）** | Vulkan NR 后端结构、生产/消费调度、通用输入接入与模型工具，MIT |
| **RenoDX**（Carlos Lopez Jr.） | HDR 处理通路的来源血统，MIT |
| 社区项目 **dlssnr on intel** | Intel 平台 DLSS-NR 移植与集成思路参考 |
| B 站 UP 主 **灰葉白鸠** | 提供技术支持 |

许可原文位置汇总：`THIRD-PARTY/*.txt`（上游通知副本）、`third_party/dlssnr_amd_model_tools/LICENSE`、`source/network-*/LICENSE`、根 `LICENSE`。
