# Release notes

## v1.1 — 2026-11-06 — `dlssnr-b390`（巫师3 DX12 / Arc B390）

首个在游戏内**实测生效**的改名版。

### 修复

**后端初始化失败（根因）** —— 编进 `dlssnr_core.dll` 的冻结资源清单里残留两条过期着色器条目，
指向不存在的 `…__small-bc629c9b8e24-e4efe71639f.spv` 与 `…__small-55e04be51248-10a2916c26b3.spv`。
`checked_plan()` 在读取派发计划前先逐条校验冻结清单，`read_bytes` 读不到文件即抛出：

```
[nr] port: init failed after 0.7 s: cannot read ...\dlssnr-b390\manifest\../assets/spv/e4efe716…__small-bc629c9b8e24-e4efe71639f.spv
[nr] runtime: nrvk_backend_init failed (-2) … no frame will be recorded
```

结果 NR 从头到尾 passthrough：画面无变化、`nr_tune_read.txt` 不生成。
v1.0 的内核同样带这两条（其 `frozen-files.sha256` 也是 112 条），因此同样不可能初始化成功——同一个根因。

修复：从 `generated/port_network_bridge_backend_frozen.inc` 删除这两条，`frozen_manifest_sha` /
`frozen_runtime_sha` 保持不变，重新编译。清单从 112 条降为 110 条，与原版 V3.7 二进制一致
（原版内核中 `e4efe716` 出现 0 次，重编版此前是 3 次）。

### 改名

- 目录 `dlssnr-amd\` → `dlssnr-b390\`；配置 `dlssnr-amd.ini` → `dlssnr-b390.ini`；日志 `dlssnr-b390.log`
- 后端源码字符串 `dlssnr-b390` ×29、`dlssnr-amd` ×0（V3.7 原版为 `dlssnr-amd` ×29）
- 资源与清单内容未做任何改动，冻结身份仍为 manifest `bf6c087a…` / runtime `c168a078…`
- 安装脚本会自动清掉 `dlssnr-amd` 遗留目录/ini，以及错误的 5.3MB `nvngx_dlssnr.dll`

### 验证

- 离线：`frozen-files.sha256` 110 条全部命中；`verify.ps1` 结果 `cold 70/0`、`spv 40/0`、身份双 ✔（权重文件为调试副本，见下）
- 二进制：内置 110 条资源路径在完整数据中全部存在（缺失 0）
- 游戏内：巫师3 次世代版 DX12 出图，画质档位与 `nr_tune_read.txt` 正常，日志出现 `port: plan read`
- 游戏范围：已验证巫师3、黑神话：悟空等 DX12 游戏；其他游戏需自行测试。
- 硬件范围：仅验证 Intel Arc B390（Xe3）核显 + 正式版驱动，其他 Intel GPU 未测
- 不做帧生成：NR 只负责光线重建，FG 由 OptiScaler 走 XeFG / FSRFG

### 权重可自行复现（v1.1 起）

- 包内 `tools/` 带完整抽取与打包链：从官方 `nvngx_dlssnr.dll` 出发，`extract_model.py` + `pack_b580_weights.py`
  可逐字节复现全部 70 条 cold 资源，与 `frozen-files.sha256` 一致
- 需带 `--full-model`，或 `--blend-scalar` 指向那份 4 字节的 `ioport_blend_scale`
- 早期版本有 8 条不一致（swin_7/22/23 的 finaltile、swin_8、split16_b9/b10/b20/b21），现已改为打包阶段直接产出内核消费布局，规则见 `ASSETS.md`
- 复现仅为你自己的合法副本服务；权重仍不得随包分发

### 包内容变化

- **权重不分发**：`assets\cold` 留空，需用户从自己已有的 v3.7/v1.0 目录或备份复制 70 个 `.bin`（哈希见 `frozen-files.sha256`）
- 同时不再分发：NVIDIA `nvngx_dlssnr.dll`（310.9.16.0，276,162,560 B）、OptiScaler、FSR/XeSS 组件、游戏文件
- `source/` 保留改名后的完整源码（不含 `research-archive`），并附上修正后的 frozen inc
- 包体由 287 MB 降到 41 MB

## v1.0 — 2026-11-05

初版 `dlssnr-b390` 打包。内核受上述过期冻结条目影响，游戏内始终无法初始化；文件哈希清单本身也是 112 条，
包含两条指向不存在文件的记录，`verify.ps1` 会直接失败。

## v3.7 — 2026-11-05

可正常工作的 `dlssnr-amd` 版本，也是 v1.1 的数据与画质基准：同一份 85 权重 / 42 着色器 / 3 清单，
同一冻结身份，同一 `small` 网络（108 次派发）。v1.1 即「V3.7 的行为 + b390 的命名」。

## 致谢与版权

- **mochizuki0323 / DLSSNR-AMD**（MIT，copyright 2026 mochizuki0323）：Vulkan NR 后端、生产消费调度与通用输入接入的主要参考；上游许可与声明保留在 `LICENSE`、`LICENSES.md`、`third_party/dlssnr_amd_model_tools/LICENSE`。
- 社区项目 **dlssnr on intel**：Intel 平台 DLSS-NR 移植与集成思路参考。
- **B 站 UP 主 灰葉白鸠**：提供技术支持。
- 不随包分发 NVIDIA `nvngx_dlssnr.dll`、由其抽取的 `assets\cold` 权重、OptiScaler 二进制、FSR XMX / XeSS FG 组件与游戏文件；公开发布须遵循各组件自身的分发条件。
