# Release notes

## v1.3 — 2026-11-06 — 首帧切帧崩溃修复；分发不再含任何 NVIDIA 产物

一句话：**"第一次切帧就崩"（`0xC0000005` / 设备移除）已修好**；发布形态改为纯源码包，仓库与
Release 附件都不含 `nvngx_dlssnr.dll`、不含由它派生的 39 支 `__small-*.spv`、不含 70 个 cold 权重。

> English: fixes the first frame-cut crash (`0xC0000005` / device removed) and ships **no**
> NVIDIA artifacts — no `nvngx_dlssnr.dll`, no derived `__small-*.spv`, no cold weights.

### 一、修掉"第一次切帧就崩"

**症状**：NR 首次成功执行（日志 `[nr] runtime: first frame … -> ran`）之后，游戏遇到第一次帧切分
就退出；事件查看器里是 `0xC0000005`，一部分机器直接 `DXGI_ERROR_DEVICE_REMOVED`。

**根因**：切帧路径要把录制下来的命令列表状态重放给 continuation。旧构建把 `SetDescriptorHeaps`
也一起重放了——堆描述符只对设置它的那条列表有效，重放到 continuation 后不再合法，随后的 `Close`
返回 `E_INVALIDARG`，列表没关就提交，设备被移除，游戏随即写 0。

**修复**：重放组只保留"让命令列表保持可关闭"的那部分，不再重放 `SetDescriptorHeaps`（游戏切帧后
会立刻自己重设堆）。日志新增可核对行：

```
[nr] replay groups that keep a command list closable: N
[nr] … Close hr=0
```

**影响面**：此前只有"不切帧"的少数场景能跑完整，切帧类 DX12 游戏（巫师3、黑神话：悟空等）一进
NR 就崩；修好后这些游戏可以持续运行，兼容面不再取决于游戏是否切帧。

### 二、内核与冻结身份

| 项目 | 旧构建 | v1.3 |
|---|---|---|
| `dlssnr_core.dll` | 5,431,431 B | **5,437,700 B**，SHA-256 `2F581B61…` |
| 冻结清单 | 110 条（cold 70 + spv 40） | **112 条**（cold 70 + spv 42） |
| manifest / runtime 身份 | `bf6c087a…` / `c168a078…` | **`83ded71e…` / `2a0ed57d…`** |
| 派发 ladder | 37 档 | 37 档（未变，上限 1280x800） |

内核是本文档所述修复的重新编译产物，`frozen-files.sha256`、`frozen-identity.txt` 与
`generated/port_network_bridge_backend_frozen.inc` 已同步到同一份身份。

### 三、分发形态：一个 NVIDIA 文件都不带

- 不含 `nvngx_dlssnr.dll`（NVIDIA 版权运行时，310.9.16.0，276,162,560 B）。
- 不含 39 支 `__small-*.spv`（由该运行时派生），也不含 `assets\cold` 的 70 个权重 `.bin`。
- 不含 OptiScaler、XeSS / FidelityFX 二进制，不含任何游戏文件。
- `.gitignore` 已排除 `nvngx*.dll`、`**/assets/spv/*.spv`、`**/assets/cold/*.bin`、`*.dll`、`*.bin`。
- 新增 `tools/check_no_nv.ps1`：发版前扫目录或 zip，命中 NVIDIA 运行时/派生着色器/权重即报错退出。
- 需要自备：`nvngx_dlssnr.dll`、OptiScaler、70 个权重；42 支着色器用你自己那份 DLL 配合
  `tools\build_assets.py` 生成。`verify.ps1` 现在把「未随包」（cold / spv 缺失，只计数）与
  「哈希不符」（真失败）分开报告，不会再因为没带 NV 资产而误判安装损坏。

### 四、试装期一并修掉的四项

1. **NR 面板全灰**：`[RenoDX] Enabled = 1` 会让 RenoDX 主路径锁死 OptiScaler 的 19 项 NR 设置（设计行为）。配置基线改为 `[RenoDX] Enabled = 0`，回到 OptiScaler Route。
2. **每帧 passthrough**：ladder 分辨率表在编译期固化，运行时往 `manifest\ladder\` 加 plan 既不进表也过不了冻结校验。改为让 NR 回到超分前（`RunBeforeSR = true`），输入变成渲染分辨率，1920x1200 输出命中 960x600 档。
3. **`dlssnr-b390.log` 恒为 0 字节**：OptiScaler 先建了同名文件占住句柄。包内预置一行非空占位，内核可正常追加。
4. **两个代理并存**：`dxgi.dll` 与 `winmm.dll` 同时存在会抢载。约定一个目录只留 `setup_windows.bat` 生成的那一个。

### 五、配置基线（本版起固定）

```ini
dlssnr-b390.ini   [RenoDX] Enabled = 0
dlssnr-b390.ini   [RenoDX] NRHookPoint = 0
OptiScaler.ini    [DlssNr] Enabled = true
OptiScaler.ini    [DlssNr] RunBeforeSR = true
```

### 六、验证

- 巫师3 次世代版 DX12，1920x1200 + DLSS 质量档：`port: ladder -> 960x600` → `-> ran`，持续游玩不再崩。
- 黑神话：悟空 DX12：同样出图，切帧路径稳定。
- 离线：`frozen-files.sha256` 112 条与内核内置身份一致；已存在的 spv / cold 逐条哈希通过。
- 硬件范围：仅验证 Intel Arc B390（Xe3）+ 正式版驱动，其他 Intel GPU 未测。
- 不做帧生成：NR 只负责光线重建，FG 交给 OptiScaler 的 XeFG / FSR FG。

## v1.2 — 2026-11-06 — 打包修正

修 `install.ps1` 的目录扫描与旧命名清理，补 2 档 ladder 计划，包体结构不变。

该版 Release 附件（zip）里带有 42 支着色器，其中 39 支 `__small-*.spv` 由 NVIDIA 运行时派生，
**不符合本项目"公开渠道不含 NVIDIA 产物"的要求**，自 v1.3 起不再随包分发，旧附件已撤下。

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

- 离线：`frozen-files.sha256` 110 条全部命中；`verify.ps1` 结果 `cold 70/0`、`spv 40/0`、身份双 ✔
- 二进制：内置 110 条资源路径在完整数据中全部存在（缺失 0）
- 游戏内：巫师3 次世代版 DX12 出图，画质档位与 `nr_tune_read.txt` 正常，日志出现 `port: plan read`
- 游戏范围：已验证巫师3、黑神话：悟空等 DX12 游戏；其他游戏需自行测试
- 硬件范围：仅验证 Intel Arc B390（Xe3）核显 + 正式版驱动，其他 Intel GPU 未测

### 权重可自行复现（v1.1 起）

- 仓库 `tools/` 带完整抽取与打包链：从官方 `nvngx_dlssnr.dll` 出发，`extract_model.py` + `pack_b580_weights.py`
  可逐字节复现全部 70 条 cold 资源，与 `frozen-files.sha256` 一致
- 需带 `--full-model`，或 `--blend-scalar` 指向那份 4 字节的 `ioport_blend_scale`
- 早期版本有 8 条不一致（swin_7/22/23 的 finaltile、swin_8、split16_b9/b10/b20/b21），现已改为打包阶段直接产出内核消费布局，规则见 `ASSETS.md`
- 复现仅为你自己的合法副本服务；权重仍不得随包分发

## v1.0 — 2026-11-05

初版 `dlssnr-b390` 打包。内核受上述过期冻结条目影响，游戏内始终无法初始化；文件哈希清单本身也是 112 条，
包含两条指向不存在文件的记录，`verify.ps1` 会直接失败。

## v3.7 — 2026-11-05

可正常工作的 `dlssnr-amd` 版本，也是 v1.1 的数据与画质基准：同一份权重 / 42 着色器 / 3 清单，
同一冻结身份，同一 `small` 网络（108 次派发）。v1.1 即「V3.7 的行为 + b390 的命名」。

## 致谢与版权

- **mochizuki0323 / DLSSNR-AMD**（MIT，copyright 2026 mochizuki0323）：Vulkan NR 后端、生产消费调度与通用输入接入的主要参考；上游许可与声明保留在 `LICENSE`、`LICENSES.md`、`third_party/dlssnr_amd_model_tools/LICENSE`。
- 社区项目 **dlssnr on intel**：Intel 平台 DLSS-NR 移植与集成思路参考。
- **B 站 UP 主 灰葉白鸠**：提供技术支持。
- 不随包分发 NVIDIA `nvngx_dlssnr.dll`、由其派生的着色器与 `assets\cold` 权重、OptiScaler 二进制、FSR XMX / XeSS FG 组件与游戏文件；公开发布须遵循各组件自身的分发条件。
