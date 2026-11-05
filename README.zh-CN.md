# Intel DLSS-NR for 巫师3 —— 新用户安装指南（v1.1 / `dlssnr-b390`）

> **下载**：本仓库只放源码与文档；可直接运行的包（内核 + 42 支 spv + 安装脚本）在
> [Releases → v1.1](https://github.com/chuanloop/dlssnr-b390/releases/tag/v1.1)。

你只需要：巫师3、一块 Intel Arc B390（Xe3 核显）及已安装的显卡驱动，以及**自备**的三样东西——
官方 DLSS-NR 运行时 `nvngx_dlssnr.dll`、一个 `OptiScaler`、以及你自己的权重文件（`assets\cold`）。
本包带 NR 内核、42 个着色器、派发计划与安装/校验脚本，**不含任何权重或从 NVIDIA 运行时抽出的模型数据**，
也不含 `nvngx_dlssnr.dll` 本体。本包**不需要** Python、Visual Studio、Vulkan SDK，不需要你自己编译或抽取。

[English](README.md) · [更新说明](RELEASE-NOTES.md) · [包内容清单](ASSETS.md) · [源码与构建](INSTALL.md)

## 一、准备

- Windows 10/11 64 位。当前性能与稳定性测试平台为 **Intel Arc B390（Xe3 核显）**，驱动需为 Intel 或整机厂商提供的正式版本。
- 安装脚本以**巫师3 次世代版**为默认目标：游戏目录为 `...\The Witcher 3\bin\x64_dx12\witcher3.exe`，**必须以 DX12 启动**。已测巫师3、黑神话：悟空等 DX12 游戏，其他游戏需自行测试。
- 先退出游戏。把整个压缩包解压到一个普通目录，**不要在压缩包里直接运行脚本**，也不要只挑 `install.ps1` 出来单独放。
- **你必须自备，本包不附带：**

  | 项目 | 要求 | 放在哪 |
  |---|---|---|
  | `nvngx_dlssnr.dll` | 官方 310.9.16.0，必须 **276,162,560 字节**，SHA-256 `CDFA6C1E…ABF3ED` | `bin\x64_dx12\` |
  | `OptiScaler` | 较新的上游构建即可（本包不附二进制） | `bin\x64_dx12\` |
  | 权重 `assets\cold` | **70 个** `.bin`，须与 `frozen-files.sha256` 里的 cold 条目逐条一致 | `<x64_dx12>\dlssnr-b390\assets\cold\` |

  `nvngx_dlssnr.dll` 是 NVIDIA 版权运行时，权重也是由它离线抽取得到的，两者都不能随包分发，请从游戏原版目录或你自己的备份取回。
  **如果目录里有一个约 5.3 MB 的同名 `nvngx_dlssnr.dll`，那是有害的旧转发壳，不是官方文件**，`install.ps1` 会删掉它。

- **关于权重**（本包刻意不带，规避版权）：内核启动时会逐条校验编译进去的冻结清单，权重缺失或哈希不符就直接 `init failed`，画面毫无变化。
  - 合法来源是**你自己已有的那份**：旧版本目录（如 v3.7 的 `dlssnr-amd\assets\cold`）、你之前安装到游戏的 `dlssnr-b390\assets\cold`，或你自己的备份。
  - 复制过来后先跑 `verify.ps1`，`cold 70 / 失败 0` 才算对。
  - **重新抽取现在能对齐冻结清单**：跑完整链路（`extract_model.py` + `pack_b580_weights.py`）可逐字节复现全部 70 条 cold 资源（必须带 `--full-model`，或 `--blend-scalar` 指向那份 4 字节的 `ioport_blend_scale`）。早期版本有 8 条不同（`swin_7/22/23` 的 `finaltile`、`swin_8`、`split16_b9/b10/b20/b21`），现在打包阶段直接产出内核消费布局。细节见 `ASSETS.md`。
- 若之前装过 `dlssnr-amd\`、`intel-b390-dlssnr\`、`dlssnr-amd.ini` 等旧命名残留，`install.ps1` 会自动清理，不需要手工删。
- 不要把 `amd_fidelityfx_loader_dx12.dll`、`amd_fidelityfx_framegeneration_dx12.dll`、`amd_fidelityfx_upscaler_dx12.dll` 当成 OPT 专属文件删除——巫师3 与 FSR 方案本身也使用它们。

## 二、安装

1. 在解压出来的目录里，**右键 `install.ps1` → 使用 PowerShell 运行**；或在 PowerShell 里执行：

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\install.ps1
   ```

2. 脚本会自动扫描 Steam 库定位 `bin\x64_dx12`，并打印目标目录让你确认。位置不对就显式指定：

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\install.ps1 -GameDir "D:\SteamLibrary\steamapps\common\The Witcher 3\bin\x64_dx12"
   ```

   加 `-DryRun` 只预览不动文件；加 `-SkipVerify` 跳过安装后的哈希校验。

3. 看到六步走完、末尾提示 `安装完成` 即成功：定位目录 → 校验包内文件（内核 / 42 着色器 / 3 清单；`assets\cold` 为空会明确提示）→ 复制 → 清旧名残留 → 调 `verify.ps1` 逐文件比对哈希 → 报告 `OptiScaler.ini` 与官方 DLL 状态。
4. 游戏目录无写权限时，用**管理员** PowerShell 运行同一条命令。脚本不改显卡驱动、不改注册表、不改游戏存档、不改 TDR。
5. 手动安装（不想用脚本时，复制三样即可）：

   | 从 | 到 |
   |---|---|
   | `game-files\dlssnr_core.dll` | `<x64_dx12>\dlssnr_core.dll` |
   | `game-files\dlssnr-b390.ini` | `<x64_dx12>\dlssnr-b390.ini` |
   | `game-files\dlssnr-b390\` | `<x64_dx12>\dlssnr-b390\`（含着色器与清单；权重另行放入 `assets\cold`） |

6. 让 OptiScaler 加载本内核。在 `<x64_dx12>\OptiScaler.ini` 中设置：

   ```ini
   NvngxPath=.\dlssnr_core.dll
   ```

## 三、第一次进入游戏

- 从 Steam 原入口启动，选 **DX12**。
- 在游戏画面设置中启用 **DLSS / 超分**。NR 的处理分辨率跟随游戏送入的渲染分辨率（"原生"指渲染分辨率，不是显示器分辨率）。
- 按 **Insert** 打开 OptiScaler 面板，确认 NR 已启用、路由目标是 `dlssnr_core.dll`。
- 玩一分钟，用两个文件确认真的在跑：

  ```
  <x64_dx12>\dlssnr-b390.log            → port: plan read, 222 buffers and 108 passes
                                          随后 network built at <W>x<H>
  <x64_dx12>\dlssnr-b390\nr_tune_read.txt → 已生成，内容形如 ini=ok tune=ok detail=1.000000 …
  ```

- **本包只有 Small 网络**（108 次派发 / 70 个 cold 资源 / 222 个缓冲），没有 Full 232 派发档位，面板里找不到网络切换是正常的。
- 调参直接改 `dlssnr-b390\nr_tune.txt`，保存即生效（游戏内也生效，不用重启）；`nr_tune_read.txt` 是内核回读出来的当前生效值，用它核对。
- **Ctrl+F11 是 int4 混合路径开关，不是 NR 总开关**：关掉它画面会变，但不代表 NR 被关闭。
- 本包没有网页控制面板，也不带自定义滤镜编辑器，参数在 ini 与 `nr_tune.txt` 里改。

## 四、卸载／更新

- 退出游戏，删除这四项即卸载干净（本包不注入系统目录）：

  ```
  <x64_dx12>\dlssnr_core.dll
  <x64_dx12>\dlssnr-b390.ini
  <x64_dx12>\dlssnr-b390.log
  <x64_dx12>\dlssnr-b390\
  ```

  并把 `OptiScaler.ini` 里的 `NvngxPath` 还原成官方值。
- 本包**不写安装记录、不做自动备份**（`install.ps1` 只覆盖同名文件）。升级前请自行备份 `dlssnr_core.dll` 与原 `dlssnr-b390\` 目录。
- 更新请先按上面卸掉旧版再装新版，不要新旧文件叠放在一起。
- `nvngx_dlssnr.dll` 与 OptiScaler 属于你自己的组件，卸载本包时**不要删**，否则官方 DLSS 也会一起失效。

## 五、遇到问题

先看日志再动手，别叠装多个版本碰运气：

| 现象 | 说明 |
|---|---|
| `dlssnr-b390.log` 里没有任何内容 | 内核没被加载：目录放错、`NvngxPath` 没指向 `dlssnr_core.dll`，或有 5.3 MB 的假 `nvngx_dlssnr.dll` 占着名字 |
| `port: init failed … cannot read …__small-<hash>-<hash>.spv` | 冻结清单里有过期条目，属 v1.0 的缺陷；换 v1.1 的内核 |
| `port: init failed … requires exact build-frozen manifest/runtime` | 资源被重新生成或替换过（例如用新工具重抽权重），恢复随包的 `dlssnr-b390\manifest` 与 `assets\cold` |
| 有 `plan read` 但画面没变 | OptiScaler 没把 NR 路由过来：检查 `NvngxPath`、游戏内 DLSS 档位、是否被别的代理 DLL 抢占 |
| 卡顿或 NR 暂停 | 保存日志后退出；不要靠调大 TDR、改频率或反复切换后端掩盖 |

离线复核安装完整性：

```powershell
powershell -ExecutionPolicy Bypass -File .\verify.ps1 -Path "<x64_dx12>"
```

期望 `cold 70/0`、`spv 40/0`、两个身份 `✔`、结尾 `✔ 全部文件与 core 内置身份一致`。

## 六、版本内容

- 内核 `dlssnr_core.dll` 5,431,431 字节，由改名后的 V3.7 源码树编译；内置身份 manifest `bf6c087a…` / runtime `c168a078…`。
- 完整改名：目录 `dlssnr-b390\`、配置 `dlssnr-b390.ini`、日志 `dlssnr-b390.log`，二进制内 `dlssnr-b390` ×29、旧名 ×0。
- 修复了 v1.0 无法初始化的根因：编译进内核的冻结资源清单里的 2 条过期着色器条目已删除，清单从 112 条回到 110 条，与 V3.7 行为一致。详见 [RELEASE-NOTES.md](RELEASE-NOTES.md)。
- 数据与画质基准为 V3.7：同一份 42 着色器 / 3 清单 / 同一冻结身份 / 同一 Small 网络（108 派发）。**70 个权重不在包内分发**，需使用与 V3.7 同批的文件，哈希见 `frozen-files.sha256`。
- v1.1 是首个正式发布版；本次只改命名、修初始化与打包，不改模型、着色器与推理代码。
- "面向新用户"是指不需要开发环境，不代表无条件兼容所有 Intel GPU 与所有游戏。

## 致谢与版权

- **mochizuki0323 / DLSSNR-AMD**（MIT，copyright 2026 mochizuki0323）：本包 Vulkan NR 后端、生产消费调度与通用输入接入大量参考该项目及其 OptiScaler 集成，上游许可与声明保留在 `LICENSE`、`LICENSES.md` 与 `third_party/dlssnr_amd_model_tools/LICENSE`。
- 社区项目 **dlssnr on intel**：Intel 平台上 DLSS-NR 的移植与集成思路参考，特此致谢。
- **B 站 UP 主 灰葉白鸠**：提供技术支持，特此致谢。
- 不随包分发：NVIDIA `nvngx_dlssnr.dll`、**由它离线抽取出的权重与模型数据（`assets\cold`）**、OptiScaler 二进制、FSR XMX / XeSS FG 组件、巫师3 游戏文件。公开发布需分别遵守各组件自身的分发条件。
- VC++ 运行库官方来源（仅在系统缺少时按需安装）：https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
