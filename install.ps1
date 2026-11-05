#Requires -Version 5.1
<#
    DLSS-NR (B390) v1.1 安装脚本
    把 game-files 里的文件安装到巫师3 DX12 目录，并清掉旧命名残留。

    用法：
      powershell -ExecutionPolicy Bypass -File .\install.ps1
      powershell -ExecutionPolicy Bypass -File .\install.ps1 -GameDir "D:\SteamLibrary\steamapps\common\The Witcher 3\bin\x64_dx12"
      powershell -ExecutionPolicy Bypass -File .\install.ps1 -GameDir "<...>\x64_dx12" -DryRun
#>
[CmdletBinding()]
param(
    [string]$GameDir,
    [switch]$DryRun,
    [switch]$SkipVerify
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$gf   = Join-Path $root 'game-files'

function Write-Step([string]$text) { Write-Host ''; Write-Host "== $text" -ForegroundColor Cyan }
function Write-Ok([string]$text)   { Write-Host "   [ok] $text" -ForegroundColor Green }
function Write-Warn2([string]$text){ Write-Host "   [warn] $text" -ForegroundColor Yellow }

# ---------------------------------------------------------------- 1. 定位游戏目录
Write-Step '1/6 定位游戏目录'

$candidates = New-Object System.Collections.Generic.List[string]
if ($GameDir) { $candidates.Add($GameDir) }
$candidates.Add((Join-Path $root 'game-files-target'))
foreach ($drive in (Get-PSDrive -PSProvider FileSystem | Select-Object -ExpandProperty Root)) {
    $candidates.Add((Join-Path $drive 'SteamLibrary\steamapps\common\The Witcher 3\bin\x64_dx12'))
}
foreach ($pf in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
    if ($pf) { $candidates.Add((Join-Path $pf 'Steam\steamapps\common\The Witcher 3\bin\x64_dx12')) }
}

$target = $null
foreach ($c in $candidates) {
    if (-not $c) { continue }
    if (Test-Path (Join-Path $c 'witcher3.exe')) { $target = (Resolve-Path $c).Path; break }
}
if ($GameDir -and (Test-Path (Join-Path $GameDir 'witcher3.exe'))) { $target = (Resolve-Path $GameDir).Path }

if (-not $target) {
    $target = $GameDir
    if (-not $target) {
        Write-Host '   没有自动找到 witcher3.exe，请用 -GameDir 指定 bin\x64_dx12 目录。' -ForegroundColor Yellow
        exit 1
    }
    Write-Warn2 "未在该目录看到 witcher3.exe，仍按指定路径安装：$target"
}
Write-Ok "目标目录：$target"

# ---------------------------------------------------------------- 2. 检查包内容
Write-Step '2/6 校验包内容'
foreach ($p in @('dlssnr_core.dll', 'dlssnr-b390.ini', 'dlssnr-b390')) {
    $full = Join-Path $gf $p
    if (-not (Test-Path $full)) { throw "包内缺少 game-files\$p" }
}
$coreSize = (Get-Item (Join-Path $gf 'dlssnr_core.dll')).Length
Write-Ok "内核 dlssnr_core.dll $coreSize 字节"
$coldDir = Join-Path $gf 'dlssnr-b390\assets\cold'
$coldCount = if (Test-Path $coldDir) { (Get-ChildItem $coldDir -File).Count } else { 0 }
$spvCount  = (Get-ChildItem (Join-Path $gf 'dlssnr-b390\assets\spv') -File).Count
$maniCount = (Get-ChildItem (Join-Path $gf 'dlssnr-b390\manifest') -File).Count
if ($coldCount -gt 0) {
    Write-Ok "权重 $coldCount 个（随包） · 着色器 $spvCount 个 · 清单 $maniCount 个"
} else {
    Write-Ok "着色器 $spvCount 个 · 清单 $maniCount 个"
    Write-Warn2 '本包不含权重（assets\cold 为空），需你按 README 自行准备，否则内核会 init failed。'
}
if ($spvCount -lt 40) { throw '着色器数量不足，包可能不完整。' }

# ---------------------------------------------------------------- 3. 复制文件
Write-Step '3/6 安装文件'

$plan = @(
    @{ From = (Join-Path $gf 'dlssnr_core.dll');  To = (Join-Path $target 'dlssnr_core.dll') },
    @{ From = (Join-Path $gf 'dlssnr-b390.ini');  To = (Join-Path $target 'dlssnr-b390.ini') }
)
foreach ($item in $plan) {
    if ($DryRun) { Write-Host ("   would copy {0}" -f (Split-Path $item.From -Leaf)) }
    else { Copy-Item $item.From $item.To -Force; Write-Ok (Split-Path $item.To -Leaf) }
}

$dstData = Join-Path $target 'dlssnr-b390'
if ($DryRun) {
    Write-Host '   would copy game-files\dlssnr-b390 -> dlssnr-b390 (keep existing assets\cold)'
} else {
    New-Item -ItemType Directory -Path $dstData -Force | Out-Null
    # 用 /E 而不是 /MIR：目标目录里可能有用户自备的 assets\cold 权重，不能被清掉
    $rc = robocopy (Join-Path $gf 'dlssnr-b390') $dstData /E /NFL /NDL /NJH /NJS /NP
    if ($LASTEXITCODE -ge 8) { throw "robocopy 失败，退出码 $LASTEXITCODE" }
    Write-Ok "dlssnr-b390\ 已同步（$spvCount 着色器 / $maniCount 清单）"
}

# ---------------------------------------------------------------- 4. 清理旧命名残留
Write-Step '4/6 清理旧命名残留'

$legacy = @('dlssnr-amd', 'intel-b390-dlssnr', 'nvngx_dlls')
foreach ($name in $legacy) {
    $p = Join-Path $target $name
    if (Test-Path $p) {
        if ($DryRun) { Write-Host "   would remove $name" }
        else {
            $item = Get-Item $p -Force
            if ($item.LinkType) { [System.IO.Directory]::Delete($p) }
            else { Remove-Item $p -Recurse -Force }
            Write-Ok "已删除旧目录 $name"
        }
    }
}
foreach ($name in @('dlssnr-amd.ini', 'intel-b390-dlssnr.ini', 'nvngx.dll_dlssnr.dll')) {
    $p = Join-Path $target $name
    if (Test-Path $p) {
        if ($DryRun) { Write-Host "   would remove $name" }
        else { Remove-Item $p -Force; Write-Ok "已删除旧文件 $name" }
    }
}
# 旧的错误转发壳（5.3MB 的 nvngx_dlssnr.dll）会顶掉官方运行时，必须清掉
$fake = Join-Path $target 'nvngx_dlssnr.dll'
if (Test-Path $fake) {
    $sz = (Get-Item $fake).Length
    if ($sz -lt 100MB) {
        if ($DryRun) { Write-Host "   would remove bogus nvngx_dlssnr.dll ($sz bytes)" }
        else { Remove-Item $fake -Force; Write-Warn2 "删除错误的 nvngx_dlssnr.dll（$sz 字节，应为 276162560 字节）" }
    }
}

# ---------------------------------------------------------------- 5. 校验安装结果
Write-Step '5/6 校验安装结果'
if ($SkipVerify) {
    Write-Warn2 '按 -SkipVerify 跳过'
} elseif ($DryRun) {
    Write-Host '   dry-run，跳过'
} else {
    $verify = Join-Path $root 'verify.ps1'
    if (Test-Path $verify) {
        # robocopy 成功时退出码为 1，会残留到 $LASTEXITCODE；调用脚本前必须清零，
        # 否则 verify 明明通过也会走到下面的异常分支。
        $LASTEXITCODE = 0
        & $verify -Path $target -ListPath (Join-Path $root 'frozen-files.sha256')
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { Write-Warn2 'verify.ps1 报告异常，请查看上面的输出' }
    } else {
        Write-Warn2 '包内没有 verify.ps1'
    }
}

# ---------------------------------------------------------------- 6. 依赖提示
Write-Step '6/6 还需你手动确认的两件事'

$opti = Join-Path $target 'OptiScaler.ini'
if (Test-Path $opti) {
    $txt = Get-Content $opti -Raw
    if ($txt -match '(?im)^\s*NvngxPath\s*=\s*\.\\dlssnr_core\.dll\s*$') {
        Write-Ok 'OptiScaler.ini 已指向 .\dlssnr_core.dll'
    } else {
        Write-Warn2 'OptiScaler.ini 里的 NvngxPath 需要设为 .\dlssnr_core.dll'
    }
} else {
    Write-Warn2 "没找到 OptiScaler.ini：请先装好 OptiScaler，再把 NvngxPath 设为 .\dlssnr_core.dll"
}

$official = Join-Path $target 'nvngx_dlssnr.dll'
if (Test-Path $official) {
    $sz = (Get-Item $official).Length
    if ($sz -eq 276162560) { Write-Ok '官方 nvngx_dlssnr.dll 在位（310.9.16.0，276162560 字节）' }
    else { Write-Warn2 "nvngx_dlssnr.dll 大小异常（$sz），应为 276162560 字节" }
} else {
    Write-Warn2 '目录里没有官方 nvngx_dlssnr.dll（310.9.16.0）。本包不附带它，请自行从游戏原版目录取回备份。'
}

Write-Host ''
Write-Host '安装完成。进游戏后检查：' -ForegroundColor Cyan
Write-Host "  · $target\dlssnr-b390.log 出现 'port: plan read, 222 buffers and 108 passes'（不是 'init failed'）"
Write-Host "  · $dstData\nr_tune_read.txt 已生成且有内容（参数通道已打通）"
