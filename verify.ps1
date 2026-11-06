<#
.SYNOPSIS
  Verify a dlssnr-b390 installation against the hashes compiled into dlssnr_core.dll.

.DESCRIPTION
  The core refuses to initialise unless the on-disk manifest, runtime plan, shaders and
  weights match the identity baked into it at build time. This script checks the same
  things offline, per category, so a failure names the missing or stale file instead of
  showing up as a silent "NR did nothing" in game.

.EXAMPLE
  .\verify.ps1 -Path 'D:\Games\SomeGame\Binaries\Win64'
#>
param(
    [string]$Path = '',
    [string]$ListPath = ''
)

$ErrorActionPreference = 'Stop'
if (-not $Path) { $Path = $PSScriptRoot }
if (-not $Path) { throw '无法确定目录，请用 -Path 指定' }
$root = (Resolve-Path $Path).Path

# Accept either the release root or an installed game directory.
$nr = Join-Path $root 'dlssnr-b390'
if (-not (Test-Path $nr)) { $nr = Join-Path $root 'game-files\dlssnr-b390' }
if (-not (Test-Path $nr)) { throw "找不到 dlssnr-b390 目录（试过 $root 和 $root\game-files）" }
if (-not $ListPath) {
    $cand = Join-Path $root 'frozen-files.sha256'
    if (Test-Path $cand) { $ListPath = $cand }
    else {
        $cand2 = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'frozen-files.sha256'
        if (Test-Path $cand2) { $ListPath = $cand2 } else { throw '缺少 frozen-files.sha256（用 -ListPath 指定）' }
    }
}
$identity = Join-Path $root 'frozen-identity.txt'
if (-not (Test-Path $identity)) {
    $identity = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'frozen-identity.txt'
}

"  资源目录: $nr"
''

$stats = @{}
$fail = @()
foreach ($line in (Get-Content $ListPath -Encoding UTF8)) {
    if ($line -notmatch '^([0-9a-f]{64})\s+(.+)$') { continue }
    $want = $Matches[1]; $rel = $Matches[2]
    $cat = if ($rel -like 'assets/cold/*') { 'cold' } elseif ($rel -like 'assets/spv/*') { 'spv' } elseif ($rel -like 'ladder/*') { 'ladder' } else { 'manifest' }
    if (-not $stats.ContainsKey($cat)) { $stats[$cat] = @{ ok = 0; bad = 0; miss = 0 } }
    $file = Join-Path $nr ($rel -replace '/', '\')
    if (-not (Test-Path $file)) {
        # 权重与 NVIDIA 运行时派生着色器都不随仓库/发布包分发：缺失只计数，不算失败
        if ($cat -eq 'cold' -or $cat -eq 'spv') { $stats[$cat].miss++ }
        else { $stats[$cat].bad++; $fail += "缺少: $rel" }
        continue
    }
    if ((Get-FileHash $file -Algorithm SHA256).Hash.ToLower() -eq $want) { $stats[$cat].ok++ }
    else { $stats[$cat].bad++; $fail += "哈希不符: $rel" }
}

Write-Host '=== 分类校验 ===' -ForegroundColor Cyan
foreach ($k in @('manifest', 'spv', 'ladder', 'cold')) {
    if ($stats.ContainsKey($k)) {
        $note = if ($stats[$k].miss -gt 0) { "  （缺失 $($stats[$k].miss)，未随包分发）" } else { '' }
        '  {0,-9} 通过 {1,3} / 失败 {2,3}{3}' -f $k, $stats[$k].ok, $stats[$k].bad, $note
    }
}
''
Write-Host '=== manifest / runtime 身份 ===' -ForegroundColor Cyan
if (Test-Path $identity) {
    foreach ($line in (Get-Content $identity -Encoding UTF8)) {
        if ($line -match '^(\S+)\s+([0-9a-f]{64})$') {
            $name = $Matches[1]; $want = $Matches[2]
            $file = Join-Path $nr ('manifest\' + $name)
            if (Test-Path $file) {
                $got = (Get-FileHash $file -Algorithm SHA256).Hash.ToLower()
                '  {0,-28} {1}' -f $name, $(if ($got -eq $want) { '✔' } else { "✗ $got" })
                if ($got -ne $want) { $fail += "manifest 身份不符: $name" }
            } else { '  {0,-28} ✗ 缺失' -f $name; $fail += "缺少 manifest\$name" }
        }
    }
} else { '  跳过（缺 frozen-identity.txt）' }

''
Write-Host '=== core 内置身份 ===' -ForegroundColor Cyan
$core = Join-Path $root 'dlssnr_core.dll'
if (-not (Test-Path $core)) { $core = Join-Path $root 'game-files\dlssnr_core.dll' }
if (Test-Path $core) {
    $bytes = [IO.File]::ReadAllBytes($core)
    $ascii = [Text.Encoding]::ASCII.GetString($bytes)
    $hits = 0
    if (Test-Path $identity) {
        foreach ($line in (Get-Content $identity -Encoding UTF8)) {
            if ($line -match '^(\S+)\s+([0-9a-f]{64})$') {
                $want = $Matches[2]
                $found = $ascii.Contains($want)
                '  {0,-28} 内置 {1}' -f $Matches[1], $(if ($found) { '✔' } else { '✗' })
                if ($found) { $hits++ } else { $fail += "core 未内置 $($Matches[1]) 的身份" }
            }
        }
    }
    '  dlssnr_core.dll {0:N0} B' -f (Get-Item $core).Length
} else {
    '  ✗ 缺少 dlssnr_core.dll（本仓库不带二进制；校验游戏安装目录时请确认文件已复制）'
}

''
if ($stats.ContainsKey('cold') -and $stats['cold'].ok -eq 0) {
    Write-Warning "assets\cold 为空：公开渠道不含权重，需你自备 70 个 .bin 放到 $nr\assets\cold（须与 frozen-files.sha256 的 cold 条目逐条一致）。"
}
if ($stats.ContainsKey('spv') -and $stats['spv'].ok -eq 0) {
    Write-Warning "assets\spv 为空：公开渠道不含 NVIDIA 运行时派生着色器，需用你自己那份 nvngx_dlssnr.dll 配合 tools\build_assets.py 生成（须与 frozen-files.sha256 的 spv 条目逐条一致）。"
}

if ($fail.Count -gt 0) {
    Write-Host "✗ 校验未通过，问题 $($fail.Count) 处：" -ForegroundColor Red
    $fail | Select-Object -First 12 | ForEach-Object { '    ' + $_ }
    exit 1
}
$missing = 0
foreach ($k in $stats.Keys) { $missing += $stats[$k].miss }
if ($missing -gt 0) {
    Write-Host "✔ 已存在文件与 core 内置身份全部一致；另有 $missing 项属未随包分发资产（cold/spv），补齐后再跑一次即可全绿。" -ForegroundColor Green
}
else {
    Write-Host '✔ 全部文件与 core 内置身份一致。' -ForegroundColor Green
}
