<#
.SYNOPSIS
  Release guard: fail when a folder or zip contains NVIDIA runtime artifacts or files derived from them.

.DESCRIPTION
  This project publishes sources only. Nothing that comes from NVIDIA's DLSS-NR runtime may be
  uploaded: the runtime itself (`nvngx*.dll`), the `__small-*.spv` shaders derived from it, the
  cold weight files, or third-party upscaler binaries that would silently replace game files.

  Run it on the exact tree you are about to commit, zip or attach to a release:

    .\tools\check_no_nv.ps1 -Path .
    .\tools\check_no_nv.ps1 -Path ..\NR-INTEL-dlssnr-b390-v1.3-20261106.zip

  Exit code 0 = clean, 1 = at least one forbidden file found. `.gitignore` already keeps these
  paths out of git, so a hit here usually means a manual `git add -f`, a packaging script, or a
  release attachment.

.EXAMPLE
  .\tools\check_no_nv.ps1 -Path . -Quiet
#>
param(
    [string]$Path = '.',
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'

# name shown in the report, regex matched against the path inside the tree/archive
$rules = @(
    @{ Name = 'NVIDIA 运行时 nvngx*.dll'; Rule = '(^|[/\\])nvngx[^/\\]*\.dll$' }
    @{ Name = '由运行时派生的着色器 __small-*'; Rule = '__small-[0-9a-f]{6,}' }
    # 本仓库自有的 port_network_bridge_*.spv 允许存在（由 source/ 编译），只有 __small-* 属派生内容
    @{ Name = '权重资产 assets/cold/*.bin'; Rule = '[/\\]assets[/\\]cold[/\\][^/\\]+\.bin$' }
    @{ Name = '任意权重 .bin'; Rule = '\.bin$' }
    @{ Name = '上采样器二进制'; Rule = '(^|[/\\])(OptiScaler|dxgi|winmm|version|dbghelp|d3d12)\.dll$' }
    @{ Name = '模型容器'; Rule = '\.(safetensors|npz|npy|onnx|pt|cubin)$' }
)

function Test-Entry {
    param([string]$Name, [long]$Length)

    $hits = @()
    foreach ($r in $rules) {
        if ($Name -match $r.Rule) { $hits += $r.Name }
    }
    return [pscustomobject]@{ Path = $Name; Size = $Length; Hits = ($hits -join ' / ') }
}

if (-not (Test-Path -LiteralPath $Path)) { throw "路径不存在: $Path" }
$full = (Resolve-Path -LiteralPath $Path).Path

$entries = @()
if ((Get-Item -LiteralPath $full).PSIsContainer) {
    $entries = Get-ChildItem -LiteralPath $full -Recurse -File -Force |
        ForEach-Object { Test-Entry -Name $_.FullName.Substring($full.Length).TrimStart('\', '/') -Length $_.Length }
}
elseif ($full -match '\.(zip|7z)$') {
    if ($full -notmatch '\.zip$') { throw '只支持 .zip 归档；.7z 请先解压再扫' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($full)
    try {
        $entries = $archive.Entries |
            Where-Object { $_.Name -ne '' } |
            ForEach-Object { Test-Entry -Name $_.FullName -Length $_.Length }
    }
    finally { $archive.Dispose() }
}
else { throw "既不是目录也不是 zip: $full" }

$bad = @($entries | Where-Object { $_.Hits })
$total = @($entries).Count

if (-not $Quiet) {
    Write-Host ''
    Write-Host "=== 扫描: $full（$total 个文件）===" -ForegroundColor Cyan
}

if ($bad.Count -eq 0) {
    if (-not $Quiet) {
        Write-Host '✔ 未发现 NVIDIA 运行时、派生着色器、权重或上采样器二进制。' -ForegroundColor Green
    }
    exit 0
}

Write-Host "✗ 发现 $($bad.Count) 个不应分发的文件：" -ForegroundColor Red
$bad | Group-Object Hits | ForEach-Object {
    Write-Host ("  [{0}] {1} 个" -f $_.Name, $_.Count) -ForegroundColor Yellow
    $_.Group | Select-Object -First 5 | ForEach-Object { Write-Host ('      ' + $_.Path) }
    if ($_.Count -gt 5) { Write-Host ('      … 另有 {0} 个' -f ($_.Count - 5)) }
}
Write-Host ''
Write-Host '这些文件只能留在本地：`.gitignore` 排除它们，发布包和 Release 附件也都不带。' -ForegroundColor Yellow
exit 1
