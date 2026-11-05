param(
    [Parameter(Mandatory=$true)][string]$Dll,
    [Parameter(Mandatory=$true)][string]$OptiScalerSource,
    [Parameter(Mandatory=$true)][string]$Glslang,
    [Parameter(Mandatory=$true)][string]$VulkanInclude,
    [string]$Python = "python",
    [string]$CMake = "cmake",
    [ValidateSet("small", "vk102")][string]$Profile = "small",
    [string]$Output = "",
    [string]$PackageOutput = "",
    [string]$FullModel = ""
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repoRoot
try {
    if (!(Test-Path -LiteralPath (Join-Path $VulkanInclude "vulkan/vulkan.h"))) {
        throw "-VulkanInclude must name the include directory containing vulkan/vulkan.h"
    }
    if (!$Output) { $Output = "local/$Profile" }
    if (!$PackageOutput) { $PackageOutput = if ($Profile -eq "small") { "local/game-package" } else { "local/game-package-vk102" } }
    $modelPath = [IO.Path]::GetFullPath((Join-Path $repoRoot $Output))
    $modelArgs = @()
    if ($Profile -eq "small") {
        if (!$FullModel) { throw "Optimized Small requires -FullModel pointing to the optimized vk102 model output (for its shared blend scalar). Build vk102 first." }
        $modelArgs = @("--full-model", [IO.Path]::GetFullPath($FullModel))
    }
    & $Python -B tools/extract_model.py --profile $Profile --dll $Dll --out $modelPath @modelArgs
    if ($LASTEXITCODE) { throw "Model extraction failed" }
    & $Python -B tools/build_assets.py --model $modelPath --glslang $Glslang
    if ($LASTEXITCODE) { throw "Shader/plan preparation failed" }
    & $CMake -S source/vulkan -B build/vulkan -G "Visual Studio 17 2022" -A x64 "-DNR_MODEL_DIR=$modelPath" "-DVULKAN_INCLUDE_DIR=$VulkanInclude"
    if ($LASTEXITCODE) { throw "Vulkan configure failed" }
    & $CMake --build build/vulkan --config Release
    if ($LASTEXITCODE) { throw "Vulkan build failed" }
    $ctestPath = Join-Path (Split-Path -Parent (Get-Command $CMake).Source) "ctest.exe"
    & $ctestPath --test-dir build/vulkan -C Release --output-on-failure
    if ($LASTEXITCODE) { throw "Backend/component CPU tests failed" }
    & $CMake -S source/bridge -B build/bridge -G "Visual Studio 17 2022" -A x64 "-DOPTISCALER_SOURCE=$OptiScalerSource" -DNRB_XESS_IDENTITY_TEST=ON -DNRB_DEFERRED_IDENTITY_TEST=ON -DNRB_LIVE_NR_TEST=ON -DNRB_MULTI_ROUTE_LIVE=ON -DNRB_TAIL_ORDER_PROBE=ON
    if ($LASTEXITCODE) { throw "ASI configure failed" }
    & $CMake --build build/bridge --config Release
    if ($LASTEXITCODE) { throw "ASI build failed" }
    & $ctestPath --test-dir build/bridge -C Release --output-on-failure
    if ($LASTEXITCODE) { throw "Bridge CPU tests failed" }
    & $Python -B tools/test_panel.py --out "build/panel-test-$Profile" --backend $(if ($Profile -eq "small") { "small" } else { "102" })
    if ($LASTEXITCODE) { throw "Panel test failed" }
    & $Python -B tools/assemble_package.py --model $modelPath --out $PackageOutput
    if ($LASTEXITCODE) { throw "Game package assembly failed" }
    Write-Host "Built $PackageOutput. Read INSTALL.md before copying game-files. No game files were changed."
} finally { Pop-Location }
