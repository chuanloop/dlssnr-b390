@echo off
REM Incremental rebuild of the small-profile backend in an MSVC environment.
REM frame.inc / frozen.inc / ladder.inc are compile-time includes, so a plan change needs this,
REM not a full tools\Build.ps1 (which re-extracts the model and would overwrite the pinned plan).
setlocal
cd /d "%~dp0.."
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
  echo vcvars64 failed
  exit /b 1
)
cmake --build build\vulkan-small %*
exit /b %errorlevel%
