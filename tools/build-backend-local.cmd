@echo off
rem Local build of the Vulkan backend on this machine.
rem
rem Why this exists next to tools\Build.ps1 — three separate obstacles, in order:
rem
rem   1. The "Visual Studio 17 2022" generator cannot find the compiler when CMake is
rem      launched from a Git Bash session with no vcvars environment, even though
rem      VS Build Tools 2022 (VC 14.44) is installed. -> call vcvars64, use Ninja.
rem
rem   2. reg.exe is on this host's program blacklist (Security Center -> Command
rem      Security). vcvars64.bat asks the registry where the Windows SDK lives; that
rem      query is blocked, so it sets PATH (cl.exe is found) but NOT LIB/INCLUDE for
rem      the SDK. The first symptom is:
rem          LINK : fatal error LNK1104: cannot open file 'kernel32.lib'
rem      which CMake reports as "The C++ compiler is not able to compile a simple
rem      test program". -> append the SDK paths by hand below. Nothing else needs
rem      reg.exe once LIB/INCLUDE are right.
rem
rem   3. CMake still probes the toolchain unless told otherwise -> pass
rem      CMAKE_C/CXX_COMPILER=cl so it skips instance discovery.
rem
rem usage:  tools\build-backend-local.cmd [small|vk102] [clean]
rem         clean  wipes the build directory first
setlocal enabledelayedexpansion
set PROFILE=%~1
if "%PROFILE%"=="" set PROFILE=small
set DO_CLEAN=%~2
rem optional third argument: a target that is EXCLUDE_FROM_ALL, e.g. port_network_model_runner
set EXTRA_TARGET=%~3

set VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat
set MSVC_VER=14.44.35207
set MSVC=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\%MSVC_VER%
set SDK_VER=10.0.26100.0
set SDK=C:\Program Files (x86)\Windows Kits\10
set SDK_BIN=%SDK%\bin\%SDK_VER%\x64
rem CMake writes whatever we pass straight into CMakeRCCompiler.cmake, and a backslash
rem there is an escape character ("Invalid character escape '\P'"), so hand it forward
rem slashes.
set SDK_BIN_FWD=C:/Program Files (x86)/Windows Kits/10/bin/%SDK_VER%/x64
set REPO=%~dp0..
set VK_INC=C:\VulkanSDK\1.4.363.0\Include
set BUILD=%REPO%\build\vulkan-%PROFILE%

call "%VCVARS%" >nul || exit /b 1
cd /d "%REPO%" || exit /b 1

rem --- repair the SDK environment that the blocked registry query left empty -------
set "INCLUDE=%MSVC%\include;%SDK%\Include\%SDK_VER%\ucrt;%SDK%\Include\%SDK_VER%\um;%SDK%\Include\%SDK_VER%\shared;%SDK%\Include\%SDK_VER%\winrt;%SDK%\Include\%SDK_VER%\cppwinrt;%INCLUDE%"
set "LIB=%MSVC%\lib\x64;%SDK%\Lib\%SDK_VER%\ucrt\x64;%SDK%\Lib\%SDK_VER%\um\x64;%LIB%"

if /i "%DO_CLEAN%"=="clean" if exist "%BUILD%" rmdir /s /q "%BUILD%"

echo [1/2] configure %PROFILE%  (MSVC %MSVC_VER%, SDK %SDK_VER%)
cmake -S source\vulkan -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
      -DCMAKE_RC_COMPILER="%SDK_BIN_FWD%/rc.exe" -DCMAKE_MT="%SDK_BIN_FWD%/mt.exe" ^
      -DNR_MODEL_DIR="%REPO%\local\%PROFILE%" -DVULKAN_INCLUDE_DIR="%VK_INC%" || exit /b 1

echo [2/2] build
cmake --build "%BUILD%" || exit /b 1
if not "%EXTRA_TARGET%"=="" (
  echo [extra] %EXTRA_TARGET%
  cmake --build "%BUILD%" --target %EXTRA_TARGET% || exit /b 1
)

echo done: %BUILD%
endlocal
