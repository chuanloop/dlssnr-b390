cmake_minimum_required(VERSION 3.20)
project(dlssnr_provider LANGUAGES CXX)

# L3': the module OptiScaler loads and calls
# through the private dlssnr_* ABI. Deliberately standalone - it must not pull in
# the Vulkan backend until the ABI is confirmed.

set(DLSSNR_NGX_SDK_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../_deps/optiscaler/external/nvngx_dlss_sdk"
    CACHE PATH "NGX headers; only NVSDK_NGX_Parameter's vtable layout is used")

if(NOT EXISTS "${DLSSNR_NGX_SDK_DIR}/nvsdk_ngx_params.h")
  message(FATAL_ERROR
    "nvsdk_ngx_params.h not found in ${DLSSNR_NGX_SDK_DIR}. Point "
    "DLSSNR_NGX_SDK_DIR at the OptiScaler checkout's external/nvngx_dlss_sdk.")
endif()

add_library(dlssnr_provider SHARED dlssnr_probe.cpp)
target_include_directories(dlssnr_provider PRIVATE "${DLSSNR_NGX_SDK_DIR}")
target_compile_definitions(dlssnr_provider PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)

if(MSVC)
  # /EHa: the parameter probe relies on SEH around virtual calls into a foreign
  # object; only SEH with unwinding is safe there.
  target_compile_options(dlssnr_provider PRIVATE /EHa /utf-8 /W4)
endif()

# Deployed under this exact name: OptiScaler looks for it beside itself, and the
# model container only accepts callers whose module path contains "nvngx.dll".
set_target_properties(dlssnr_provider PROPERTIES
  OUTPUT_NAME "nvngx.dll_dlssnr"
  PREFIX "")

# Offline harness: implements NVSDK_NGX_Parameter the way the caller does and
# drives the D3D12 entries, so the parameter-reading path is verified before a
# game launch is spent on it.
add_executable(dlssnr_probe_host tests/host_test.cpp)
target_include_directories(dlssnr_probe_host PRIVATE "${DLSSNR_NGX_SDK_DIR}")
target_link_libraries(dlssnr_probe_host PRIVATE dlssnr_provider)
if(MSVC)
  target_compile_options(dlssnr_probe_host PRIVATE /utf-8)
endif()

# NGX core stand-in. OptiScaler's NR path refuses to start without an NGX core
# (see ngx_stub.cpp); deploying this as _nvngx.dll satisfies that check without
# touching the game's own upscaler path.
add_library(ngx_stub SHARED ngx_stub.cpp)
target_include_directories(ngx_stub PRIVATE "${DLSSNR_NGX_SDK_DIR}")
target_compile_definitions(ngx_stub PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
set_target_properties(ngx_stub PROPERTIES OUTPUT_NAME "_nvngx" PREFIX "")
if(MSVC)
  target_compile_options(ngx_stub PRIVATE /EHa /utf-8 /W4)
endif()
