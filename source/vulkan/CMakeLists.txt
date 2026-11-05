cmake_minimum_required(VERSION 3.24)
project(nrvk_model LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(NR_MODEL_DIR "" CACHE PATH "Output directory of extract_model.py and build_assets.py")
if(NOT EXISTS "${NR_MODEL_DIR}/generated/port_network_bridge_backend_frozen.inc")
  message(FATAL_ERROR "Run tools/build_assets.py first and set NR_MODEL_DIR")
endif()
# The frame extent is a property of the plan, not of this source tree. tools/freeze_plan.py
# writes it from whichever port_network_runtime.txt is on disk, so the same backend can be
# built for 1280x720, 960x540 or anything else the plan supports.
if(NOT EXISTS "${NR_MODEL_DIR}/generated/port_network_bridge_backend_frame.inc")
  message(FATAL_ERROR "Run tools/freeze_plan.py first and set NR_MODEL_DIR")
endif()
find_path(VULKAN_INCLUDE_DIR vulkan/vulkan.h HINTS "$ENV{VULKAN_SDK}/Include" REQUIRED)
set(BRIDGE_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../bridge/include")
add_compile_options(/utf-8 /O2)
include_directories("${VULKAN_INCLUDE_DIR}" "${BRIDGE_INCLUDE}" "${NR_MODEL_DIR}/generated")
# `port_network_bridge_backend.cpp` logs the port's build phases through `nr::logf`, which lives in
# the host runtime's core (`10-SRC/upstream-host/windows/src/core/nr_log.hpp`). The game build gets
# that on its include path from `windows/build/build_optiscaler_nr.sh`; this CMake target did not, so
# the file was unbuildable here after the logging was added. The header is self-contained (std only).
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../../upstream-host/windows/src/core")
add_library(port_backend STATIC port_network_bridge_backend.cpp)
target_link_libraries(port_backend PUBLIC bcrypt)
add_library(port_network_bridge_backend SHARED port_network_bridge_backend.cpp)
target_link_libraries(port_network_bridge_backend PRIVATE bcrypt)
target_link_options(port_network_bridge_backend PRIVATE
  /EXPORT:nrvk_backend_probe /EXPORT:nrvk_backend_init /EXPORT:nrvk_backend_frame
  /EXPORT:nrvk_backend_release /EXPORT:port_network_bridge_backend_submit_result
  /EXPORT:port_network_bridge_backend_last_error /EXPORT:nrvk_backend_set_timestamp_query)
add_library(nrvk_bridge SHARED nrb_vulkan_processor.cpp)
target_link_libraries(nrvk_bridge PRIVATE port_backend bcrypt d3d12 dxgi)
target_link_options(nrvk_bridge PRIVATE /EXPORT:NRVK_AttachToBridge /EXPORT:NRVK_Shutdown /EXPORT:NRVK_InteropSelfTestD3D12 /EXPORT:NRVK_DiagSetState /EXPORT:NRVK_DiagGetLastCpuSubstage /EXPORT:NRVK_DiagHarvestRetired /EXPORT:NRVK_DiagGetRecords /EXPORT:NRVK_DiagGetDeviceInfo)
add_executable(port_queue_backend_cpu port_queue_backend_cpu.cpp)
target_link_libraries(port_queue_backend_cpu PRIVATE bcrypt)
add_executable(nrvk_bridge_cpu_abi nrvk_bridge_cpu_abi.cpp)
enable_testing()
add_test(NAME backend_record COMMAND port_queue_backend_cpu "${NR_MODEL_DIR}/manifest" "$<TARGET_FILE:port_network_bridge_backend>")
add_test(NAME component_abi COMMAND nrvk_bridge_cpu_abi "$<TARGET_FILE:nrvk_bridge>" "$<TARGET_FILE:port_network_bridge_backend>" "${NR_MODEL_DIR}/manifest")

# Offline experimental runner: opt-in target, never executed by default builds.
add_executable(port_network_model_runner EXCLUDE_FROM_ALL port_network_model_runner.cpp)

# Offline GPU harness: one frame through gather + the frozen plan + scatter, output dumped.
# The bridge's geometry (canvas vs frame, padding, stride) otherwise needs a game launch per guess.
find_library(VULKAN_LIB NAMES vulkan-1 HINTS "$ENV{VULKAN_SDK}/Lib" "$ENV{VULKAN_SDK}/Lib32")
add_executable(port_network_bridge_harness EXCLUDE_FROM_ALL port_network_bridge_harness.cpp)
target_link_libraries(port_network_bridge_harness PRIVATE port_backend ${VULKAN_LIB})

# Runs a single AMD runtime shader (EASU upscale / downscale) on a raw image, so their kernel can be
# compared against our gather/scatter on identical pixels.
add_executable(port_network_amd_probe EXCLUDE_FROM_ALL port_network_amd_probe.cpp)
target_link_libraries(port_network_amd_probe PRIVATE ${VULKAN_LIB})
