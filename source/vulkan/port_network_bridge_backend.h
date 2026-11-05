#pragma once
#include "nrb_vulkan_processor.h"
#include "nr_bridge.h"

// Independent extension: WB's frozen v3 structs are not modified.
// Set v3.abi_size=sizeof(PortNetworkBridgeFrameIO), not sizeof(v3).
// The bridge acquires each image into GENERAL / queue_family before frame(),
// and releases the output after frame(). This metadata describes actual images.
struct PortNetworkBridgeImage {
    VkImage image;
    VkFormat format;
    VkExtent2D extent;
    VkRect2D valid_rect;
    VkImageUsageFlags usage;
    VkImageLayout layout;
    uint32_t queue_family;
};
struct PortNetworkBridgeFrameIO {
    NRVK_BackendFrameIO v3;
    uint64_t frame_id, control_epoch;
    NRB_Controls nrb_controls;
    float model_controls[20]; // Resolved ioport_images.glsl ctl(0..19), never guessed.
    uint32_t noise_seed;       // Explicit shader PCG seed; caller defines frame_id mapping.
    PortNetworkBridgeImage images[5]; // color,motion,depth,exposure,output.
};

// Call once after the bridge's actual vkQueueSubmit result. Successful submit
// transfers the record to its real network completion fence. The bridge must
// not reset that fence until this context has observed completion. Failure
// cancels the record and leaves the committed history generation unchanged.
extern "C" int port_network_bridge_backend_submit_result(NRVK_Backend*, VkFence, VkResult);
extern "C" const char* port_network_bridge_backend_last_error();
extern "C" void nrvk_backend_set_timestamp_query(NRVK_Backend*, VkQueryPool, uint32_t);
// Last active NR grid actually dispatched (top-left region of the frozen
// 1280x720 canvas). Diagnostic only; reports 0,0 before the first frame.
extern "C" void nrvk_backend_last_grid(uint32_t* width, uint32_t* height);

// Required at VkDevice creation: Vulkan1.3, cooperativeMatrix,
// shaderFloat16, storageBuffer16BitAccess, subgroupSizeControl,
// computeFullSubgroups, synchronization2, vulkanMemoryModel=true.
// vulkanMemoryModelDeviceScope/AvailabilityVisibilityChains=false.
// init() checks physical support; it cannot inspect which logical-device
// features WB enabled. WB must supply this exact enabled-feature contract.
// Single sequential context. Adaptive input: render_width/height describe the
// actual source window (any size >=8x8); the gather pass rescales it onto the
// frozen 1280x720 network grid and the scatter pass writes the result back at
// render size (the output surface stays render-sized, never the SR target).
// The temporal history breaks automatically across render-size changes.
// params.output_width/height describe the later SR target. MVLowRes=1 means
// render-resolution motion relative to SR and is supported. Known NGX origins
// already carry current->previous pixel scale; no second route sign conversion.
// depth is optional (controls11=have_depth); output may be RGBA16F or RGBA32F.
// First game route is the existing normalized LDR proxy: exposure1/auto0/HDR0,
// jitter0, non-jittered native MV. Raw HDR, non-unit exposure and nonzero jitter
// require a separately defined front conversion and remain explicit -6.
// input_height must explicitly be720; current NRB UI's360/480/540 modes are
// unsupported. WB needs a declared native720 mode, not a silently ignored knob.
// -6=missing extended fields/unsupported first-path domain,
// -7=record pending or previous network fence not complete, -8=init GPU resource error.
