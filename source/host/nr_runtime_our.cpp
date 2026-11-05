// The `Runtime` replacement seam: upstream's graph executor, replaced by the Intel port's backend.
//
// Linked instead of upstream's `Runtime` when `NR_OUR_RUNTIME=1`; upstream's `nr_runtime.cpp` wraps
// `struct Runtime::Impl` and the `Runtime::` members in `#if !NR_OUR_RUNTIME`, so that translation
// unit keeps everything else it exports. Contract and mapping tables are described alongside the\r\n// host bridge sources.
//
// Why this file exists: `windows/src/pe/nr_pe_session.cpp` (the fence bridge, the D3D12 command-list
// split, NGX feature 18, OptiScaler routing, NR Control) only ever holds a `std::unique_ptr<Runtime>`
// and calls `record*` on it. Swapping this one class puts our network behind that host without
// touching the ~19 955 lines above the seam.
//
// Stage R2: `record_all` now records the Intel port's network. One frame becomes
//   acquire (GENERAL) -> nrvk_backend_frame(cmd) -> release (frame.after)
// with the four input views our gather kernel binds and the caller's colour image written in place,
// which is the same shape `nrb_vulkan_processor.cpp` uses for the game route.
#include "nr_runtime.hpp"
#include "nr_log.hpp"
#include "port_network_bridge_backend.h"   // NRVK_*, PortNetworkBridgeFrameIO, NRB_Controls
// The plan this build ships (rows/buffers/passes), so the count below follows the model instead of
// a number that has to be edited by hand when the build switches to another one.
#include "port_network_bridge_backend_model_identity.inc"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>   // process footprint for the crash heartbeat (see memory_mb)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <cstdint>
#include <map>
#include <utility>

namespace nr {
namespace {

// The plan this build ships: `kPlanRows` rows expanded to `kPlanPasses` dispatches, from the
// generated model identity of the package the build points at (`NR_PORT_MODEL`). `RecordResult::
// network_dispatches` is a log/UI field only, but reporting it honestly is the difference between
// a diagnostic and a decoration. It used to be the literal 232 (the full model at 720p); reading it
// from the generated header is what lets the same source build the Small package (108).
constexpr uint32_t kOurNetworkDispatches = static_cast<uint32_t>(kPlanPasses);

// The game dies every few minutes inside `igc-default64.dll` (the Intel shader compiler) with
// 0xc0000409 / FAST_FAIL_FATAL_APP_EXIT, i.e. an `abort()` from that DLL's CRT, and a fastfail
// cannot be caught by any exception handler we install. Two things we *can* watch from here
// separate the plausible causes: a footprint that keeps growing (something of ours leaking until
// the compiler's own allocation fails) versus a flat footprint (the compiler tripping over the
// shader it is asked to build). Resolved once, from kernel32, so nothing extra has to be linked.
struct ProcFootprint { unsigned long long working_set_mb{}, commit_mb{}; };
ProcFootprint footprint() {
    struct Pmc {
        unsigned long cb;
        unsigned long page_faults;
        size_t peak_ws, ws, quota_peak_paged, quota_paged, quota_peak_nonpaged, quota_nonpaged;
        size_t pagefile, peak_pagefile;
    };
    using Fn = int (*)(void*, Pmc*, unsigned long);
    static Fn fn = []() -> Fn {
        auto load = [](HMODULE m, const char* name) -> Fn {
            return m ? reinterpret_cast<Fn>(reinterpret_cast<void*>(GetProcAddress(m, name))) : nullptr;
        };
        if (Fn f = load(GetModuleHandleA("kernel32.dll"), "K32GetProcessMemoryInfo")) return f;
        return load(LoadLibraryA("psapi.dll"), "GetProcessMemoryInfo");
    }();
    ProcFootprint out{};
    if (!fn) return out;
    Pmc p{};
    p.cb = sizeof(p);
    if (!fn(GetCurrentProcess(), &p, sizeof(p))) return out;
    out.working_set_mb = p.ws >> 20;
    out.commit_mb = p.pagefile >> 20;
    return out;
}

// A barrier with no ownership transfer, exactly as upstream's own `barrier()` does it: the host
// guarantees the images belong to this queue family, and moving them across families here would
// fight the fence bridge.
void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
                   VkPipelineStageFlags src_stage, VkAccessFlags src_access,
                   VkPipelineStageFlags dst_stage, VkAccessFlags dst_access,
                   VkImageAspectFlags aspect) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

VkImageAspectFlags depth_aspects(VkFormat f) {
    switch (f) {
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
        case VK_FORMAT_D32_SFLOAT:
            return VK_IMAGE_ASPECT_DEPTH_BIT;
        case VK_FORMAT_D16_UNORM_S8_UINT:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        default:
            return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

// The twenty floats the network's `ioport_images.glsl` reads, resolved the way AMD's
// nr_graph.cpp:2364-2374 does and the way port_network_model_runner.cpp:resolve_controls already
// mirrors it. Kept identical to that function on purpose: two copies that disagree would make the
// offline number and the in-game picture describe different networks.
//
// c[8] is the *history* word, not a reset flag: 1 = use the history, 0 = treat this frame as the
// first. A reset request therefore writes 0.
void resolve_controls(float c[20], const Controls& ct, uint32_t w, uint32_t h, bool reset,
                      bool have_depth, bool inverted, float mvx, float mvy) {
    const float structure = ct.local_structure;
    const float skin = ct.skin_structure;
    const float mask_skin = ct.automatic_mask ? (skin < 0 ? structure : skin) : -1.0f;
    const float mask_other = ct.automatic_mask ? structure : -1.0f;
    const bool any_mask = std::max(mask_skin, mask_other) >= 0.0f;
    c[0] = float(std::clamp(ct.style, 0, 2)) / 128.0f;                        // s434
    c[1] = ct.local_tone;                                                     // aux0
    c[2] = any_mask ? 1.0f : structure;                                       // aux1
    c[3] = any_mask ? (mask_skin < 0 ? structure : mask_skin) : -1.0f;        // gate0
    c[4] = any_mask ? (mask_other < 0 ? structure : mask_other) : -1.0f;      // gate1
    c[5] = 1.0f;
    c[6] = 1.0f;
    c[7] = ct.intensity;                                                      // model intensity
    c[8] = reset ? 0.0f : 1.0f;                                               // history enable
    c[9] = float(w);
    c[10] = float(h);
    c[11] = have_depth ? 1.0f : 0.0f;
    c[12] = inverted ? 1.0f : 0.0f;
    c[13] = mvx / float(w ? w : 1u);
    c[14] = mvy / float(h ? h : 1u);
    c[15] = 1.0f;
    c[16] = 1.0f;
    c[17] = 0.0f;
    c[18] = 0.0f;
    c[19] = 1.0f;
}

}  // namespace

struct Runtime::Impl {
    VkDevice device{};
    uint32_t family{};
    NRVK_Backend* backend{};

    // Our gather kernel binds four input images. Two of them have no source at this seam:
    //
    //  * motion -- the gather always reads a motion texel and multiplies it through, so a frame
    //    without engine data needs **zeros**, not the colour image (that would hand the network
    //    its own picture as motion vectors). It also has to be a 2-channel format and a full
    //    render-sized allocation, because `frame_supported` checks the format and the crop.
    //    So: one render-sized R32G32F image, cleared once, reused.
    //  * depth -- `frame_supported` accepts a *null* image ("original have_depth=false branch")
    //    and then the gather skips the tap via `pre_offsets.w==0`. Allocating nothing is the
    //    faithful choice; passing the colour image instead is rejected on format.
    VkImage zero_motion_image{};
    VkDeviceMemory zero_motion_memory{};
    VkImageView zero_motion_view{};
    uint32_t zero_motion_w{}, zero_motion_h{};
    bool zero_motion_cleared{};

    // The backend writes its answer into `out_view`, which its own host
    // (nrb_vulkan_processor.cpp:978) always supplies as a *separate* slot image. Binding the
    // colour image there instead reads as a read-modify-write on a storage image and the frame
    // comes back wrong (measured: mean 0.449 vs 0.499, values above 1.0). So the seam owns a
    // render-sized target and copies it back over the colour, which is what makes the upstream
    // "colour is modified in place" contract true again. One full-frame copy per frame.
    VkImage output_image{};
    VkDeviceMemory output_memory{};
    VkImageView output_view{};
    uint32_t output_w{}, output_h{};
    // The target is created in the *frame's* format (RGBA16F in this title), not always 32F: the
    // write-back is a plain image copy, and Vulkan only allows that between identical formats.
    // A 32F target forced a 32F->16F transfer, which is not a legal copy (format mismatch) nor a
    // legal blit (different compatibility classes) - and this driver does not validate, it drops
    // the command. That is the "the network ran, rc=0, and the frame did not change by a single
    // bit" defect; it is invisible in the log because every other number is correct.
    VkFormat output_format{};
    // One-shot readback of the game's colour image right after the write-back (nr_dump_frame.bin),
    // so "did the answer reach the frame" can be answered from files instead of from memory.
    VkBuffer fd_buf{};
    VkDeviceMemory fd_mem{};
    void* fd_map{};
    VkDeviceSize fd_bytes{};
    bool fd_written{};
    // Periodic re-capture from frame 1200 on: frame 3 is still a loading screen, so a sample taken
    // there cannot show the artefact the player sees. Each capture is written 60 frames later.
    bool fd_pending{};
    uint64_t fd_serial{};
    // Signalled right after a frame is recorded, so the backend's -7 gate ("prior record not
    // submitted") is answered. See the call site for why an early signal is safe here.
    VkFence submit_fence{};
    VkQueue queue{};
    // The network's own GPU time. The backend writes timestamps 1..5 around its segments
    // (port_network_bridge_backend.cpp `write_ts`); 5-1 is the whole network. Without it the frame
    // interval (32.76 ms in game) cannot be split into "network" and "everything else".
    // Three pools, rotating: frame N writes pool N%3 and reads pool (N+1)%3, which was written two
    // frames ago. Two frames is enough for the GPU to have finished it, so the read never sees
    // VK_NOT_READY and no WAIT_BIT is needed (that would hang the render thread on a skipped frame).
    VkQueryPool ts_pool[3]{};
    unsigned ts_index{};
    float ts_period{};
    double ts_sum_ms{};
    unsigned ts_frames{};
    unsigned ts_probe{};
    bool ts_inited{};

    // One view per distinct (image, format). The host hands the same images back every frame, so
    // this is a cache and not a leak: everything here is destroyed in ~Runtime.
    struct ViewEntry { VkImageView view; VkFormat format; };
    std::map<VkImage, ViewEntry> views;

    uint32_t model_width{};
    uint32_t model_height{};
    bool logged_frame{};
    // The backend continues its temporal history only when `frame_id == committed_frame + 1`
    // (port_network_bridge_backend.cpp:1017). This used to be hard-coded to 0 on every frame, so
    // the test could never pass and the network ran as a single-frame spatial filter for its whole
    // life: `history=0` in every log line, detail never accumulating, output noisy and unstable.
    // A per-session counter is all the backend needs; it is 1-based because 0 doubles as "none".
    uint64_t frame_serial{};
    // Result of the last fence-only submit: logged by the heartbeat, because a rejected submit is
    // silent otherwise (the backend only loses its commit bookkeeping).
    VkResult last_submit{VK_SUCCESS};
    bool logged_blit{};   // the answer is converted back with a blit rather than an image copy
    bool ready{};

    VkImageView view_for(VkImage image, VkFormat format, VkImageAspectFlags aspect) {
        auto it = views.find(image);
        if (it != views.end() && it->second.format == format) return it->second.view;
        if (it != views.end()) {
            vkDestroyImageView(device, it->second.view, nullptr);
            views.erase(it);
        }
        VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ci.image = image;
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = format;
        ci.subresourceRange = {aspect, 0, 1, 0, 1};
        VkImageView view = VK_NULL_HANDLE;
        const VkResult r = vkCreateImageView(device, &ci, nullptr, &view);
        if (r != VK_SUCCESS) {
            nr::logf("[nr] runtime: vkCreateImageView failed (%d) for format %d", int(r), int(format));
            return VK_NULL_HANDLE;
        }
        views[image] = ViewEntry{view, format};
        return view;
    }

    bool make_output(VkPhysicalDevice phy, uint32_t w, uint32_t h, VkFormat format) {
        if (output_image && output_w == w && output_h == h && output_format == format) return true;
        if (output_image) {
            vkDestroyImageView(device, output_view, nullptr);
            vkDestroyImage(device, output_image, nullptr);
            vkFreeMemory(device, output_memory, nullptr);
            output_image = VK_NULL_HANDLE;
            output_view = VK_NULL_HANDLE;
            output_memory = VK_NULL_HANDLE;
        }
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = format;
        ici.extent = {w, h, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device, &ici, nullptr, &output_image) != VK_SUCCESS) return false;
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device, output_image, &req);
        VkPhysicalDeviceMemoryProperties mem{};
        vkGetPhysicalDeviceMemoryProperties(phy, &mem);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
            if ((req.memoryTypeBits & (1u << i)) &&
                (mem.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                type = i;
                break;
            }
        }
        if (type == UINT32_MAX) return false;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        if (vkAllocateMemory(device, &ai, nullptr, &output_memory) != VK_SUCCESS) return false;
        if (vkBindImageMemory(device, output_image, output_memory, 0) != VK_SUCCESS) return false;
        output_view = view_for(output_image, format, VK_IMAGE_ASPECT_COLOR_BIT);
        output_w = w;
        output_h = h;
        output_format = format;
        return output_view != VK_NULL_HANDLE;
    }

    static std::filesystem::path module_dir() {
        char buf[MAX_PATH]{};
        const unsigned long n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
        if (!n) return std::filesystem::current_path();
        return std::filesystem::path(std::string(buf, n)).parent_path();
    }
    bool make_frame_dump(VkDeviceSize bytes) {
        if (fd_buf) return true;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        // Two halves: [0, bytes) takes the game's colour image after the write-back, [bytes, 2*bytes)
        // takes the network's own target. No single sample can say which of the two is wrong.
        bi.size = bytes * 2;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &bi, nullptr, &fd_buf) != VK_SUCCESS) return false;
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, fd_buf, &req);
        VkPhysicalDeviceMemoryProperties mem{};
        vkGetPhysicalDeviceMemoryProperties(physical, &mem);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
            const auto flags = mem.memoryTypes[i].propertyFlags;
            if ((req.memoryTypeBits & (1u << i)) &&
                (flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                type = i;
                break;
            }
        }
        if (type == UINT32_MAX) return false;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        if (vkAllocateMemory(device, &ai, nullptr, &fd_mem) != VK_SUCCESS) return false;
        if (vkBindBufferMemory(device, fd_buf, fd_mem, 0) != VK_SUCCESS) return false;
        if (vkMapMemory(device, fd_mem, 0, VK_WHOLE_SIZE, 0, &fd_map) != VK_SUCCESS) {
            fd_map = nullptr;
            return false;
        }
        fd_bytes = bytes;
        return true;
    }
    void write_frame_dump() {
        if (!fd_map || !fd_pending || !fd_bytes) return;
        fd_pending = false;
        fd_written = true;
        // Named by the frame it was captured on: several samples per session are needed because a
        // session that dies in the Intel compiler's abort() does not last long enough for one late
        // sample to be enough to tell "already wrong at frame 300" from "wrong only later".
        char name[64];
        std::snprintf(name, sizeof name, "nr_dump_f%04llu.bin",
                      static_cast<unsigned long long>(fd_serial));
        if (FILE* f = std::fopen((module_dir() / name).string().c_str(), "wb")) {
            std::fwrite(fd_map, 1, size_t(fd_bytes), f);
            std::fclose(f);
        }
        std::snprintf(name, sizeof name, "nr_dumpout_f%04llu.bin",
                      static_cast<unsigned long long>(fd_serial));
        if (FILE* f = std::fopen((module_dir() / name).string().c_str(), "wb")) {
            std::fwrite(static_cast<const char*>(fd_map) + fd_bytes, 1, size_t(fd_bytes), f);
            std::fclose(f);
        }
        nr::logf("[nr] runtime: frame and network target dumped at frame %llu (%llu bytes each)",
                 static_cast<unsigned long long>(fd_serial),
                 static_cast<unsigned long long>(fd_bytes));
    }

    bool make_zero_motion(VkPhysicalDevice phy, uint32_t w, uint32_t h) {
        if (zero_motion_image && zero_motion_w == w && zero_motion_h == h) return true;
        if (zero_motion_image) {
            vkDestroyImageView(device, zero_motion_view, nullptr);
            vkDestroyImage(device, zero_motion_image, nullptr);
            vkFreeMemory(device, zero_motion_memory, nullptr);
            zero_motion_image = VK_NULL_HANDLE;
            zero_motion_view = VK_NULL_HANDLE;
            zero_motion_memory = VK_NULL_HANDLE;
            zero_motion_cleared = false;
        }
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R32G32_SFLOAT;
        ici.extent = {w, h, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device, &ici, nullptr, &zero_motion_image) != VK_SUCCESS) return false;
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device, zero_motion_image, &req);
        VkPhysicalDeviceMemoryProperties mem{};
        vkGetPhysicalDeviceMemoryProperties(phy, &mem);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
            if ((req.memoryTypeBits & (1u << i)) &&
                (mem.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                type = i;
                break;
            }
        }
        if (type == UINT32_MAX) return false;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        if (vkAllocateMemory(device, &ai, nullptr, &zero_motion_memory) != VK_SUCCESS) return false;
        if (vkBindImageMemory(device, zero_motion_image, zero_motion_memory, 0) != VK_SUCCESS)
            return false;
        zero_motion_view = view_for(zero_motion_image, VK_FORMAT_R32G32_SFLOAT,
                                    VK_IMAGE_ASPECT_COLOR_BIT);
        zero_motion_w = w;
        zero_motion_h = h;
        zero_motion_cleared = false;
        return zero_motion_view != VK_NULL_HANDLE;
    }

    VkPhysicalDevice physical{};
};

Runtime::Runtime(const HostDevice& host, const RuntimeConfig& config)
    : Runtime(host, config, ControlMaskConfig{}) {}

Runtime::Runtime(const HostDevice& host, const RuntimeConfig& config, const ControlMaskConfig& mask)
    : Runtime(host, config, mask, TemporalConfig{}) {}

Runtime::Runtime(const HostDevice& host, const RuntimeConfig& config, const ControlMaskConfig& mask,
                 const TemporalConfig& temporal)
    : impl_(std::make_unique<Impl>()) {
    (void)mask;
    (void)temporal;
    // The build runs off the render thread, so a build that never finishes looks exactly like
    // "the network is building" forever. These two lines bracket it: the first one proves this
    // constructor was entered at all, the ones in nrvk_backend_init say how far it got.
    nr::logf("[nr] runtime: building the Intel port network, root '%s'", config.root.c_str());
    impl_->device = host.device;
    impl_->physical = host.physical;
    impl_->family = host.queue_family;
    impl_->queue = host.queue;

    // Where the network lives, by AMD's own rule. nr_runtime.cpp:1159 says an installed copy keeps
    // everything it loads in `dlssnr-b390/` beside the DLL and finds it with exactly this test:
    //
    //     const auto data = is_directory(root / "dlssnr-b390") ? root / "dlssnr-b390" : root;
    //
    // Ours keeps the same shape one level in (`dlssnr-b390/manifest/*.json` + `dlssnr-b390/assets/`),
    // because the plan refers to its shaders as `../assets/spv/...`; `dlssnr-b390/manifest` is then
    // the manifest dir and `dlssnr-b390/assets` is its sibling, so the relative paths keep working
    // untouched. The development tree (`local/vk102`) matches the second candidate.
    const std::filesystem::path root(config.root);
    std::filesystem::path manifest_dir = root;
    for (const auto& candidate : {root / "dlssnr-b390" / "manifest", root / "manifest", root}) {
        if (std::filesystem::is_regular_file(candidate / "port_network_manifest.json")) {
            manifest_dir = candidate;
            break;
        }
    }
    const std::string manifest_dir_text = manifest_dir.string();

    NRVK_Config cfg{};
    cfg.abi_size = sizeof(NRVK_Config);
    cfg.manifest_dir = manifest_dir_text.c_str();
    cfg.cache_dir = nullptr;
    cfg.status_json_path = nullptr;
    cfg.output_ring_slots = 3;
    cfg.log_verbose = 0;
    // Hand the backend the extent the session was created for, so it can pick the matching ladder
    // plan before the first dispatch instead of assuming the installed plan's extent.
    cfg.render_width = config.width;
    cfg.render_height = config.height;

    const int rc = nrvk_backend_init(host.instance, host.device, host.physical, host.queue,
                                     host.queue_family, &cfg, &impl_->backend);
    if (rc != 0) {
        impl_->backend = nullptr;
        nr::logf("[nr] runtime: nrvk_backend_init failed (%d) for '%s'; no frame will be recorded",
                 rc, manifest_dir_text.c_str());
        return;
    }
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(impl_->device, &fci, nullptr, &impl_->submit_fence) != VK_SUCCESS)
        nr::logf("[nr] runtime: the submit fence could not be created; every frame after the first "
                 "will be refused (-7)");
    VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qci.queryCount = 8;
    bool pools_ok = true;
    for (int i = 0; i < 3; ++i)
        if (vkCreateQueryPool(impl_->device, &qci, nullptr, &impl_->ts_pool[i]) != VK_SUCCESS)
            pools_ok = false;
    if (pools_ok) {
        nrvk_backend_set_timestamp_query(impl_->backend, impl_->ts_pool[0], 0);
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(impl_->physical, &props);
        impl_->ts_period = props.limits.timestampPeriod;
        nr::logf("[nr] runtime: network GPU timer on (%.3f ns per tick)", double(impl_->ts_period));
    } else {
        nr::logf("[nr] runtime: the timestamp pool could not be created; frame interval only");
        for (int i = 0; i < 3; ++i)
            if (impl_->ts_pool[i]) { vkDestroyQueryPool(impl_->device, impl_->ts_pool[i], nullptr); impl_->ts_pool[i] = VK_NULL_HANDLE; }
    }
    impl_->ready = true;
    nr::logf("[nr] runtime: Intel port executor ready (manifest '%s', %u dispatches)",
             manifest_dir_text.c_str(), kOurNetworkDispatches);
}

Runtime::~Runtime() {
    if (!impl_) return;
    if (impl_->backend) nrvk_backend_release(impl_->backend);
    for (auto& kv : impl_->views) vkDestroyImageView(impl_->device, kv.second.view, nullptr);
    if (impl_->submit_fence) vkDestroyFence(impl_->device, impl_->submit_fence, nullptr);
    for (int i = 0; i < 3; ++i)
        if (impl_->ts_pool[i]) vkDestroyQueryPool(impl_->device, impl_->ts_pool[i], nullptr);
    if (impl_->output_view) vkDestroyImageView(impl_->device, impl_->output_view, nullptr);
    if (impl_->output_image) vkDestroyImage(impl_->device, impl_->output_image, nullptr);
    if (impl_->output_memory) vkFreeMemory(impl_->device, impl_->output_memory, nullptr);
    if (impl_->zero_motion_view) vkDestroyImageView(impl_->device, impl_->zero_motion_view, nullptr);
    if (impl_->zero_motion_image) vkDestroyImage(impl_->device, impl_->zero_motion_image, nullptr);
    if (impl_->zero_motion_memory) vkFreeMemory(impl_->device, impl_->zero_motion_memory, nullptr);
}

// One body behind every entry point, exactly as the upstream comment demands: the mask and
// temporal paths must not be able to drift apart in barriers or ordering.
//
// `engine` is set only by record_engine; `frame` is its colour either way, so the body is the same.
ControlMaskResult Runtime::record_all(VkCommandBuffer cmd, const ColourFrame& frame, const Controls& c,
                                      const ControlMaskFrame* mask, const TemporalFrame* temporal,
                                      bool* consumed, const EngineFrame* engine) {
    ControlMaskResult out{};
    if (consumed) *consumed = false;
    if (!impl_->ready) return out;

    // Frame-interval probe. The host calls this once per frame it hands to the network and the fence
    // bridge serialises us with the game's queue, so the wall time between two calls is the frame time
    // the player sees. Printed every 240 frames (after a 60-frame warm-up) so a normal session leaves
    // a number in dlssnr-b390.log without any overlay: mean, worst and the implied rate.
    {
        static std::chrono::steady_clock::time_point prev{};
        static double sum_ms = 0.0, max_ms = 0.0;
        static unsigned frames = 0;
        const auto now = std::chrono::steady_clock::now();
        if (frames) {
            const double dt = std::chrono::duration<double, std::milli>(now - prev).count();
            if (frames > 60) {
                sum_ms += dt;
                if (dt > max_ms) max_ms = dt;
                if ((frames - 60) % 240 == 0) {
                    nr::logf("[nr] frame interval: %.2f ms mean, %.2f ms worst, over 240 frames (%.1f fps)",
                             sum_ms / 240.0, max_ms, 240000.0 / sum_ms);
                    sum_ms = 0.0; max_ms = 0.0;
                }
            }
            ++frames;
        } else {
            frames = 1;
        }
        prev = now;
    }

    // The network's own GPU time: the backend wrote timestamps 1..5 around its segments last frame,
    // and the frame interval printed above is what the player waits for. The difference between the
    // two is everything that is not the network (game render, blit, composite, fence waits).
    if (impl_->ts_pool[0]) {
        // A query pool must be reset before its first write; there is no command buffer at init, so
        // the first recorded frame does both pools.
        if (!impl_->ts_inited) {
            for (int i = 0; i < 3; ++i) vkCmdResetQueryPool(cmd, impl_->ts_pool[i], 0, 6);
            impl_->ts_inited = true;
        }
        const unsigned write_idx = impl_->ts_index;
        const unsigned read_idx = (write_idx + 1u) % 3u;   // written two frames ago
        uint64_t ts[6] = {};
        const VkResult r = vkGetQueryPoolResults(impl_->device, impl_->ts_pool[read_idx], 0, 6,
                                                 sizeof ts, ts, sizeof(uint64_t),
                                                 VK_QUERY_RESULT_64_BIT);
        if (r == VK_SUCCESS && ts[5] > ts[1] && ts[1]) {
            const double ms = double(ts[5] - ts[1]) * double(impl_->ts_period) / 1e6;
            impl_->ts_sum_ms += ms;
            if (++impl_->ts_frames % 240 == 0) {
                nr::logf("[nr] network gpu: %.2f ms mean over 240 frames", impl_->ts_sum_ms / 240.0);
                impl_->ts_sum_ms = 0.0;
            }
        } else if (impl_->ts_probe < 5) {
            ++impl_->ts_probe;
            nr::logf("[nr] network gpu: no result yet (r=%d ts1=%llu ts5=%llu)", int(r),
                     (unsigned long long)ts[1], (unsigned long long)ts[5]);
        }
        // Clear the pool just read, then point the backend at the other one for this frame.
        vkCmdResetQueryPool(cmd, impl_->ts_pool[read_idx], 0, 6);
        nrvk_backend_set_timestamp_query(impl_->backend, impl_->ts_pool[write_idx], 0);
        impl_->ts_index = (write_idx + 1u) % 3u;
    }

    const uint32_t rw = frame.width;
    const uint32_t rh = frame.height;
    if (!rw || !rh) return out;

    const bool have_motion = engine && engine->motion.image;
    const bool have_depth = engine && engine->depth.image;
    const bool reset = engine ? engine->reset : (temporal ? temporal->reset : false);

    // --- views: the order our gather kernel binds (colour, motion, depth, exposure) -------------
    VkImageView colour_view = impl_->view_for(frame.image, frame.format, VK_IMAGE_ASPECT_COLOR_BIT);
    if (!colour_view) return out;

    VkImageView motion_view = VK_NULL_HANDLE;
    if (have_motion) {
        motion_view = impl_->view_for(engine->motion.image, engine->motion.format,
                                      VK_IMAGE_ASPECT_COLOR_BIT);
    } else {
        // A render-sized zero field: `frame_supported` checks the motion format (2 channels) and
        // the crop, so a 1x1 stand-in would be rejected before it ever reached the shader.
        if (!impl_->make_zero_motion(impl_->physical, rw, rh)) {
            nr::logf("[nr] runtime: zero-motion stand-in could not be created");
            return out;
        }
        if (!impl_->zero_motion_cleared) {
            image_barrier(cmd, impl_->zero_motion_image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_IMAGE_ASPECT_COLOR_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, impl_->zero_motion_image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            image_barrier(cmd, impl_->zero_motion_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                          VK_IMAGE_ASPECT_COLOR_BIT);
            impl_->zero_motion_cleared = true;
        }
        motion_view = impl_->zero_motion_view;
    }
    VkImageView depth_view = have_depth
        ? impl_->view_for(engine->depth.image, engine->depth.format,
                          depth_aspects(engine->depth.format))
        : VK_NULL_HANDLE;
    if (!motion_view) return out;
    if (have_depth && !depth_view) return out;

    // --- the colour image is written in place, so it enters as a storage-capable GENERAL target --
    const bool need_in = frame.before != VK_IMAGE_LAYOUT_GENERAL;
    image_barrier(cmd, frame.image, need_in ? frame.before : VK_IMAGE_LAYOUT_GENERAL,
                  VK_IMAGE_LAYOUT_GENERAL,
                  need_in ? frame.before_stage : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  need_in ? frame.before_access
                          : (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT),
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT);

    // --- the frame, in the shape nrb_vulkan_processor.cpp uses for the game route ----------------
    // The colour image this title hands us is RGBA16F (the log prints format 97). The target is
    // therefore created in that same format, so the write-back below is a same-format image copy,
    // which is the only transfer Vulkan allows here.
    const VkFormat out_format = (frame.format == VK_FORMAT_R16G16B16A16_SFLOAT ||
                                 frame.format == VK_FORMAT_R32G32B32A32_SFLOAT)
                                    ? frame.format : VK_FORMAT_R32G32B32A32_SFLOAT;
    if (!impl_->make_output(impl_->physical, rw, rh, out_format)) {
        nr::logf("[nr] runtime: output target could not be created");
        return out;
    }
    image_barrier(cmd, impl_->output_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT);
    // depth may be null (the no-engine branch); the backend only demands views for images it is
    // given, and samples exposure only on the HDR path this build never takes.
    VkImageView views[4] = {colour_view, motion_view, depth_view, colour_view};

    NRVK_FrameParams fp{};
    fp.abi_size = sizeof(NRVK_FrameParams);
    fp.render_width = rw;
    fp.render_height = rh;
    fp.output_width = rw;
    fp.output_height = rh;
    fp.motion_scale_x = engine ? engine->motion_scale_x : 1.0f / float(rw ? rw : 1u);
    fp.motion_scale_y = engine ? engine->motion_scale_y : 1.0f / float(rh ? rh : 1u);
    fp.motion_base_x = engine ? engine->motion_x : 0;
    fp.motion_base_y = engine ? engine->motion_y : 0;
    fp.depth_base_x = engine ? engine->depth_x : 0;
    fp.depth_base_y = engine ? engine->depth_y : 0;
    fp.depth_inverted = (engine && engine->depth_inverted) ? 1u : 0u;
    fp.reset_history = reset ? 1u : 0u;
    fp.low_resolution_motion = 0;
    fp.hdr_input = 0;          // LDR proxy path
    fp.auto_exposure = 0;
    fp.pre_exposure = 1.0f;
    fp.exposure_scale = 1.0f;
    fp.jitter_x = fp.jitter_y = 0.0f;
    fp.model_intensity = c.intensity;
    fp.local_tone = c.local_tone;
    fp.local_structure = c.local_structure;

    PortNetworkBridgeFrameIO ext{};
    ext.v3.abi_size = sizeof(PortNetworkBridgeFrameIO);
    ext.v3.io_view_count = 4;
    ext.v3.io_views = views;
    ext.v3.out_view = impl_->output_view;   // the backend's own target; copied back below
    ext.v3.params = &fp;
    ext.frame_id = ++impl_->frame_serial;
    ext.control_epoch = 0;
    // The backend validates this block itself (`frame_supported`, port_network_bridge_backend.cpp:559)
    // and rejects the frame with -6 when `abi_size`, `input_height` or the motion origin are left
    // at zero -- which is exactly what a zero-initialised struct is. The upstream `Controls` has
    // no equivalent of these three, so the seam has to state them from what it does know.
    ext.nrb_controls.abi_size = sizeof(NRB_Controls);
    ext.nrb_controls.enabled = c.enabled ? 1u : 0u;
    ext.nrb_controls.input_height = rh;
    ext.nrb_controls.style = uint32_t(std::clamp(c.style, 0, 2));
    ext.nrb_controls.history = reset ? NRB_HISTORY_RESET : NRB_HISTORY_REFERENCE;
    ext.nrb_controls.graph_replay = 0;
    ext.nrb_controls.auto_mask = c.automatic_mask ? 1u : 0u;
    ext.nrb_controls.skin_structure_enabled = c.skin_structure >= 0.0f ? 1u : 0u;
    ext.nrb_controls.display_strength = 1.0f;
    ext.nrb_controls.model_intensity = c.intensity;
    ext.nrb_controls.local_tone = c.local_tone;
    ext.nrb_controls.local_structure = c.local_structure;
    ext.nrb_controls.skin_structure = c.skin_structure;
    // The upstream host has no motion-origin concept; name the one this route actually delivers.
    fp.motion_scale_origin = NRB_MV_DLSS_NGX;
    // `resolve_controls` takes the motion scale in *source pixels* and divides by the extent
    // itself, exactly as the runner calls it. Upstream states the same quantity as a normalised
    // conversion, so multiply back out: at 1:1 both give c[13] = 1/width, which is what an
    // offline measurement with zero guide planes uses.
    const float mvx_px = engine ? engine->motion_scale_x * float(rw) : 1.0f;
    const float mvy_px = engine ? engine->motion_scale_y * float(rh) : 1.0f;
    resolve_controls(ext.model_controls, c, rw, rh, reset, have_depth,
                     engine && engine->depth_inverted, mvx_px, mvy_px);
    ext.noise_seed = 0;
    ext.images[0] = {frame.image, frame.format, {rw, rh}, {{0, 0}, {rw, rh}}, frame.usage,
                     VK_IMAGE_LAYOUT_GENERAL, impl_->family};
    if (have_motion) {
        const uint32_t mw = engine->motion_texture_width ? engine->motion_texture_width
                                                         : engine->motion.width;
        const uint32_t mh = engine->motion_texture_height ? engine->motion_texture_height
                                                          : engine->motion.height;
        ext.images[1] = {engine->motion.image, engine->motion.format, {mw, mh},
                         {{int32_t(engine->motion_x), int32_t(engine->motion_y)},
                          {engine->motion.width, engine->motion.height}},
                         VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_GENERAL, impl_->family};
    } else {
        ext.images[1] = {impl_->zero_motion_image, VK_FORMAT_R32G32_SFLOAT, {rw, rh},
                         {{0, 0}, {rw, rh}}, VK_IMAGE_USAGE_SAMPLED_BIT,
                         VK_IMAGE_LAYOUT_GENERAL, impl_->family};
    }
    if (have_depth) {
        ext.images[2] = {engine->depth.image, engine->depth.format,
                         {engine->depth.width, engine->depth.height},
                         {{int32_t(engine->depth_x), int32_t(engine->depth_y)},
                          {engine->depth.width, engine->depth.height}},
                         VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_GENERAL, impl_->family};
    }
    // else: leave images[2] null. `frame_supported` accepts that as the original's
    // have_depth=false branch, and the gather then skips the tap. Pointing it at the colour image
    // instead is rejected on format -- RGBA32F is not one of the accepted depth formats.
    ext.images[3] = ext.images[0];   // exposure stand-in: the LDR path never reads it
    ext.images[4] = {impl_->output_image, impl_->output_format, {rw, rh},
                     {{0, 0}, {rw, rh}}, VK_IMAGE_USAGE_STORAGE_BIT,
                     VK_IMAGE_LAYOUT_GENERAL, impl_->family};

    const int rc = nrvk_backend_frame(impl_->backend, cmd, &ext.v3);
    if (rc == 0 && impl_->submit_fence) {
        // The backend refuses the *next* frame until it is told this one was submitted
        // (port_network_bridge_backend.cpp:738 -7, cleared only by submit_result). Upstream submits
        // the recorded command buffer on its own timeline, so `Runtime::record` never holds that
        // fence. The signal we can give honestly is on the same queue, after the record; the D3D12
        // side waits for our work before the continuation runs, so an early signal cannot let two
        // frames overlap in practice.
        vkResetFences(impl_->device, 1, &impl_->submit_fence);
        // A fence-only submit still has to carry a valid VkSubmitInfo. `submitCount = 0,
        // pSubmits = nullptr` is legal Vulkan, but OptiScaler's Vulkan hook rejects a null
        // pSubmits outright (`VulkanwDx12_Hooks.cpp` hk_vkQueueSubmit: "Invalid parameters to
        // hk_vkQueueSubmit" -> VK_ERROR_INITIALIZATION_FAILED) - which is why every `history=`
        // line in the log said `committed=0`: the fence submit failed on every frame, so the
        // backend was never told a frame had been submitted. One *empty* VkSubmitInfo submits
        // nothing, passes that check and still signals the fence once the queue's earlier work
        // has completed.
        VkSubmitInfo fence_only{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        impl_->last_submit = vkQueueSubmit(impl_->queue, 1, &fence_only, impl_->submit_fence);
        if (impl_->last_submit == VK_SUCCESS)
            port_network_bridge_backend_submit_result(impl_->backend, impl_->submit_fence, VK_SUCCESS);
        else
            port_network_bridge_backend_submit_result(impl_->backend, VK_NULL_HANDLE, VK_ERROR_UNKNOWN);
    }
    if (!impl_->logged_frame) {
        nr::logf("[nr] runtime: first frame %ux%u motion=%d depth=%d reset=%d -> rc=%d", rw, rh,
                 have_motion ? 1 : 0, have_depth ? 1 : 0, reset ? 1 : 0, rc);
        impl_->logged_frame = true;
        // Written straight to disk rather than through the log sink: this route leaves the sink
        // unset, so `logf` goes to stdout and a GUI process never shows it. The extents here are
        // the textures the gather actually samples; a guide that does not match the colour extent
        // is invisible in every other diagnostic and produces exactly a picture-shaped artefact.
        if (FILE* f = std::fopen((Impl::module_dir() / "nr_probe.txt").string().c_str(), "w")) {
            std::fprintf(f, "frame      %ux%u fmt %d reset %d\n", rw, rh, int(frame.format), reset ? 1 : 0);
            std::fprintf(f, "motion     %ux%u fmt %d have %d\n", ext.images[1].extent.width,
                         ext.images[1].extent.height, int(ext.images[1].format), have_motion ? 1 : 0);
            std::fprintf(f, "depth      %ux%u fmt %d have %d\n", ext.images[2].extent.width,
                         ext.images[2].extent.height, int(ext.images[2].format), have_depth ? 1 : 0);
            std::fprintf(f, "output     %ux%u fmt %d\n", ext.images[4].extent.width,
                         ext.images[4].extent.height, int(ext.images[4].format));
            std::fclose(f);
        }
    }
    // Every 600 frames (~16 s at this frame rate): footprint and the fence-submit result, so a
    // session that ends in the Intel compiler's abort() can be told apart from a plain driver bug.
    if (impl_->frame_serial % 600 == 0) {
        const ProcFootprint fp = footprint();
        nr::logf("[nr] runtime: heartbeat frame %llu ws %llu MB commit %llu MB fence-submit %d",
                 static_cast<unsigned long long>(impl_->frame_serial), fp.working_set_mb,
                 fp.commit_mb, static_cast<int>(impl_->last_submit));
    }
    if (rc != 0) {
        // A refused frame used to cost a whole game restart to diagnose: the log said `rc=-6` and
        // nothing else. -6 is `frame_supported()`, and its own refusal reason names the exact
        // condition, so print it together with the guide sizes -- the usual cause is the host
        // handing a post-upscale colour with render-size motion/depth, which cannot line up.
        nr::logf("[nr] runtime: frame refused (%d): %s | colour %ux%u, motion %ux%u, depth null=%d %ux%u",
                 rc, port_network_bridge_backend_last_error ? port_network_bridge_backend_last_error() : "?",
                 rw, rh, ext.images[1].extent.width, ext.images[1].extent.height,
                 have_depth ? 0 : 1, ext.images[2].extent.width, ext.images[2].extent.height);
        return out;
    }

    // Fulfil the upstream contract: `ColourFrame::image` is "modified in place". The backend wrote
    // a separate target, so bring it back.
    image_barrier(cmd, impl_->output_image, VK_IMAGE_LAYOUT_GENERAL,
                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    image_barrier(cmd, frame.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT);
    {
        // **A transfer between RGBA32F and RGBA16F is not a legal command.** vkCmdCopyImage requires
        // identical formats, and vkCmdBlitImage only accepts formats of one compatibility class, so
        // neither converts 32F to 16F. This driver does not validate and drops the command, which is
        // precisely "the network ran, rc=0, and the frame did not change by a single bit" - and the
        // fix is to not convert at all: the target is created in the frame's own format, and what
        // follows is a same-format copy.
        if (frame.format == impl_->output_format) {
            VkImageCopy region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.extent = {rw, rh, 1};
            vkCmdCopyImage(cmd, impl_->output_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           frame.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        } else {
            // Unreachable for a 16F or 32F frame (the target takes the frame's format); kept as a
            // last resort for an exotic one, where the driver may or may not honour the blit.
            VkImageBlit region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.srcOffsets[1] = {int32_t(rw), int32_t(rh), 1};
            region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.dstOffsets[1] = {int32_t(rw), int32_t(rh), 1};
            vkCmdBlitImage(cmd, impl_->output_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           frame.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region,
                           VK_FILTER_NEAREST);
        }
        if (!impl_->logged_blit) {
            impl_->logged_blit = true;
            nr::logf("[nr] runtime: answer published to the frame as a %s (output format %d, "
                     "frame format %d)", frame.format == impl_->output_format
                         ? "same-format image copy" : "cross-format blit",
                     int(impl_->output_format), int(frame.format));
        }
    }
    // --- proof channel: what is actually in the game's colour image after the write-back --------
    // Re-captured every 600 frames from 1200 on into a host-visible buffer and written to
    // <exe>/nr_dump_frame.bin 60 frames later, by which time the GPU has long finished it. Frame 3
    // was the old sample point, but that is still a loading screen: the artefact the player sees
    // only exists once the world is drawing, so the sample point moved to a steady frame.
    const bool wide = impl_->output_format == VK_FORMAT_R16G16B16A16_SFLOAT;
    const VkDeviceSize bytes = VkDeviceSize(rw) * rh * (wide ? 8u : 16u);
    // A session here ends in the Intel compiler's abort() after a couple of minutes, so sampling
    // from frame 1200 on can miss the artefact entirely: capture every 300 frames from 300, and
    // write it 60 frames later (the GPU has finished it by then).
    //
    // 2026-11-04: gated behind NR_DEBUG_DUMP=1 and off by default. It was unconditional, so the
    // shipped core wrote a 9.2 MB pair every 300 frames for as long as the game ran (measured:
    // 234 files / 1.0 GB in 30 minutes), which showed up as ~228 ms frame spikes and added commit
    // pressure right where the memory guard retires networks.
    static const bool fd_debug = [] {
        char v[8] = {};
        const DWORD n = GetEnvironmentVariableA("NR_DEBUG_DUMP", v, sizeof v);
        return n > 0 && v[0] == '1';
    }();
    const bool fd_capture = fd_debug && impl_->frame_serial >= 300 && impl_->frame_serial % 300 == 0;
    if (frame.image && (fd_capture || impl_->fd_pending)) {
        if (impl_->frame_serial % 300 == 60) {
            impl_->write_frame_dump();
        } else if (fd_capture) {
            if (impl_->make_frame_dump(bytes)) {
                impl_->fd_serial = impl_->frame_serial;
                image_barrier(cmd, frame.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
                VkBufferImageCopy region{};
                region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.imageExtent = {rw, rh, 1};
                vkCmdCopyImageToBuffer(cmd, frame.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       impl_->fd_buf, 1, &region);
                // The network's own target, into the second half. It is still in TRANSFER_SRC from
                // the write-back above, so no barrier is needed here.
                VkBufferImageCopy net_region = region;
                net_region.bufferOffset = impl_->fd_bytes;
                vkCmdCopyImageToBuffer(cmd, impl_->output_image,
                                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, impl_->fd_buf, 1,
                                       &net_region);
                image_barrier(cmd, frame.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
                impl_->fd_pending = true;
            }
        }
    }
    image_barrier(cmd, frame.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT);

    impl_->model_width = rw;
    impl_->model_height = rh;

    // --- release: whatever the caller said the image must be in afterwards ----------------------
    if (frame.after != VK_IMAGE_LAYOUT_GENERAL) {
        image_barrier(cmd, frame.image, VK_IMAGE_LAYOUT_GENERAL, frame.after,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                      frame.after_stage, frame.after_access, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    out.frame.network_dispatches = kOurNetworkDispatches;
    out.frame.applied = c.enabled && c.apply_model;
    out.frame.effective_skin = c.skin_structure;
    out.frame.effective_background = c.local_structure;
    return out;
}

RecordResult Runtime::record(VkCommandBuffer cmd, const ColourFrame& frame, const Controls& c) {
    return record_all(cmd, frame, c, nullptr, nullptr, nullptr, nullptr).frame;
}

RecordResult Runtime::record(VkCommandBuffer cmd, const ColourFrame& frame, const Controls& c,
                             bool reset) {
    TemporalFrame t{};
    t.reset = reset;
    return record_all(cmd, frame, c, nullptr, &t, nullptr, nullptr).frame;
}

ControlMaskResult Runtime::record(VkCommandBuffer cmd, const ColourFrame& frame, const Controls& c,
                                  const ControlMaskFrame* mask) {
    bool consumed = false;
    ControlMaskResult out = record_all(cmd, frame, c, mask, nullptr, &consumed, nullptr);
    out.consumed = consumed;
    return out;
}

TemporalResult Runtime::record_temporal(VkCommandBuffer cmd, const ColourFrame& frame,
                                        const Controls& c, const TemporalFrame& temporal) {
    TemporalResult out{};
    out.frame = record_all(cmd, frame, c, nullptr, &temporal, nullptr, nullptr).frame;
    // Our network blends in its own history inside nrvk_bridge, so the history *was* consumed --
    // just not at this layer. Reporting false would tell the host to hold the current colour.
    out.history_consumed = out.frame.network_dispatches != 0;
    out.motion_provider = "none";
    return out;
}

EngineResult Runtime::record_engine(VkCommandBuffer cmd, const EngineFrame& frame,
                                    const Controls& c) {
    EngineResult out{};
    out.frame = record_all(cmd, frame.colour, c, nullptr, nullptr, nullptr, &frame).frame;
    out.history_consumed = out.frame.network_dispatches != 0;
    out.motion_provider = "engine";
    return out;
}

bool Runtime::takes_target(const Controls&) const {
    // Our scatter writes back at render resolution, never into the SR target.
    return false;
}

void Runtime::release_feature(uint64_t feature) { (void)feature; }
void Runtime::drop_history() {}
void Runtime::forget_history() {}
uint32_t Runtime::live_features() const { return 0; }

Runtime::TemporalHistoryView Runtime::temporal_history() const { return {}; }

uint32_t Runtime::model_width() const { return impl_ ? impl_->model_width : 0; }
uint32_t Runtime::model_height() const { return impl_ ? impl_->model_height : 0; }

NetworkModel Runtime::network() const { return NetworkModel::Original; }

bool Runtime::int4_mixed() const { return false; }

void Runtime::set_history_strength(float v) { (void)v; }
void Runtime::set_white_point(float v) { (void)v; }

float Runtime::last_gpu_ms() const { return 0.0f; }
float Runtime::average_gpu_ms() const { return 0.0f; }
float Runtime::average_network_ms() const { return 0.0f; }

std::pair<float, float> Runtime::preprocess_meter() const { return {NAN, NAN}; }

}  // namespace nr
