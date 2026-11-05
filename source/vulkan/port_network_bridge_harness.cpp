// Offline GPU harness for the bridge: one real frame through gather + the frozen plan + scatter.
//
// Why this exists: the bridge's geometry (canvas vs frame, padding, stride, centring) has only
// ever been exercised by launching Cyberpunk, and every wrong guess cost a game session. This
// program creates its own Vulkan device with the same feature contract the game gives the backend,
// feeds a synthetic marker frame or a raw RGBA f32 file, records exactly one frame through
// `nrvk_backend_frame`, reads the output image back and writes it out with statistics.
//
//     port_network_bridge_harness --manifest <dir> --render 1280x720 --out <dir> [--input file]
//                                  [--frames 2] [--pattern marker|ramp|file]
//
// The marker pattern puts a coloured square in each corner and a red/green gradient across the
// frame, so a shear (stride error), a centring offset or a channel mix is obvious in the dump and
// measurable in the printed corner positions.
#define NOMINMAX
#include <windows.h>
#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include "nrb_vulkan_processor.h"
#include "port_network_bridge_backend.h"

extern "C" int port_network_bridge_backend_submit_result(NRVK_Backend*, VkFence, VkResult);

#define VC(x) do { const VkResult r_=(x); if(r_!=VK_SUCCESS){ std::printf("FAIL %s -> %d\n",#x,int(r_)); return 2; } } while(0)

static std::vector<char> read_file(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return {};
    std::fseek(f, 0, SEEK_END); const long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> bytes(static_cast<size_t>(n));
    if (std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) bytes.clear();
    std::fclose(f);
    return bytes;
}

static uint16_t to_half(float v) {
    uint32_t b; std::memcpy(&b, &v, 4);
    const uint32_t sign = (b >> 16) & 0x8000u, e = (b >> 23) & 0xffu, m = b & 0x7fffffu;
    if (e == 255u) return uint16_t(sign | 0x7c00u | (m ? 0x200u : 0u));
    if (e < 113u) return uint16_t(sign);
    if (e > 142u) return uint16_t(sign | 0x7c00u);
    const uint32_t rounded = m + 0xfffu + ((m >> 13) & 1u);
    return uint16_t(sign | (((e - 112u) << 10) + (rounded >> 13)));
}

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t w = 0, h = 0;
};

extern "C" int nrvk_backend_buffer_info(void* backend, const char* name, void** handle,
                                        unsigned long long* bytes);

int main(int argc, char** argv) {
    std::string manifest, out_dir, input_path, pattern = "marker", dump_spec;
    uint32_t W = 1280, H = 720, frames = 2;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--manifest") manifest = next();
        else if (a == "--out") out_dir = next();
        else if (a == "--input") { input_path = next(); pattern = "file"; }
        else if (a == "--pattern") pattern = next();
        else if (a == "--dump-buffer") dump_spec = next();
        else if (a == "--frames") frames = uint32_t(std::atoi(next()));
        else if (a == "--render") { const std::string s = next(); if (const auto x = s.find('x'); x != std::string::npos) { W = std::atoi(s.substr(0, x).c_str()); H = std::atoi(s.substr(x + 1).c_str()); } }
        else { std::printf("unknown argument %s\n", a.c_str()); return 1; }
    }
    if (manifest.empty() || out_dir.empty()) {
        std::printf("usage: --manifest DIR --out DIR [--render WxH] [--input raw] [--frames N]\n");
        return 1;
    }
    CreateDirectoryA(out_dir.c_str(), nullptr);

    // ---------------------------------------------------------------- device
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    app.pApplicationName = "nr-bridge-harness";
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) { std::printf("FAIL vkCreateInstance\n"); return 2; }

    uint32_t count = 0; vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count); vkEnumeratePhysicalDevices(instance, &count, devices.data());
    VkPhysicalDevice phy = VK_NULL_HANDLE;
    for (auto d : devices) {
        VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(d, &p);
        if (p.vendorID == 0x8086 && (p.deviceID == 0xb080 || p.deviceID == 0xe20b)) { phy = d; break; }
    }
    if (!phy) { std::printf("FAIL no Intel Arc B580/B390\n"); return 2; }
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(phy, &props);
    std::printf("device: %s\n", props.deviceName);

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceShaderFloat16Int8Features half{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDevice16BitStorageFeatures storage{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDeviceSubgroupSizeControlFeatures sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    VkPhysicalDeviceSynchronization2Features sync{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
    VkPhysicalDeviceVulkanMemoryModelFeatures mm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES};
    VkPhysicalDeviceFeatures2 feats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    feats.pNext = &cm; cm.pNext = &half; half.pNext = &storage; storage.pNext = &sg; sg.pNext = &sync; sync.pNext = &mm;
    vkGetPhysicalDeviceFeatures2(phy, &feats);

    uint32_t family = 0; bool found = false;
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(phy, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qs(qn); vkGetPhysicalDeviceQueueFamilyProperties(phy, &qn, qs.data());
    for (uint32_t i = 0; i < qn; ++i) if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { family = i; found = true; break; }
    if (!found) { std::printf("FAIL no compute queue\n"); return 2; }

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    std::vector<const char*> exts = {"VK_KHR_cooperative_matrix"};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &feats; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(exts.size()); dci.ppEnabledExtensionNames = exts.data();
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(phy, &dci, nullptr, &device) != VK_SUCCESS) { std::printf("FAIL vkCreateDevice\n"); return 2; }
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, family, 0, &queue);

    auto CreateImage = [&](Image& im, VkFormat fmt, uint32_t w, uint32_t h) {
        im.format = fmt; im.w = w; im.h = h;
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D; ci.format = fmt; ci.extent = {w, h, 1};
        ci.mipLevels = 1; ci.arrayLayers = 1; ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE; ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VC(vkCreateImage(device, &ci, nullptr, &im.image));
        VkMemoryRequirements req{}; vkGetImageMemoryRequirements(device, im.image, &req);
        VkPhysicalDeviceMemoryProperties mem{}; vkGetPhysicalDeviceMemoryProperties(phy, &mem);
        uint32_t type = 0; while (type < mem.memoryTypeCount && !(req.memoryTypeBits & (1u << type))) ++type;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size; ai.memoryTypeIndex = type;
        VC(vkAllocateMemory(device, &ai, nullptr, &im.memory));
        VC(vkBindImageMemory(device, im.image, im.memory, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = im.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VC(vkCreateImageView(device, &vi, nullptr, &im.view));
        return 0;
    };

    Image color, motion, output;
    CreateImage(color, VK_FORMAT_R16G16B16A16_SFLOAT, W, H);
    CreateImage(motion, VK_FORMAT_R16G16_SFLOAT, W, H);
    CreateImage(output, VK_FORMAT_R16G16B16A16_SFLOAT, W, H);

    // ---------------------------------------------------------------- host data
    std::vector<uint16_t> color_px(size_t(W) * H * 4), motion_px(size_t(W) * H * 2, to_half(0.0f));
    if (pattern == "file") {
        auto bytes = read_file(input_path.c_str());
        const size_t need = size_t(W) * H * 4 * sizeof(float);
        if (bytes.size() < need) { std::printf("FAIL input is not %zux%u RGBA f32 (%zu < %zu)\n", size_t(W), H, bytes.size(), need); return 2; }
        const float* src = reinterpret_cast<const float*>(bytes.data());
        const size_t n = size_t(W) * H;
        for (size_t i = 0; i < n; ++i)
            for (int c = 0; c < 4; ++c) color_px[i * 4 + c] = to_half(src[c * n + i]);
    } else {
        for (uint32_t y = 0; y < H; ++y) for (uint32_t x = 0; x < W; ++x) {
            float r = float(x) / float(W - 1), g = float(y) / float(H - 1), b = 0.5f;
            auto box = [&](uint32_t bx, uint32_t by, float br, float bg, float bb) {
                if (x >= bx && x < bx + 24 && y >= by && y < by + 24) { r = br; g = bg; b = bb; }
            };
            box(16, 16, 1.0f, 0.0f, 0.0f); box(W - 40, 16, 0.0f, 1.0f, 0.0f);
            box(16, H - 40, 0.0f, 0.0f, 1.0f); box(W - 40, H - 40, 1.0f, 1.0f, 0.0f);
            const size_t i = size_t(y) * W + x;
            color_px[i * 4 + 0] = to_half(r); color_px[i * 4 + 1] = to_half(g);
            color_px[i * 4 + 2] = to_half(b); color_px[i * 4 + 3] = to_half(1.0f);
        }
    }

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = family; pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool = VK_NULL_HANDLE; VC(vkCreateCommandPool(device, &pci, nullptr, &pool));
    VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbi.commandPool = pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE; VC(vkAllocateCommandBuffers(device, &cbi, &cb));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE; VC(vkCreateFence(device, &fci, nullptr, &fence));

    auto upload = [&](Image& im, const void* data, VkDeviceSize bytes) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = bytes; bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VkBuffer staging = VK_NULL_HANDLE; VC(vkCreateBuffer(device, &bci, nullptr, &staging));
        VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(device, staging, &req);
        VkPhysicalDeviceMemoryProperties mem{}; vkGetPhysicalDeviceMemoryProperties(phy, &mem);
        uint32_t type = 0;
        while (type < mem.memoryTypeCount && !((req.memoryTypeBits & (1u << type)) &&
               (mem.memoryTypes[type].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
               (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) ++type;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = req.size; ai.memoryTypeIndex = type;
        VkDeviceMemory smem = VK_NULL_HANDLE; VC(vkAllocateMemory(device, &ai, nullptr, &smem));
        VC(vkBindBufferMemory(device, staging, smem, 0));
        void* map = nullptr; VC(vkMapMemory(device, smem, 0, VK_WHOLE_SIZE, 0, &map));
        std::memcpy(map, data, size_t(bytes)); vkUnmapMemory(device, smem);

        VC(vkResetCommandBuffer(cb, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VC(vkBeginCommandBuffer(cb, &bi));
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = im.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkBufferImageCopy region{}; region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {im.w, im.h, 1};
        vkCmdCopyBufferToImage(cb, staging, im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VC(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        VC(vkQueueSubmit(queue, 1, &si, fence));
        VC(vkWaitForFences(device, 1, &fence, VK_TRUE, 60ull * 1000 * 1000 * 1000));
        VC(vkResetFences(device, 1, &fence));
        vkDestroyBuffer(device, staging, nullptr); vkFreeMemory(device, smem, nullptr);
        return 0;
    };
    if (upload(color, color_px.data(), color_px.size() * 2)) return 2;
    if (upload(motion, motion_px.data(), motion_px.size() * 2)) return 2;

    // output needs to be in GENERAL too
    {
        VC(vkResetCommandBuffer(cb, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VC(vkBeginCommandBuffer(cb, &bi));
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = output.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VC(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        VC(vkQueueSubmit(queue, 1, &si, fence));
        VC(vkWaitForFences(device, 1, &fence, VK_TRUE, 60ull * 1000 * 1000 * 1000));
        VC(vkResetFences(device, 1, &fence));
    }

    // ---------------------------------------------------------------- backend
    NRVK_Config cfg{};
    cfg.abi_size = sizeof(cfg);
    cfg.manifest_dir = manifest.c_str();
    cfg.output_ring_slots = 3;
    NRVK_Backend* backend = nullptr;
    if (int rc = nrvk_backend_init(instance, device, phy, queue, family, &cfg, &backend); rc != 0) {
        std::printf("FAIL nrvk_backend_init rc=%d: %s\n", rc, port_network_bridge_backend_last_error());
        return 3;
    }

    VkImageView views[4] = {color.view, motion.view, VK_NULL_HANDLE, VK_NULL_HANDLE};
    NRVK_FrameParams p{};
    p.abi_size = sizeof(p);
    p.motion_scale_x = float(W); p.motion_scale_y = float(H);
    p.motion_scale_origin = 1;
    p.pre_exposure = 1.0f; p.exposure_scale = 1.0f;
    p.hdr_input = 0;
    p.render_width = W; p.render_height = H;
    p.output_width = W; p.output_height = H;

    PortNetworkBridgeFrameIO io{};
    io.v3.abi_size = sizeof(io);
    io.v3.io_view_count = 4;
    io.v3.io_views = views;
    io.v3.out_view = output.view;
    io.v3.params = &p;
    io.images[0] = {color.image, VK_FORMAT_R16G16B16A16_SFLOAT, {W, H}, {0, 0, W, H}, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_GENERAL, family};
    io.images[1] = {motion.image, VK_FORMAT_R16G16_SFLOAT, {W, H}, {0, 0, W, H}, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_GENERAL, family};
    io.images[4] = {output.image, VK_FORMAT_R16G16B16A16_SFLOAT, {W, H}, {0, 0, W, H}, VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_LAYOUT_GENERAL, family};
    io.nrb_controls.abi_size = sizeof(NRB_Controls);
    io.nrb_controls.input_height = H;      // the gate compares this to render_height
    io.nrb_controls.enabled = 1;
    // Exactly the mapping the game uses (resolve_model_controls): a zero-filled c20 is not the
    // same thing -- ctl7 is the model's intensity, ctl9/10 are the history plane's size, and a
    // network fed "intensity 0, history 0x0" publishes its input back unchanged.
    for (int i = 0; i < 20; ++i) io.model_controls[i] = 0.0f;
    io.model_controls[1] = 1.0f;   // local tone strength
    io.model_controls[2] = 1.0f;   // structure (auto mask on)
    io.model_controls[3] = 1.0f;   // skin structure
    io.model_controls[4] = 1.0f;   // non-skin structure
    io.model_controls[5] = 1.0f;   // image noise
    io.model_controls[6] = 1.0f;   // image const
    io.model_controls[7] = 1.0f;   // intensity
    io.model_controls[8] = 1.0f;   // temporal on
    io.model_controls[9] = float(W); io.model_controls[10] = float(H);   // history plane size
    io.model_controls[13] = 1.0f; io.model_controls[14] = 1.0f;          // motion scale / frame
    io.model_controls[15] = io.model_controls[16] = 1.0f;
    io.model_controls[19] = 1.0f;  // history strength
    // NRVK_C20="i:v,i:v,..." overrides single controls, for bisecting which one the model
    // cannot live with. Comma separated, index:value.
    if (const char* c20 = std::getenv("NRVK_C20")) {
        std::string spec(c20), item;
        size_t pos = 0;
        while (pos <= spec.size()) {
            size_t comma = spec.find(',', pos);
            item = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            if (const auto colon = item.find(':'); colon != std::string::npos) {
                const int idx = std::atoi(item.substr(0, colon).c_str());
                const float val = float(std::atof(item.substr(colon + 1).c_str()));
                if (idx >= 0 && idx < 20) io.model_controls[idx] = val;
            }
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }
    io.frame_id = 1;
    io.control_epoch = 1;

    for (uint32_t f = 0; f < frames; ++f) {
        io.frame_id = f + 1;
        io.nrb_controls.history = (f == 0) ? NRB_HISTORY_REFERENCE : NRB_HISTORY_REFERENCE;
        VC(vkResetCommandBuffer(cb, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VC(vkBeginCommandBuffer(cb, &bi));
        const int rc = nrvk_backend_frame(backend, cb, &io.v3);
        if (rc != 0) { std::printf("FAIL nrvk_backend_frame rc=%d: %s\n", rc, port_network_bridge_backend_last_error()); return 4; }
        VC(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        // One fence per frame, never reset: the backend keeps the handle of the frame it
        // submitted and asks for its status at the top of the next one. Resetting it here
        // made frame 2 fail with rc=-7 ("prior network not completed") and hid exactly the
        // multi-frame path (history) that a single frame cannot exercise.
        VkFence frame_fence = VK_NULL_HANDLE;
        VkFenceCreateInfo fci2{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VC(vkCreateFence(device, &fci2, nullptr, &frame_fence));
        const VkResult sr = vkQueueSubmit(queue, 1, &si, frame_fence);
        port_network_bridge_backend_submit_result(backend, frame_fence, sr);
        if (sr != VK_SUCCESS) { std::printf("FAIL submit %d\n", int(sr)); return 4; }
        VC(vkWaitForFences(device, 1, &frame_fence, VK_TRUE, 120ull * 1000 * 1000 * 1000));
    }

    // ------------------------------------------------------- raw buffer dump
    // `--dump-buffer <plan buffer name>:<path>` copies the buffer verbatim. The scatter path
    // rounds to fp16 and clamps, which is exactly what hides a bad composite.
    if (!dump_spec.empty()) {
        const auto colon = dump_spec.find(':');
        const std::string name = dump_spec.substr(0, colon);
        const std::string path = colon == std::string::npos ? std::string() : dump_spec.substr(colon + 1);
        void* handle = nullptr; unsigned long long nbytes = 0;
        if (nrvk_backend_buffer_info(backend, name.c_str(), &handle, &nbytes) != 0 || !handle) {
            std::printf("FAIL dump: no plan buffer named %s\n", name.c_str()); return 5;
        }
        VkBufferCreateInfo dci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        dci.size = nbytes; dci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkBuffer dbuf = VK_NULL_HANDLE; VC(vkCreateBuffer(device, &dci, nullptr, &dbuf));
        VkMemoryRequirements dreq{}; vkGetBufferMemoryRequirements(device, dbuf, &dreq);
        VkPhysicalDeviceMemoryProperties dmem{}; vkGetPhysicalDeviceMemoryProperties(phy, &dmem);
        uint32_t dtype = 0;
        while (dtype < dmem.memoryTypeCount && !((dreq.memoryTypeBits & (1u << dtype)) &&
               (dmem.memoryTypes[dtype].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
               (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) ++dtype;
        VkMemoryAllocateInfo dai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        dai.allocationSize = dreq.size; dai.memoryTypeIndex = dtype;
        VkDeviceMemory dmemh = VK_NULL_HANDLE; VC(vkAllocateMemory(device, &dai, nullptr, &dmemh));
        VC(vkBindBufferMemory(device, dbuf, dmemh, 0));
        VC(vkResetCommandBuffer(cb, 0));
        VkCommandBufferBeginInfo dbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        dbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VC(vkBeginCommandBuffer(cb, &dbi));
        VkBufferCopy bc{0, 0, nbytes};
        vkCmdCopyBuffer(cb, reinterpret_cast<VkBuffer>(handle), dbuf, 1, &bc);
        VC(vkEndCommandBuffer(cb));
        VkFence dfence = VK_NULL_HANDLE; VkFenceCreateInfo dfci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VC(vkCreateFence(device, &dfci, nullptr, &dfence));
        VkSubmitInfo dsi{VK_STRUCTURE_TYPE_SUBMIT_INFO}; dsi.commandBufferCount = 1; dsi.pCommandBuffers = &cb;
        VC(vkQueueSubmit(queue, 1, &dsi, dfence));
        VC(vkWaitForFences(device, 1, &dfence, VK_TRUE, 120ull * 1000 * 1000 * 1000));
        void* mapped = nullptr; VC(vkMapMemory(device, dmemh, 0, nbytes, 0, &mapped));
        if (FILE* f = std::fopen(path.c_str(), "wb")) { std::fwrite(mapped, 1, size_t(nbytes), f); std::fclose(f); }
        vkUnmapMemory(device, dmemh);
        std::printf("[dump] %s -> %s (%llu bytes)\n", name.c_str(), path.c_str(), nbytes);
    }

    // ---------------------------------------------------------------- readback
    const VkDeviceSize out_bytes = VkDeviceSize(W) * H * 8;   // RGBA16F
    VkBufferCreateInfo rci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    rci.size = out_bytes; rci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer readback = VK_NULL_HANDLE; VC(vkCreateBuffer(device, &rci, nullptr, &readback));
    VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(device, readback, &req);
    VkPhysicalDeviceMemoryProperties mem{}; vkGetPhysicalDeviceMemoryProperties(phy, &mem);
    uint32_t type = 0;
    while (type < mem.memoryTypeCount && !((req.memoryTypeBits & (1u << type)) &&
           (mem.memoryTypes[type].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
           (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) ++type;
    VkMemoryAllocateInfo rai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; rai.allocationSize = req.size; rai.memoryTypeIndex = type;
    VkDeviceMemory rmem = VK_NULL_HANDLE; VC(vkAllocateMemory(device, &rai, nullptr, &rmem));
    VC(vkBindBufferMemory(device, readback, rmem, 0));

    VC(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo bi2{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi2.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VC(vkBeginCommandBuffer(cb, &bi2));
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(cb, output.image, VK_IMAGE_LAYOUT_GENERAL, readback, 1, &region);
    VC(vkEndCommandBuffer(cb));
    {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        VC(vkQueueSubmit(queue, 1, &si, fence));
        VC(vkWaitForFences(device, 1, &fence, VK_TRUE, 120ull * 1000 * 1000 * 1000));
    }
    void* map = nullptr; VC(vkMapMemory(device, rmem, 0, VK_WHOLE_SIZE, 0, &map));
    const uint16_t* px = static_cast<const uint16_t*>(map);

    auto half_to_float = [](uint16_t h) {
        const uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
        float v;
        if (e == 0) v = float(m) * 5.9604645e-8f;
        else if (e == 31) v = m ? NAN : INFINITY;
        else v = std::ldexp(1.0f + float(m) / 1024.0f, int(e) - 15);
        return s ? -v : v;
    };

    double sum[3] = {0, 0, 0}, mn[3] = {1e9, 1e9, 1e9}, mx[3] = {-1e9, -1e9, -1e9};
    std::vector<unsigned char> rgb(size_t(W) * H * 3);
    for (size_t i = 0; i < size_t(W) * H; ++i) {
        for (int c = 0; c < 3; ++c) {
            const float v = half_to_float(px[i * 4 + c]);
            sum[c] += v; mn[c] = std::min(mn[c], double(v)); mx[c] = std::max(mx[c], double(v));
            rgb[i * 3 + c] = static_cast<unsigned char>(std::min(1.0f, std::max(0.0f, v)) * 255.0f + 0.5f);
        }
    }
    const double n = double(W) * H;
    std::printf("out mean %.4f %.4f %.4f | min %.4f max %.4f\n", sum[0] / n, sum[1] / n, sum[2] / n, mn[0], mx[0]);

    std::string base = out_dir + "/out";
    { FILE* f = std::fopen((base + ".ppm").c_str(), "wb"); if (f) { std::fprintf(f, "P6\n%u %u\n255\n", W, H); std::fwrite(rgb.data(), 1, rgb.size(), f); std::fclose(f); } }
    { FILE* f = std::fopen((base + ".rgba16f").c_str(), "wb"); if (f) { std::fwrite(map, 1, size_t(out_bytes), f); std::fclose(f); } }
    std::printf("wrote %s.ppm + .rgba16f\n", base.c_str());

    // corner probes: where did the markers end up? (a shear or an offset shows up here)
    auto at = [&](uint32_t x, uint32_t y, int c) { return half_to_float(px[(size_t(y) * W + x) * 4 + c]); };
    for (auto [x, y, name] : {std::tuple<uint32_t, uint32_t, const char*>{16, 16, "TL"}, {W - 40, 16, "TR"}, {16, H - 40, "BL"}, {W - 40, H - 40, "BR"}}) {
        std::printf("  %s(%u,%u) rgb %.2f %.2f %.2f\n", name, x, y, at(x, y, 0), at(x, y, 1), at(x, y, 2));
    }

    vkUnmapMemory(device, rmem);
    nrvk_backend_release(backend);
    vkDestroyBuffer(device, readback, nullptr); vkFreeMemory(device, rmem, nullptr);
    vkDestroyFence(device, fence, nullptr); vkDestroyCommandPool(device, pool, nullptr);
    for (auto* im : {&color, &motion, &output}) { vkDestroyImageView(device, im->view, nullptr); vkDestroyImage(device, im->image, nullptr); vkFreeMemory(device, im->memory, nullptr); }
    vkDestroyDevice(device, nullptr); vkDestroyInstance(instance, nullptr);
    return 0;
}
