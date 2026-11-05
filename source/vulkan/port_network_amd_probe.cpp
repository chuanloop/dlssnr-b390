// Runs AMD's own upscale kernel (runtime_upscale.spv, FSR1 EASU) on a raw RGBA32F image.
//
// Why: everything the bridge does upstream of the network is our re-implementation. When the image
// still looks soft, there is no way to tell "our gather/scatter ate the detail" from "the model does
// not produce detail" without running their kernel on the same pixels. This probes exactly that:
// one image in, one image out, nothing else.
//
// Interface, read out of the SPIR-V (spirv-dis):
//   set 0 binding 0  storage image RGBA32F   input   (OpImageRead, NonWritable)
//   set 0 binding 1  combined image sampler  unused by main (debug "shown" path)
//   set 0 binding 2  storage image RGBA32F   output  (OpImageWrite, NonReadable)
//   set 0 binding 3  storage image RGBA32F   unused by main (debug "up_shown" path)
//   push constants  5 x uint32: dstWidth, dstHeight, srcWidth, srcHeight, unused
//   LocalSize 8x8, guarded by x < dstWidth && y < dstHeight
//
//     port_network_amd_probe --spv <runtime_upscale.spv> --input <raw RGBA32F> --src 1024x576
//                            --dst 1280x720 --out <dir>
//
// The output is written as out.ppm (display) and out.rgba16f (for numeric comparisons).
#define NOMINMAX
#include <windows.h>
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define VC(x) do { const VkResult r_=(x); if(r_!=VK_SUCCESS){ std::printf("FAIL %s -> %d\n",#x,int(r_)); return 2; } } while(0)

static std::vector<char> read_file(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::printf("FAIL open %s\n", path); return {}; }
    std::fseek(f, 0, SEEK_END); const long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> bytes(static_cast<size_t>(n));
    if (std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) bytes.clear();
    std::fclose(f);
    return bytes;
}

static uint32_t memory_type(VkPhysicalDevice phy, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(phy, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return 0;
}

struct Img {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t w = 0, h = 0;
};

static Img make_image(VkDevice dev, VkPhysicalDevice phy, uint32_t w, uint32_t h, VkImageUsageFlags usage) {
    Img out{}; out.w = w; out.h = h;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D; ci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    ci.extent = {w, h, 1}; ci.mipLevels = 1; ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT; ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage; ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(dev, &ci, nullptr, &out.image) != VK_SUCCESS) { std::printf("FAIL vkCreateImage\n"); return {}; }
    VkMemoryRequirements req{}; vkGetImageMemoryRequirements(dev, out.image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = memory_type(phy, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(dev, &ai, nullptr, &out.memory) != VK_SUCCESS) { std::printf("FAIL vkAllocateMemory(image)\n"); return {}; }
    vkBindImageMemory(dev, out.image, out.memory, 0);
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = out.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(dev, &vi, nullptr, &out.view) != VK_SUCCESS) { std::printf("FAIL vkCreateImageView\n"); return {}; }
    return out;
}

int main(int argc, char** argv) {
    std::string spv, input_path, out_dir;
    uint32_t sw = 0, sh = 0, dw = 0, dh = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        auto size = [&](uint32_t& w, uint32_t& h, const char* s) {
            const std::string v(s);
            if (const auto x = v.find('x'); x != std::string::npos) { w = std::atoi(v.substr(0, x).c_str()); h = std::atoi(v.substr(x + 1).c_str()); }
        };
        if (a == "--spv") spv = next();
        else if (a == "--input") input_path = next();
        else if (a == "--out") out_dir = next();
        else if (a == "--src") size(sw, sh, next());
        else if (a == "--dst") size(dw, dh, next());
        else { std::printf("unknown argument %s\n", a.c_str()); return 1; }
    }
    if (spv.empty() || input_path.empty() || out_dir.empty() || !sw || !dw) {
        std::printf("usage: --spv FILE --input raw --src WxH --dst WxH --out DIR\n");
        return 1;
    }
    CreateDirectoryA(out_dir.c_str(), nullptr);

    // ------------------------------------------------------------------ setup
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3; app.pApplicationName = "amd-easu-probe";
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
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

    uint32_t family = 0;
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(phy, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qs(qn); vkGetPhysicalDeviceQueueFamilyProperties(phy, &qn, qs.data());
    for (uint32_t i = 0; i < qn; ++i) if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { family = i; break; }

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice dev = VK_NULL_HANDLE;
    if (vkCreateDevice(phy, &dci, nullptr, &dev) != VK_SUCCESS) { std::printf("FAIL vkCreateDevice\n"); return 2; }
    VkQueue queue = VK_NULL_HANDLE; vkGetDeviceQueue(dev, family, 0, &queue);

    VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cp.queueFamilyIndex = family; cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool = VK_NULL_HANDLE; VC(vkCreateCommandPool(dev, &cp, nullptr, &pool));
    VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = pool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE; VC(vkAllocateCommandBuffers(dev, &ca, &cb));

    Img src = make_image(dev, phy, sw, sh, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    Img dst = make_image(dev, phy, dw, dh, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (!src.image || !dst.image) return 2;

    // ------------------------------------------------------------- upload src
    {
        auto bytes = read_file(input_path.c_str());
        const size_t want = size_t(sw) * sh * 4 * sizeof(float);
        if (bytes.size() < want) { std::printf("FAIL input %zu bytes, need %zu\n", bytes.size(), want); return 2; }
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = want; bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VkBuffer stage = VK_NULL_HANDLE; VC(vkCreateBuffer(dev, &bci, nullptr, &stage));
        VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(dev, stage, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memory_type(phy, req.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory mem = VK_NULL_HANDLE; VC(vkAllocateMemory(dev, &ai, nullptr, &mem));
        VC(vkBindBufferMemory(dev, stage, mem, 0));
        void* mapped = nullptr; VC(vkMapMemory(dev, mem, 0, want, 0, &mapped));
        std::memcpy(mapped, bytes.data(), want); vkUnmapMemory(dev, mem);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VC(vkBeginCommandBuffer(cb, &bi));
        VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_dst.srcQueueFamilyIndex = to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst.image = src.image; to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_dst);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {sw, sh, 1};
        vkCmdCopyBufferToImage(cb, stage, src.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        VkImageMemoryBarrier to_read{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; to_read.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_read.srcQueueFamilyIndex = to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_read.image = src.image; to_read.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        to_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_read);
        VkImageMemoryBarrier to_gen{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_gen.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; to_gen.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_gen.srcQueueFamilyIndex = to_gen.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_gen.image = dst.image; to_gen.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        to_gen.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_gen);
        VC(vkEndCommandBuffer(cb));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence = VK_NULL_HANDLE;
        VC(vkCreateFence(dev, &fci, nullptr, &fence));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        VC(vkQueueSubmit(queue, 1, &si, fence));
        VC(vkWaitForFences(dev, 1, &fence, VK_TRUE, 60ull * 1000 * 1000 * 1000));
        vkDestroyFence(dev, fence, nullptr);
        vkDestroyBuffer(dev, stage, nullptr); vkFreeMemory(dev, mem, nullptr);
    }

    // --------------------------------------------------------- pipeline (AMD)
    {
        auto bytes = read_file(spv.c_str());
        if (bytes.empty()) { std::printf("FAIL spv %s\n", spv.c_str()); return 2; }
        std::vector<uint32_t> words(bytes.size() / 4);
        std::memcpy(words.data(), bytes.data(), bytes.size());

        VkDescriptorSetLayoutBinding b[4]{};
        b[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dli.bindingCount = 4; dli.pBindings = b;
        VkDescriptorSetLayout layout = VK_NULL_HANDLE; VC(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &layout));

        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 20};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1; pli.pSetLayouts = &layout; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &range;
        VkPipelineLayout pl = VK_NULL_HANDLE; VC(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));

        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = bytes.size(); smi.pCode = words.data();
        VkShaderModule module = VK_NULL_HANDLE; VC(vkCreateShaderModule(dev, &smi, nullptr, &module));
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = module; cpi.stage.pName = "main";
        cpi.layout = pl;
        VkPipeline pipe = VK_NULL_HANDLE;
        const VkResult pr = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe);
        if (pr != VK_SUCCESS) { std::printf("FAIL vkCreateComputePipelines -> %d (shader may need features this device lacks)\n", int(pr)); return 2; }

        VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1; dpi.poolSizeCount = 2; dpi.pPoolSizes = sizes;
        VkDescriptorPool dp = VK_NULL_HANDLE; VC(vkCreateDescriptorPool(dev, &dpi, nullptr, &dp));
        VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsa.descriptorPool = dp; dsa.descriptorSetCount = 1; dsa.pSetLayouts = &layout;
        VkDescriptorSet set = VK_NULL_HANDLE; VC(vkAllocateDescriptorSets(dev, &dsa, &set));

        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR; sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VkSampler sampler = VK_NULL_HANDLE; VC(vkCreateSampler(dev, &sci, nullptr, &sampler));

        VkDescriptorImageInfo ii[4]{};
        ii[0] = {VK_NULL_HANDLE, src.view, VK_IMAGE_LAYOUT_GENERAL};
        ii[1] = {sampler, src.view, VK_IMAGE_LAYOUT_GENERAL};
        ii[2] = {VK_NULL_HANDLE, dst.view, VK_IMAGE_LAYOUT_GENERAL};
        ii[3] = {VK_NULL_HANDLE, dst.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w[4]{};
        for (int i = 0; i < 4; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = set; w[i].dstBinding = uint32_t(i); w[i].descriptorCount = 1;
            w[i].descriptorType = b[i].descriptorType; w[i].pImageInfo = &ii[i];
        }
        vkUpdateDescriptorSets(dev, 4, w, 0, nullptr);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VC(vkBeginCommandBuffer(cb, &bi));
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, nullptr);
        const uint32_t push[5] = {dw, dh, sw, sh, 0};
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
        vkCmdDispatch(cb, (dw + 7) / 8, (dh + 7) / 8, 1);
        VkImageMemoryBarrier to_copy{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_copy.oldLayout = VK_IMAGE_LAYOUT_GENERAL; to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_copy.srcQueueFamilyIndex = to_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_copy.image = dst.image; to_copy.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        to_copy.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_copy);
        VC(vkEndCommandBuffer(cb));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence = VK_NULL_HANDLE;
        VC(vkCreateFence(dev, &fci, nullptr, &fence));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        VC(vkQueueSubmit(queue, 1, &si, fence));
        VC(vkWaitForFences(dev, 1, &fence, VK_TRUE, 120ull * 1000 * 1000 * 1000));
        vkDestroyFence(dev, fence, nullptr);
    }

    // ---------------------------------------------------------------- readback
    {
        const size_t bytes = size_t(dw) * dh * 4 * sizeof(float);
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = bytes; bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkBuffer buf = VK_NULL_HANDLE; VC(vkCreateBuffer(dev, &bci, nullptr, &buf));
        VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(dev, buf, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memory_type(phy, req.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory mem = VK_NULL_HANDLE; VC(vkAllocateMemory(dev, &ai, nullptr, &mem));
        VC(vkBindBufferMemory(dev, buf, mem, 0));

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VC(vkBeginCommandBuffer(cb, &bi));
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {dw, dh, 1};
        vkCmdCopyImageToBuffer(cb, dst.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &region);
        VC(vkEndCommandBuffer(cb));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence = VK_NULL_HANDLE;
        VC(vkCreateFence(dev, &fci, nullptr, &fence));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        VC(vkQueueSubmit(queue, 1, &si, fence));
        VC(vkWaitForFences(dev, 1, &fence, VK_TRUE, 60ull * 1000 * 1000 * 1000));

        void* mapped = nullptr; VC(vkMapMemory(dev, mem, 0, bytes, 0, &mapped));
        const std::string raw = out_dir + "/out.f32";
        if (FILE* f = std::fopen(raw.c_str(), "wb")) { std::fwrite(mapped, 1, bytes, f); std::fclose(f); }

        // Display dump: clip to 0..1, no colour management (the numbers are what matter).
        const std::string ppm = out_dir + "/out.ppm";
        if (FILE* f = std::fopen(ppm.c_str(), "wb")) {
            std::fprintf(f, "P6\n%u %u\n255\n", dw, dh);
            const float* p = static_cast<const float*>(mapped);
            for (uint32_t i = 0; i < dw * dh; ++i) {
                unsigned char rgb[3];
                for (int c = 0; c < 3; ++c) {
                    float v = p[i * 4 + c];
                    v = v < 0 ? 0 : (v > 1 ? 1 : v);
                    rgb[c] = static_cast<unsigned char>(v * 255.0f + 0.5f);
                }
                std::fwrite(rgb, 1, 3, f);
            }
            std::fclose(f);
        }
        std::printf("wrote %s + out.ppm (%ux%u)\n", raw.c_str(), dw, dh);
        vkUnmapMemory(dev, mem);
    }

    vkDeviceWaitIdle(dev);
    return 0;
}
