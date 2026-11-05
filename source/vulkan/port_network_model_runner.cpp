// Actual FP16 network dispatch sequence. Cold JSON is lowered by port_network_plan.py.
#define VK_NO_PROTOTYPES
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#include <windows.h>
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define FN(f) static PFN_##f f;
FN(vkDestroyInstance) FN(vkEnumeratePhysicalDevices) FN(vkGetPhysicalDeviceProperties2)
FN(vkGetPhysicalDeviceFeatures2) FN(vkGetPhysicalDeviceMemoryProperties)
FN(vkGetPhysicalDeviceQueueFamilyProperties) FN(vkCreateDevice) FN(vkDestroyDevice)
FN(vkGetDeviceQueue) FN(vkCreateCommandPool) FN(vkDestroyCommandPool)
FN(vkAllocateCommandBuffers) FN(vkResetCommandBuffer) FN(vkBeginCommandBuffer)
FN(vkEndCommandBuffer) FN(vkCreateBuffer) FN(vkDestroyBuffer)
FN(vkGetBufferMemoryRequirements) FN(vkAllocateMemory) FN(vkFreeMemory)
FN(vkGetBufferMemoryRequirements2) FN(vkCmdFillBuffer) FN(vkCmdUpdateBuffer)
FN(vkBindBufferMemory) FN(vkMapMemory) FN(vkUnmapMemory)
FN(vkCreateDescriptorSetLayout) FN(vkDestroyDescriptorSetLayout)
FN(vkCreateDescriptorPool) FN(vkDestroyDescriptorPool) FN(vkAllocateDescriptorSets)
FN(vkUpdateDescriptorSets) FN(vkCreateShaderModule) FN(vkDestroyShaderModule)
FN(vkCreatePipelineLayout) FN(vkDestroyPipelineLayout) FN(vkCreateComputePipelines)
FN(vkDestroyPipeline) FN(vkCmdBindPipeline) FN(vkCmdBindDescriptorSets)
FN(vkCmdPushConstants) FN(vkCmdDispatch) FN(vkCmdCopyBuffer)
FN(vkCmdPipelineBarrier) FN(vkCmdWriteTimestamp2) FN(vkCmdResetQueryPool)
FN(vkCreateQueryPool) FN(vkDestroyQueryPool) FN(vkGetQueryPoolResults)
FN(vkCreateFence) FN(vkDestroyFence) FN(vkResetFences) FN(vkWaitForFences)
FN(vkQueueSubmit) FN(vkQueueWaitIdle)
#undef FN

#define CHECK(call) do { VkResult code = (call); if (code != VK_SUCCESS) \
    throw std::runtime_error(std::string(#call) + " VkResult=" + std::to_string(code)); } while (0)

struct Buffer {
    std::string name, path;
    VkDeviceSize bytes = 0, allocated = 0;
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t memoryType = 0;
    VkDeviceSize alignment=0, memoryOffset=0;
    uint32_t typeBits=0, firstPass=0, lastPass=0;
    bool dedicatedRequired=false,dedicatedPreferred=false,resident=true,clearFirst=false,ownsMemory=true;
};
struct Binding { uint32_t binding, resource; VkDeviceSize offset, bytes; };
struct Pass {
    uint32_t row, subgroup, shared, gx, gy, gz;
    std::string name, spv;
    std::vector<Binding> bindings;
    std::vector<uint32_t> push;
    size_t pipeline = 0;
    VkDescriptorSet set = VK_NULL_HANDLE;
};
struct Output { size_t resource; std::string dtype, file; };
struct History { size_t source,destination; VkDeviceSize sourceOffset,destinationOffset,bytes,controlsOffset; };
struct Plan { std::vector<Buffer> buffers; std::vector<Pass> passes; std::vector<Output> outputs; std::vector<History> history; };
struct Pipeline { VkPipeline handle; VkPipelineLayout layout; VkDescriptorSetLayout descriptors; };

#include "port_network_bridge_backend_model_identity.inc"
static std::string read_quoted_path(std::istream& in) {
    char ch=0; in>>std::ws; if(!(in>>ch)||ch!='"') throw std::runtime_error("expected quoted path");
    std::string value; while(in.get(ch)){ if(ch=='"') return value; value.push_back(ch);}
    throw std::runtime_error("unterminated quoted path");
}

static Plan read_plan(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot read network plan " + path);
    std::string word;
    size_t rows, nb, np, no;
    file >> word >> rows >> nb >> np >> no;
    if (word != "PORT_NETWORK_1" || rows != kPlanRows || !nb || !np || !no)
        throw std::runtime_error("network requires a complete plan matching the built model");
    Plan plan;
    const auto planDir = std::filesystem::absolute(path).parent_path();
    auto assetPath = [&](const std::string& value) {
        if (value == "-") return value;
        auto p = std::filesystem::path(value);
        return (p.is_absolute() ? p : planDir / p).lexically_normal().string();
    };
    for (size_t i = 0; i < nb; ++i) {
        Buffer buffer;
        file >> word;
        buffer.name = read_quoted_path(file);
        file >> buffer.bytes;
        buffer.path = assetPath(read_quoted_path(file));
        if (word != "BUFFER" || !buffer.bytes) throw std::runtime_error("invalid BUFFER");
        plan.buffers.push_back(buffer);
    }
    std::vector<bool> covered(kPlanRows, false);
    int previous = -1;
    for (size_t i = 0; i < np; ++i) {
        Pass pass;
        size_t nbind, npush;
        file >> word >> pass.row;
        pass.name = read_quoted_path(file);
        pass.spv = assetPath(read_quoted_path(file));
        file >> pass.subgroup >> pass.shared >> nbind >> npush >> pass.gx >> pass.gy >> pass.gz;
        if (word != "PASS" || pass.row >= kPlanRows || int(pass.row) < previous ||
            pass.shared > 49152 || pass.subgroup != 32 || npush > 32 || !pass.gx || !pass.gy || !pass.gz)
            throw std::runtime_error("invalid PASS or unsupported baseline resource configuration");
        previous = int(pass.row);
        covered[pass.row] = true;
        for (size_t j = 0; j < nbind; ++j) {
            Binding binding;
            file >> word >> binding.binding >> binding.resource >> binding.offset >> binding.bytes;
            if (word != "BIND" || binding.resource >= nb || !binding.bytes ||
                binding.offset + binding.bytes > plan.buffers[binding.resource].bytes)
                throw std::runtime_error("invalid BIND byte range");
            pass.bindings.push_back(binding);
        }
        file >> word;
        if (word != "PUSH") throw std::runtime_error("missing PUSH");
        for (size_t j = 0; j < npush; ++j) { uint32_t value; file >> value; pass.push.push_back(value); }
        plan.passes.push_back(pass);
    }
    for (bool present : covered) if (!present) throw std::runtime_error("uncovered native plan row");
    for (size_t i = 0; i < no; ++i) {
        Output output;
        file >> word >> output.resource >> output.dtype;
        output.file = read_quoted_path(file);
        if (word != "OUTPUT" || output.resource >= nb) throw std::runtime_error("invalid OUTPUT");
        plan.outputs.push_back(output);
    }
    if (!file) throw std::runtime_error("truncated prepared network plan");
    while(file>>word) {
        if(word=="POLICY") {
            size_t resource; int resident,clear;
            file>>resource>>resident;
            auto& b=plan.buffers.at(resource);
            file>>b.firstPass>>b.lastPass>>clear;
            b.resident=resident!=0;b.clearFirst=clear!=0;
        } else if(word=="HISTORY") {
            History h;
            file>>h.source>>h.destination>>h.sourceOffset>>h.destinationOffset>>h.bytes>>h.controlsOffset;
            plan.history.push_back(h);
        } else throw std::runtime_error("unknown plan policy "+word);
    }
    return plan;
}

static std::vector<char> read_bytes(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read " + path);
    return std::vector<char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

static double host_ms(LARGE_INTEGER a, LARGE_INTEGER b, LARGE_INTEGER frequency) {
    return 1000.0 * double(b.QuadPart - a.QuadPart) / double(frequency.QuadPart);
}

// The frame extent lives in the plan's output port (its push carries the frame size; see
// tools/gen_plan.py FRAME_DIM_WORDS). Nothing else in the plan states it.
//
// Recognise the port by role, not by block number: the full layout calls it
// `ioport_block70_2`, the small profile `r75_ioport_block30_2`. Matching the literal name
// silently returned 0x0 for the small plan, which fed `resolve_controls` a zero extent and
// would have written a picture with the wrong motion/tone normalisation while still
// reporting a plausible gpu_ms.
static bool is_output_port(const std::string& name) {
    return name.find("ioport_block") != std::string::npos &&
           name.size() > 2 && name.compare(name.size() - 2, 2, "_2") == 0;
}

static void plan_frame_extent(const Plan& plan, uint32_t* w, uint32_t* h) {
    for(const auto& pass:plan.passes) {
        if(!is_output_port(pass.name)||pass.push.size()<4) continue;
        *w=pass.push[2]; *h=pass.push[3]; return;
    }
    *w=*h=0;
}

// The twenty floats the model reads through ctl(), resolved exactly as AMD's own driver does
// (20-AMD/dlssnr-src/DLSSNR-RDNA3/windows/src/core/nr_graph.cpp:2364-2374), expression for
// expression, with AMD's CLI defaults. Writing them is what makes an offline run say something
// about the picture: the network's tone and strength controls are ctl(0..19), so leaving them at
// the cold buffer's zeros makes the port emit black and every plan looks equally "fine, finite,
// N milliseconds".
//
// This used to hardcode the mask-off branch (c[3]=c[4]=-1) while the reference ran with the mask
// on, which cost a measured 43.78 dB of configuration difference and put a per-channel DC into
// every per-layer comparison (07-进展.md, M1-T34/T36). Keep it parameterised and print the
// resolved five like AMD does, so the configuration lands in the log next to the score.
struct ModelControls {
    int style = 0;              // --style, AMD clamps to 0..2
    float tone = 1.0f;          // --local-tone-strength
    float structure = 1.0f;     // --local-structure-strength
    float skin = -1.0f;         // --skin-structure-strength, -1 = fall back to structure
    bool mask = true;           // --auto-mask 1
};

static void resolve_controls(float c[20], uint32_t w, uint32_t h, float reset, float have_depth,
                             float mvx, float mvy, float inverted, const ModelControls& mc) {
    const float structure = mc.structure;
    const float skin = mc.skin;
    const float mask_skin  = mc.mask ? (skin < 0 ? structure : skin) : -1.0f;
    const float mask_other = mc.mask ? structure : -1.0f;
    const bool  any_mask   = std::max(mask_skin, mask_other) >= 0.0f;
    c[0] = float(std::clamp(mc.style, 0, 2)) / 128.0f;                   // s434
    c[1] = mc.tone;                                                      // aux0
    c[2] = any_mask ? 1.0f : structure;                                  // aux1
    c[3] = any_mask ? (mask_skin < 0 ? structure : mask_skin) : -1.0f;    // gate0
    c[4] = any_mask ? (mask_other < 0 ? structure : mask_other) : -1.0f;  // gate1
    c[5]=1.0f;
    c[6]=1.0f;
    c[7]=1.0f;          // model intensity
    c[8]=reset;         // history enable
    c[9]=float(w);
    c[10]=float(h);
    c[11]=have_depth;
    c[12]=inverted;
    c[13]=mvx/float(w?w:1u);
    c[14]=mvy/float(h?h:1u);
    c[15]=1.0f;
    c[16]=1.0f;
    c[17]=0.0f;
    c[18]=0.0f;
    c[19]=1.0f;
}

static int execute(Plan& plan, const std::string& out, int warm, int reps, bool pooled, bool feedback, bool dump, bool saveFrames, bool perPassTiming, int patternW, int patternH, int inputW=0, int inputH=0, const std::string& inputPath=std::string(), float planeMvX=0.0f, float planeMvY=0.0f, int planeDepth=0, const ModelControls& mc=ModelControls()) {
    if (!std::filesystem::create_directory(out)) throw std::runtime_error("output directory must be fresh: " + out);
    HMODULE dll = LoadLibraryA("vulkan-1.dll");
    if (!dll) throw std::runtime_error("vulkan-1.dll missing");
    auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(dll, "vkGetInstanceProcAddr"));
    auto create = reinterpret_cast<PFN_vkCreateInstance>(gipa(VK_NULL_HANDLE, "vkCreateInstance"));
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "port_network_fp16"; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    VkInstance instance;
    CHECK(create(&ici, nullptr, &instance));
#define LOAD(f) f = reinterpret_cast<PFN_##f>(gipa(instance, #f)); if (!f) throw std::runtime_error("missing " #f);
    LOAD(vkDestroyInstance) LOAD(vkEnumeratePhysicalDevices) LOAD(vkGetPhysicalDeviceProperties2)
    LOAD(vkGetPhysicalDeviceFeatures2) LOAD(vkGetPhysicalDeviceMemoryProperties)
    LOAD(vkGetPhysicalDeviceQueueFamilyProperties) LOAD(vkCreateDevice) LOAD(vkDestroyDevice)
    LOAD(vkGetDeviceQueue) LOAD(vkCreateCommandPool) LOAD(vkDestroyCommandPool)
    LOAD(vkAllocateCommandBuffers) LOAD(vkResetCommandBuffer) LOAD(vkBeginCommandBuffer)
    LOAD(vkEndCommandBuffer) LOAD(vkCreateBuffer) LOAD(vkDestroyBuffer)
    LOAD(vkGetBufferMemoryRequirements) LOAD(vkAllocateMemory) LOAD(vkFreeMemory)
    LOAD(vkGetBufferMemoryRequirements2) LOAD(vkCmdFillBuffer) LOAD(vkCmdUpdateBuffer)
    LOAD(vkBindBufferMemory) LOAD(vkMapMemory) LOAD(vkUnmapMemory)
    LOAD(vkCreateDescriptorSetLayout) LOAD(vkDestroyDescriptorSetLayout)
    LOAD(vkCreateDescriptorPool) LOAD(vkDestroyDescriptorPool) LOAD(vkAllocateDescriptorSets)
    LOAD(vkUpdateDescriptorSets) LOAD(vkCreateShaderModule) LOAD(vkDestroyShaderModule)
    LOAD(vkCreatePipelineLayout) LOAD(vkDestroyPipelineLayout) LOAD(vkCreateComputePipelines)
    LOAD(vkDestroyPipeline) LOAD(vkCmdBindPipeline) LOAD(vkCmdBindDescriptorSets)
    LOAD(vkCmdPushConstants) LOAD(vkCmdDispatch) LOAD(vkCmdCopyBuffer)
    LOAD(vkCmdPipelineBarrier) LOAD(vkCmdWriteTimestamp2) LOAD(vkCmdResetQueryPool)
    LOAD(vkCreateQueryPool) LOAD(vkDestroyQueryPool) LOAD(vkGetQueryPoolResults)
    LOAD(vkCreateFence) LOAD(vkDestroyFence) LOAD(vkResetFences) LOAD(vkWaitForFences)
    LOAD(vkQueueSubmit) LOAD(vkQueueWaitIdle)
#undef LOAD
    uint32_t nd = 0;
    CHECK(vkEnumeratePhysicalDevices(instance, &nd, nullptr));
    std::vector<VkPhysicalDevice> devices(nd);
    CHECK(vkEnumeratePhysicalDevices(instance, &nd, devices.data()));
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    for (auto device : devices) {
        VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        vkGetPhysicalDeviceProperties2(device, &props);
        // B580 (0xe20b, Xe2 dGPU) verified; Arc B390 ships 0xb080 (Xe3 iGPU),
        // adapted — same Intel driver family, same 8x16x16 fp16 coopmat recipe.
        if (props.properties.vendorID == 0x8086 &&
            (props.properties.deviceID == 0xe20b || props.properties.deviceID == 0xb080)) {
            physical = device; properties = props.properties;
        }
    }
    if (!physical) throw std::runtime_error("Intel Arc B580/B390 missing");
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceShaderFloat16Int8Features half{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDevice16BitStorageFeatures storage{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    VkPhysicalDeviceSynchronization2Features sync{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
    VkPhysicalDeviceVulkanMemoryModelFeatures model{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES};
    // A binding that appears more than once in a PASS is a descriptor array (plan-level
    // way to fuse several dispatches into one), so the shader indexes it at run time.
    VkPhysicalDeviceDescriptorIndexingFeatures index{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext=&cm; cm.pNext=&half; half.pNext=&storage; storage.pNext=&subgroup; subgroup.pNext=&sync; sync.pNext=&model; model.pNext=&index;
    vkGetPhysicalDeviceFeatures2(physical, &features);
    if (!cm.cooperativeMatrix || !half.shaderFloat16 || !storage.storageBuffer16BitAccess ||
        !subgroup.subgroupSizeControl || !subgroup.computeFullSubgroups || !sync.synchronization2 || !model.vulkanMemoryModel)
        throw std::runtime_error("required FP16/coop/subgroup/synchronization2/VulkanMemoryModel features unavailable");
    // Enable only the features consumed by the actual FP16 SPVs and timestamp calls.
    model.vulkanMemoryModelDeviceScope=VK_FALSE;
    model.vulkanMemoryModelAvailabilityVisibilityChains=VK_FALSE;
    cm.cooperativeMatrixRobustBufferAccess=VK_FALSE;
    half.shaderInt8=VK_FALSE;
    index.shaderStorageBufferArrayNonUniformIndexing = index.shaderStorageBufferArrayNonUniformIndexing ? VK_TRUE : VK_FALSE;
    index.runtimeDescriptorArray = index.runtimeDescriptorArray ? VK_TRUE : VK_FALSE;
    storage.uniformAndStorageBuffer16BitAccess=VK_FALSE;
    storage.storagePushConstant16=VK_FALSE; storage.storageInputOutput16=VK_FALSE;
    uint32_t nq=0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,nullptr);
    std::vector<VkQueueFamilyProperties> qprops(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,qprops.data());
    uint32_t family=0;
    while (family<nq && !(qprops[family].queueFlags&VK_QUEUE_COMPUTE_BIT)) ++family;
    if (family==nq || !qprops[family].timestampValidBits) throw std::runtime_error("compute timestamps unavailable");
    float priority=1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex=family; qci.queueCount=1; qci.pQueuePriorities=&priority;
    const char* ext="VK_KHR_cooperative_matrix";
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext=&cm; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
    dci.enabledExtensionCount=1; dci.ppEnabledExtensionNames=&ext;
    VkDevice device;
    CHECK(vkCreateDevice(physical,&dci,nullptr,&device));
    VkQueue queue;
    vkGetDeviceQueue(device,family,0,&queue);
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex=family; cpci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool;
    CHECK(vkCreateCommandPool(device,&cpci,nullptr,&pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool=pool; cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount=1;
    VkCommandBuffer command;
    CHECK(vkAllocateCommandBuffers(device,&cai,&command));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CHECK(vkCreateFence(device,&fci,nullptr,&fence));
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(physical,&memory);
    auto make_buffer=[&](Buffer& buffer,VkBufferUsageFlags usage,VkMemoryPropertyFlags flags) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size=buffer.bytes; bci.usage=usage;
        CHECK(vkCreateBuffer(device,&bci,nullptr,&buffer.handle));
        VkMemoryRequirements need;
        VkMemoryDedicatedRequirements dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
        VkMemoryRequirements2 requirement{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2}; requirement.pNext=&dedicated;
        VkBufferMemoryRequirementsInfo2 query{VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2};query.buffer=buffer.handle;
        vkGetBufferMemoryRequirements2(device,&query,&requirement);need=requirement.memoryRequirements;
        uint32_t type=0;
        while(type<memory.memoryTypeCount && !((need.memoryTypeBits&(1u<<type)) &&
              (memory.memoryTypes[type].propertyFlags&flags)==flags)) ++type;
        if(type==memory.memoryTypeCount) throw std::runtime_error("required memory type unavailable: "+buffer.name);
        buffer.memoryType=type;buffer.allocated=need.size;buffer.alignment=need.alignment;buffer.typeBits=need.memoryTypeBits;
        buffer.dedicatedRequired=dedicated.requiresDedicatedAllocation;buffer.dedicatedPreferred=dedicated.prefersDedicatedAllocation;
        if(!pooled || (flags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) || buffer.dedicatedRequired) {
            VkMemoryDedicatedAllocateInfo di{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};di.buffer=buffer.handle;
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize=need.size;mai.memoryTypeIndex=type;mai.pNext=buffer.dedicatedRequired?&di:nullptr;
            CHECK(vkAllocateMemory(device,&mai,nullptr,&buffer.memory));
            CHECK(vkBindBufferMemory(device,buffer.handle,buffer.memory,0));
        }
    };
    VkDeviceSize stage_size=0;
    if(dump) {
        for(size_t i=0;i<plan.buffers.size();++i)if(!plan.buffers[i].resident) {
            plan.outputs.push_back({i,"raw","dump_"+plan.buffers[i].name+".bin"});
            plan.buffers[i].resident=true; // the end-of-network readback is a real last consumer
        }
        // Layout index: every intermediate's extent is read back here instead of inferred later.
        // Guessing (w,h,ch) from a byte count is ambiguous, and a wrong guess silently turns any
        // frequency measurement into noise -- so record the dispatch grid and the push block of the
        // first pass that writes each buffer, plus exact byte counts.
        std::ofstream index(std::filesystem::path(out)/"index.txt");
        index<<"# written by --dump-intermediates; one PASS line per dispatch, one BUFFER line per resource\n";
        for(const auto& pass:plan.passes) {
            index<<"PASS \""<<pass.name<<"\" "<<pass.gx<<" "<<pass.gy<<" "<<pass.gz<<" push";
            for(uint32_t word:pass.push) index<<" "<<word;
            index<<" bind";
            for(const auto& b:pass.bindings) index<<" "<<b.binding<<"="<<b.resource<<"@"<<b.bytes;
            index<<"\n";
        }
        for(size_t i=0;i<plan.buffers.size();++i)
            index<<"BUFFER "<<i<<" \""<<plan.buffers[i].name<<"\" "<<plan.buffers[i].bytes<<"\n";
    }
    for(auto& buffer:plan.buffers) {
        make_buffer(buffer,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        stage_size=std::max(stage_size,buffer.bytes);
    }
    std::vector<VkDeviceMemory> poolMemory;
    VkDeviceSize poolAllocated=0;
    if(pooled) {
        // One resident and one transient pool per observed compatible type.
        // Driver alignment/size are used now; no inference from size rounding.
        struct Slot {VkDeviceSize offset,size;uint32_t last;};
        std::map<std::pair<uint32_t,bool>,std::vector<size_t>> groups;
        for(size_t i=0;i<plan.buffers.size();++i) {
            auto& b=plan.buffers[i];
            if(!b.memory) groups[{b.memoryType,b.resident}].push_back(i);
        }
        for(auto& group:groups) {
            auto ids=group.second;
            std::sort(ids.begin(),ids.end(),[&](size_t a,size_t b){return plan.buffers[a].firstPass<plan.buffers[b].firstPass;});
            VkDeviceSize size=0;std::vector<Slot> slots;
            for(size_t id:ids) {
                auto& b=plan.buffers[id];Slot* selected=nullptr;
                if(!b.resident)for(auto& s:slots)if(s.last<b.firstPass && s.size>=b.allocated && s.offset%b.alignment==0)
                    if(!selected || s.size<selected->size)selected=&s;
                if(selected) {b.memoryOffset=selected->offset;selected->last=b.lastPass;}
                else {
                    b.memoryOffset=(size+b.alignment-1)/b.alignment*b.alignment;
                    slots.push_back({b.memoryOffset,b.allocated,b.lastPass});size=b.memoryOffset+b.allocated;
                }
                b.ownsMemory=false;
            }
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize=size;allocation.memoryTypeIndex=group.first.first;
            VkDeviceMemory poolBlock;CHECK(vkAllocateMemory(device,&allocation,nullptr,&poolBlock));
            poolMemory.push_back(poolBlock);poolAllocated+=size;
            for(size_t id:ids) {auto& b=plan.buffers[id];b.memory=poolBlock;CHECK(vkBindBufferMemory(device,b.handle,poolBlock,b.memoryOffset));}
        }
    }
    Buffer stage;
    stage.name="cold_staging"; stage.bytes=stage_size;
    make_buffer(stage,VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* mapped=nullptr;
    CHECK(vkMapMemory(device,stage.memory,0,VK_WHOLE_SIZE,0,&mapped));
    auto begin=[&]() {
        CHECK(vkResetCommandBuffer(command,0));
        VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        info.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK(vkBeginCommandBuffer(command,&info));
    };
    auto submit=[&]() {
        CHECK(vkEndCommandBuffer(command)); CHECK(vkResetFences(device,1,&fence));
        VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        info.commandBufferCount=1; info.pCommandBuffers=&command;
        CHECK(vkQueueSubmit(queue,1,&info,fence));
        CHECK(vkWaitForFences(device,1,&fence,VK_TRUE,60ULL*1000*1000*1000));
    };
    auto barrier=[&](VkPipelineStageFlags src,VkPipelineStageFlags dst,VkAccessFlags a,VkAccessFlags b) {
        VkMemoryBarrier dependency{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        dependency.srcAccessMask=a; dependency.dstAccessMask=b;
        vkCmdPipelineBarrier(command,src,dst,0,1,&dependency,0,nullptr,0,nullptr);
    };
    for(auto& buffer:plan.buffers) {
        if(pooled && !buffer.resident && !buffer.dedicatedRequired) continue; // initialisation at first real use
        // Any `*_ioport_images` surface, not the two literal names: the port that composites
        // onto the frame is `ioport_70_ioport_images` in the full layout and
        // `ioport_30_ioport_images` in the small one. Matching the literals left the small
        // profile's base surface zero, so every small run emitted a bare residual
        // (verified: output == clamp(body*0.25) to the last bit, correlation 1.0) and looked
        // black while the timing was perfectly plausible.
        auto ends_with=[](const std::string& s,const std::string& tail){
            return s.size()>tail.size()&&s.compare(s.size()-tail.size(),tail.size(),tail)==0;
        };
        if(buffer.path=="-" && (patternW>0||inputW>0) && ends_with(buffer.name,"_ioport_images")) {
            // Deterministic structured frame at the requested size, in the colour
            // plane the first pass reads (RGBA32F at float offset 0). Coordinates
            // are normalised, so the same pattern at another size is the same
            // picture at another resolution - the basis for comparing plans.
            //
            // Both ends get it. `ioport_0` is what the first pass reads; `ioport_70` is the
            // surface the output port composites onto, and in a game the host's gather fills
            // it (`kIoport*FloatOffset`). Leaving the second one zero made every offline run
            // emit a bare residual with no image behind it, which is why "non-black / row
            // noise" could never say whether a plan was right: there was no picture to be
            // right about.
            memset(mapped,0,size_t(buffer.bytes));
            auto* px=static_cast<float*>(mapped);
            if(inputW>0) {
                // Planar RGBA f32 in, interleaved RGBA f32 out - the layout the ioport plane reads.
                const size_t n=size_t(inputW)*size_t(inputH);
                std::ifstream in(inputPath,std::ios::binary);
                if(!in) throw std::runtime_error("cannot read --input "+inputPath);
                std::vector<float> src(n*4);
                in.read(reinterpret_cast<char*>(src.data()),std::streamsize(src.size()*sizeof(float)));
                if(size_t(in.gcount())!=src.size()*sizeof(float))
                    throw std::runtime_error("--input is not 4 planes of f32 at the stated size");
                const int w=std::min(inputW,int(buffer.bytes/(4*2*inputH)));
                for(size_t y=0;y<size_t(inputH);++y)
                    for(int x=0;x<w;++x) {
                        float* o=px+(y*size_t(w)+size_t(x))*4;
                        for(int c=0;c<4;++c) o[c]=src[c*n+y*size_t(inputW)+size_t(x)];
                    }
                // The guide planes the host's gather fills in game: motion at 8*px+20 floats
                // (2 per pixel) and depth at 10*px+20 (1 per pixel). Offline they default to
                // zero, which makes every measurement a *spatial-only* network: zero motion
                // disables the temporal branch and zero depth every occlusion test. That is
                // why offline runs look like a plain low-pass while the game does more.
                const size_t pixels=size_t(inputW)*size_t(inputH);
                if((planeMvX!=0.0f||planeMvY!=0.0f||planeDepth!=0) &&
                   buffer.bytes>= (11*pixels+20)*sizeof(float)) {
                    float* mo=px+(8*pixels+20);
                    for(size_t k=0;k<pixels;++k) { mo[k*2]=planeMvX; mo[k*2+1]=planeMvY; }
                    float* dp=px+(10*pixels+20);
                    for(size_t y=0;y<size_t(inputH);++y)
                        for(int x=0;x<inputW;++x) {
                            const float t=float(y)/float(inputH>1?inputH-1:1);
                            dp[y*size_t(inputW)+size_t(x)] =
                                planeDepth==1?t: planeDepth==2?1.0f-t: planeDepth==3?0.0f: 1.0f;
                        }
                }
            } else {
            const size_t count=size_t(patternW)*size_t(patternH);
            for(size_t y=0;y<size_t(patternH);++y)
                for(size_t x=0;x<size_t(patternW);++x) {
                    const float nx=float(x)/float(patternW), ny=float(y)/float(patternH);
                    const float checker=((int(nx*8.0f)+int(ny*8.0f))&1)?1.0f:0.15f;
                    float* o=px+(y*size_t(patternW)+x)*4;
                    o[0]=nx; o[1]=ny; o[2]=checker; o[3]=1.0f;
                }
            (void)count;
            }
        } else if(buffer.path=="-") memset(mapped,0,size_t(buffer.bytes));
        else {
            auto bytes=read_bytes(buffer.path);
            if(bytes.size()!=buffer.bytes) throw std::runtime_error("cold input size mismatch: "+buffer.name);
            memcpy(mapped,bytes.data(),bytes.size());
        }
        begin(); VkBufferCopy copy{0,0,buffer.bytes};
        vkCmdCopyBuffer(command,stage.handle,buffer.handle,1,&copy);
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
        submit();
    }
    std::map<std::string,size_t> cache;
    std::vector<Pipeline> pipelines;
    uint32_t descriptor_count=0;
    for(auto& pass:plan.passes) {
        std::string key=pass.spv+":"+std::to_string(pass.push.size())+":"+std::to_string(pass.subgroup);
        for(auto& b:pass.bindings) key+=":"+std::to_string(b.binding);
        descriptor_count+=uint32_t(pass.bindings.size());
        if(cache.count(key)) { pass.pipeline=cache[key]; continue; }
        // Repeated binding indices collapse into one descriptor array; the order of the
        // repeats is the array index the shader sees.
        std::vector<std::pair<uint32_t,uint32_t>> slots;   // (binding, descriptorCount)
        for(auto& b:pass.bindings) {
            if(!slots.empty() && slots.back().first==b.binding) ++slots.back().second;
            else slots.push_back({b.binding,1});
        }
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for(auto& sl:slots)
            bindings.push_back({sl.first,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,sl.second,VK_SHADER_STAGE_COMPUTE_BIT,nullptr});
        VkDescriptorSetLayoutCreateInfo dsci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dsci.bindingCount=uint32_t(bindings.size()); dsci.pBindings=bindings.data();
        Pipeline pipeline{};
        CHECK(vkCreateDescriptorSetLayout(device,&dsci,nullptr,&pipeline.descriptors));
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,uint32_t(pass.push.size()*4)};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount=1; plci.pSetLayouts=&pipeline.descriptors;
        plci.pushConstantRangeCount=pass.push.empty()?0:1; plci.pPushConstantRanges=&push;
        CHECK(vkCreatePipelineLayout(device,&plci,nullptr,&pipeline.layout));
        auto bytes=read_bytes(pass.spv);
        std::vector<uint32_t> code(bytes.size()/4);
        memcpy(code.data(),bytes.data(),bytes.size());
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize=bytes.size(); smci.pCode=code.data();
        VkShaderModule module;
        CHECK(vkCreateShaderModule(device,&smci,nullptr,&module));
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo size{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        size.requiredSubgroupSize=pass.subgroup;
        VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pci.stage.pNext=&size; pci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;
        pci.stage.module=module; pci.stage.pName="main"; pci.layout=pipeline.layout;
        CHECK(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&pci,nullptr,&pipeline.handle));
        vkDestroyShaderModule(device,module,nullptr);
        pass.pipeline=pipelines.size(); cache[key]=pass.pipeline; pipelines.push_back(pipeline);
    }
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,descriptor_count};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets=uint32_t(plan.passes.size()); dpci.poolSizeCount=1; dpci.pPoolSizes=&pool_size;
    VkDescriptorPool descriptors;
    CHECK(vkCreateDescriptorPool(device,&dpci,nullptr,&descriptors));
    for(auto& pass:plan.passes) {
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool=descriptors; dai.descriptorSetCount=1;
        dai.pSetLayouts=&pipelines[pass.pipeline].descriptors;
        CHECK(vkAllocateDescriptorSets(device,&dai,&pass.set));
        // One write per distinct binding; a repeated binding writes the whole array at once.
        std::vector<std::vector<VkDescriptorBufferInfo>> infos;
        std::vector<VkWriteDescriptorSet> writes;
        for(size_t i=0;i<pass.bindings.size();) {
            const uint32_t binding=pass.bindings[i].binding;
            std::vector<VkDescriptorBufferInfo> one;
            size_t j=i;
            while(j<pass.bindings.size() && pass.bindings[j].binding==binding) {
                const auto& b=pass.bindings[j];
                one.push_back({plan.buffers[b.resource].handle,b.offset,b.bytes});
                ++j;
            }
            infos.push_back(std::move(one));
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet=pass.set; w.dstBinding=binding;
            w.descriptorCount=uint32_t(infos.back().size());
            w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo=infos.back().data();
            writes.push_back(w);
            i=j;
        }
        vkUpdateDescriptorSets(device,uint32_t(writes.size()),writes.data(),0,nullptr);
    }
    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType=VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount=2+(perPassTiming?uint32_t(plan.passes.size()+1):0);
    VkQueryPool queries;
    CHECK(vkCreateQueryPool(device,&qpci,nullptr,&queries));
    auto copy_output=[&](const Output& output,const std::string& fileName) {
        const auto& b=plan.buffers[output.resource];
        begin();
        barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferCopy copy{0,0,b.bytes};
        vkCmdCopyBuffer(command,b.handle,stage.handle,1,&copy);
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);
        submit();
        std::ofstream file(std::filesystem::path(out)/fileName,std::ios::binary);
        file.write(static_cast<const char*>(mapped),std::streamsize(b.bytes));
        if(!file)throw std::runtime_error("frame readback write failed");
    };
    uint32_t planW=0,planH=0;
    plan_frame_extent(plan,&planW,&planH);
    if(!planW||!planH){planW=uint32_t(patternW>0?patternW:0);planH=uint32_t(patternH>0?patternH:0);}
    std::cout<<"plan frame "<<planW<<"x"<<planH<<"\n";
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    std::vector<double> gpu_times,host_times;
    std::vector<std::vector<double>> pass_times(perPassTiming?plan.passes.size():0);
    for(int rep=-warm;rep<reps;++rep) {
        const uint32_t frame=uint32_t(rep+warm);
        begin(); vkCmdResetQueryPool(command,queries,0,qpci.queryCount);
        vkCmdWriteTimestamp2(command,VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,queries,0);
        barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
        {
            float controls[20];
            resolve_controls(controls,planW,planH,(feedback&&frame>0)?1.0f:0.0f,0.0f,1.0f,1.0f,0.0f,mc);
            // NRVK_CTL=<index>:<value> overwrites one control word, so each slot can be swept to
            // find which one this network actually listens to. The backend's own override never
            // reaches the harness: this executable does not link port_backend at all.
            if(const char* ctl=std::getenv("NRVK_CTL")){
                std::string list(ctl);
                size_t at=0;
                while(at<list.size()) {
                    size_t comma=list.find(',',at);
                    if(comma==std::string::npos)comma=list.size();
                    int idx=-1; float val=0.0f;
                    if(std::sscanf(list.substr(at,comma-at).c_str(),"%d:%f",&idx,&val)==2&&idx>=0&&idx<20)
                        controls[idx]=val;
                    at=comma+1;
                }
            }
            barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT);
            if(feedback && frame>0) {
                if(plan.history.size()!=2)throw std::runtime_error("feedback requires actual pre/post history connections");
                for(auto& h:plan.history) {
                    VkBufferCopy copy{h.sourceOffset,h.destinationOffset,h.bytes};
                    vkCmdCopyBuffer(command,plan.buffers[h.source].handle,plan.buffers[h.destination].handle,1,&copy);
                }
            }
            for(auto& h:plan.history)
                vkCmdUpdateBuffer(command,plan.buffers[h.destination].handle,h.controlsOffset,sizeof(controls),controls);
            barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
        }
        if(perPassTiming)vkCmdWriteTimestamp2(command,VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,queries,2);
        static const bool passBarrier=[](){
            const char* env=std::getenv("NRVK_NOPASS_BARRIER");
            return !(env&&env[0]=='1');
        }();
        // Probe: NRVK_DUP_PASS="<name-substring>:<count>" dispatches the matching pass
        // `count` times, so frame time against count has the all-in cost of one
        // dispatch as its slope (work + submit + barrier).  A plan-level probe cannot
        // do this: the runner rejects any plan that does not match the built model.
        // The pass is idempotent (same bindings, same output), so the extra dispatches
        // only cost time.
        static const std::string dupSub=[](){
            const char* env=std::getenv("NRVK_DUP_PASS"); std::string s(env?env:"");
            return s.substr(0,s.find(':'));
        }();
        static const int dupCount=[](){
            const char* env=std::getenv("NRVK_DUP_PASS"); if(!env)return 1;
            std::string s(env); size_t c=s.find(':');
            if(c==std::string::npos)return 1;
            int value=std::atoi(s.c_str()+c+1);
            return value>1?value:1;
        }();
        auto dupRepeats=[&](const std::string& passName){
            return (dupCount>1&&!dupSub.empty()&&passName.find(dupSub)!=std::string::npos)?dupCount:1;
        };
        for(size_t passIndex=0;passIndex<plan.passes.size();++passIndex) {
            auto& pass=plan.passes[passIndex];
            if(pooled) {
                bool clearing=false;
                for(auto& buffer:plan.buffers)if(!buffer.resident && buffer.firstPass==passIndex && buffer.clearFirst) {
                    if(!clearing)barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);
                    clearing=true;
                    vkCmdFillBuffer(command,buffer.handle,0,VK_WHOLE_SIZE,0);
                }
                if(clearing)barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
            }
            const auto& pipeline=pipelines[pass.pipeline];
            vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline.handle);
            vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline.layout,0,1,&pass.set,0,nullptr);
            auto push=pass.push;
            if(passIndex==0) {push[22]=feedback?frame:0;push[23]&=~4u;}
            if(!push.empty()) vkCmdPushConstants(command,pipeline.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,
                uint32_t(push.size()*4),push.data());
            for(int dup=0,reps=dupRepeats(pass.name);dup<reps;++dup)
                vkCmdDispatch(command,pass.gx,pass.gy,pass.gz);
            // A full global barrier after every pass: correct, but blunt -- it also
            // orders passes that touch disjoint buffers.  NRVK_NOPASS_BARRIER=1 drops
            // it so the frame-time difference is the *total* cost of all 108 barriers,
            // i.e. an upper bound on what dependency-driven barriers could win back.
            // Timing probe only: results under it are racy, never gate quality on it.
            if(passBarrier)
                barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
            if(perPassTiming)vkCmdWriteTimestamp2(command,VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,queries,uint32_t(passIndex+3));
        }
        vkCmdWriteTimestamp2(command,VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,queries,1);
        LARGE_INTEGER a,b;
        QueryPerformanceCounter(&a); submit(); QueryPerformanceCounter(&b);
        if(rep<0) continue;
        std::vector<uint64_t> timestamps(qpci.queryCount);
        CHECK(vkGetQueryPoolResults(device,queries,0,qpci.queryCount,timestamps.size()*sizeof(uint64_t),timestamps.data(),sizeof(uint64_t),VK_QUERY_RESULT_64_BIT));
        uint32_t bits=qprops[family].timestampValidBits;
        uint64_t mask=bits==64?UINT64_MAX:((uint64_t(1)<<bits)-1);
        auto interval_ms=[&](uint64_t a,uint64_t b){return double((b-a)&mask)*double(properties.limits.timestampPeriod)/1e6;};
        gpu_times.push_back(interval_ms(timestamps[0],timestamps[1]));
        if(perPassTiming)for(size_t i=0;i<pass_times.size();++i)
            pass_times[i].push_back(interval_ms(timestamps[i+2],timestamps[i+3]));
        host_times.push_back(host_ms(a,b,frequency));
        std::cout<<"network sample "<<rep<<" gpu_ms="<<gpu_times.back()<<std::endl;
        if(saveFrames)copy_output(plan.outputs[0],"frame_"+std::to_string(frame)+".bin");
    }
    // `--per-pass-timing` used to collect the timestamps and print nothing, so every attempt to
    // price a fusion started from a guess. Emit one machine-readable line per pass and let the
    // caller aggregate by family: `PASSMS <name> <mean_ms> <gx> <gy> <shared> <row>`.
    // The grid and the shared-memory request come from the plan, so an outlier can be priced
    // against its workgroup count and its occupancy limit without a second instrumented build.
    if(perPassTiming&&!pass_times.empty()) {
        double total=0;
        for(size_t i=0;i<pass_times.size();++i) {
            double mean=0;
            for(double v:pass_times[i]) mean+=v;
            mean/=double(pass_times[i].size()?pass_times[i].size():1);
            total+=mean;
            const auto& p=plan.passes[i];
            std::cout<<"PASSMS "<<p.name<<" "<<mean<<" "<<p.gx<<" "<<p.gy<<" "<<p.shared<<" "<<p.row<<"\n";
        }
        std::cout<<"PASSMS_TOTAL "<<total<<" "<<pass_times.size()<<"\n";
    }
    uint64_t n_nan=0,n_inf=0;
    for(auto& output:plan.outputs) {
        const auto& buffer=plan.buffers[output.resource];
        begin();
        barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferCopy copy{0,0,buffer.bytes};
        vkCmdCopyBuffer(command,buffer.handle,stage.handle,1,&copy);
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);
        submit();
        if(output.dtype=="float16") {
            auto data=static_cast<uint16_t*>(mapped);
            for(size_t i=0;i<buffer.bytes/2;++i) if((data[i]&0x7c00)==0x7c00) {
                if(data[i]&0x3ff) ++n_nan; else ++n_inf;
            }
        } else if(output.dtype=="float32") {
            auto data=static_cast<uint32_t*>(mapped);
            for(size_t i=0;i<buffer.bytes/4;++i) if((data[i]&0x7f800000)==0x7f800000) {
                if(data[i]&0x7fffff) ++n_nan; else ++n_inf;
            }
        }
        std::ofstream dump(std::filesystem::path(out)/output.file,std::ios::binary);
        dump.write(static_cast<const char*>(mapped),std::streamsize(buffer.bytes));
        if(!dump) throw std::runtime_error("output write failed: "+output.file);
    }
    CHECK(vkQueueWaitIdle(queue));
    vkDestroyQueryPool(device,queries,nullptr);
    vkDestroyDescriptorPool(device,descriptors,nullptr);
    for(auto& pipeline:pipelines) {
        vkDestroyPipeline(device,pipeline.handle,nullptr);
        vkDestroyPipelineLayout(device,pipeline.layout,nullptr);
        vkDestroyDescriptorSetLayout(device,pipeline.descriptors,nullptr);
    }
    vkUnmapMemory(device,stage.memory);
    vkDestroyBuffer(device,stage.handle,nullptr); vkFreeMemory(device,stage.memory,nullptr);
    for(auto& buffer:plan.buffers) {vkDestroyBuffer(device,buffer.handle,nullptr);if(buffer.ownsMemory)vkFreeMemory(device,buffer.memory,nullptr);}
    for(auto mem:poolMemory)vkFreeMemory(device,mem,nullptr);
    vkDestroyFence(device,fence,nullptr); vkDestroyCommandPool(device,pool,nullptr);
    vkDestroyDevice(device,nullptr); vkDestroyInstance(instance,nullptr); FreeLibrary(dll);
    auto sorted=gpu_times;
    std::sort(sorted.begin(),sorted.end());
    double sum=0;
    for(double value:gpu_times) sum+=value;
    std::ofstream receipt(std::filesystem::path(out)/"port_network_receipt.json");
    receipt<<std::setprecision(9)<<"{\n  \"status\": \""<<((n_nan||n_inf)?"NONFINITE":"GPU_EXECUTED_NUMERIC_PENDING")
           <<"\",\n  \"normal_close\": true,\n  \"original_plan_rows\": "<<kPlanRows<<",\n  \"gpu_passes\": "<<plan.passes.size()
           <<",\n  \"device\": \""<<properties.deviceName<<"\",\n  \"driver_version_raw\": "<<properties.driverVersion
           <<",\n  \"vulkan_memory_model_enabled\": true, \"vulkan_memory_model_device_scope_enabled\": false, \"vulkan_memory_model_availability_visibility_chains_enabled\": false"
           <<",\n  \"warm\": "<<warm<<", \"reps\": "<<reps<<",\n  \"nan\": "<<n_nan<<", \"inf\": "<<n_inf
           <<",\n  \"allocation_mode\": \""<<(pooled?"pool":"flat")<<"\", \"feedback\": "<<(feedback?"true":"false")
           <<", \"dump_alias_disabled\": "<<(dump?"true":"false")<<", \"pool_allocated_bytes\": "<<poolAllocated
           <<",\n  \"timestamp_scope\": \"one full actual network, all dispatches and barriers; upload/readback excluded\",\n"
           <<"  \"gpu_mean_ms\": "<<sum/reps<<", \"gpu_median_ms\": "<<sorted[reps/2]
           <<", \"gpu_p95_ms\": "<<sorted[(reps*95)/100]<<",\n  \"gpu_samples_ms\": [";
    for(size_t i=0;i<gpu_times.size();++i) receipt<<(i?",":"")<<gpu_times[i];
    receipt<<"],\n  \"host_submit_complete_ms\": [";
    for(size_t i=0;i<host_times.size();++i) receipt<<(i?",":"")<<host_times[i];
    receipt<<"],\n  \"per_pass_timing\": "<<(perPassTiming?"true":"false")
           <<",\n  \"timestamp_count_per_frame\": "<<qpci.queryCount
           <<",\n  \"per_pass_interval_scope\": \"consecutive BOTTOM_OF_PIPE boundaries: includes first-use clears, dispatch, barriers and instrumentation effects; no isolated dispatch/barrier/gap attribution\",\n"
           <<"  \"per_pass_gpu\": [\n";
    for(size_t i=0;i<pass_times.size();++i) {
        const auto& pass=plan.passes[i];auto samples=pass_times[i];
        std::sort(samples.begin(),samples.end());double passSum=0;
        for(double value:pass_times[i])passSum+=value;
        receipt<<(i?",\n":"")<<"    {\"index\": "<<i<<", \"row\": "<<pass.row
               <<", \"name\": \""<<pass.name<<"\", \"mean_ms\": "<<passSum/reps
               <<", \"median_ms\": "<<samples[reps/2]<<", \"p95_ms\": "<<samples[(reps*95)/100]
               <<", \"samples_ms\": [";
        for(size_t j=0;j<pass_times[i].size();++j)receipt<<(j?",":"")<<pass_times[i][j];
        receipt<<"]}";
    }
    receipt<<"\n  ],\n  \"memory\": [\n";
    for(size_t i=0;i<plan.buffers.size();++i) {
        const auto& buffer=plan.buffers[i];
        const auto& type=memory.memoryTypes[buffer.memoryType];
        receipt<<(i?",\n":"")<<"    {\"name\": \""<<buffer.name<<"\", \"bytes\": "<<buffer.bytes
               <<", \"allocated_bytes\": "<<buffer.allocated<<", \"memory_type\": "<<buffer.memoryType
               <<", \"alignment\": "<<buffer.alignment<<", \"type_bits\": "<<buffer.typeBits
               <<", \"dedicated_required\": "<<(buffer.dedicatedRequired?"true":"false")
               <<", \"dedicated_preferred\": "<<(buffer.dedicatedPreferred?"true":"false")
               <<", \"memory_offset\": "<<buffer.memoryOffset<<", \"resident\": "<<(buffer.resident?"true":"false")
               <<", \"clear_first_each_replay\": "<<(buffer.clearFirst?"true":"false")
               <<", \"property_flags\": "<<type.propertyFlags<<", \"heap_index\": "<<type.heapIndex
               <<", \"heap_bytes\": "<<memory.memoryHeaps[type.heapIndex].size<<", \"device_local\": true}";
    }
    receipt<<"\n  ]\n}\n";
    if(!receipt) throw std::runtime_error("receipt write failed");
    return (n_nan||n_inf)?1:0;
}

int main(int argc,char** argv) {
    std::string path,out;
    int warm=1,reps=30;
    bool cpu=false,pooled=false,feedback=false,dump=false,saveFrames=false,perPassTiming=false;
    int patternW=0,patternH=0;
    // --input=WxH:FILE feeds a real frame (planar RGBA f32, the same layout frame_probe --input
    // writes) instead of the synthetic pattern. The pattern is smooth by construction, and a
    // 4-row phase error in the network's edit is invisible on smooth content: it shows up only as
    // the *magnitude* of the edit alternating with y%4 on real texture. Hence a real frame.
    int inputW=0,inputH=0; std::string inputPath;
    // --plane-mv=X,Y and --plane-depth=ramp|invramp|zero|one fill the two guide planes the host
    // gather writes in game. Without them every offline run is a zero-motion, zero-depth network.
    float planeMvX=0.0f,planeMvY=0.0f; int planeDepth=0;
    // AMD's control options, same names and same defaults as nr_graph.cpp:2362-2367. The offline
    // run and the shipped plugin then resolve the same five conditioning features from the same
    // numbers, instead of each path carrying its own idea of what the defaults are.
    ModelControls mc;
    for(int i=1;i<argc;++i) {
        std::string arg=argv[i];
        if(arg=="--help") {
            std::cout<<"--plan TXT --out FRESH_DIRECTORY [--pool --feedback --save-frames --dump-intermediates --per-pass-timing --warm 1 --reps 30] [--input-pattern WxH] [--input WxH:FILE]"
                       " [--style N --local-tone-strength F --local-structure-strength F --skin-structure-strength F --auto-mask 0|1]"
                       " | --cpu-check --plan TXT\n";
            return 0;
        } else if(arg=="--plan" && i+1<argc) path=argv[++i];
        else if(arg=="--out" && i+1<argc) out=argv[++i];
        else if(arg=="--warm" && i+1<argc) warm=std::stoi(argv[++i]);
        else if(arg=="--reps" && i+1<argc) reps=std::stoi(argv[++i]);
        else if(arg=="--cpu-check") cpu=true;
        else if(arg=="--pool")pooled=true;
        else if(arg=="--feedback")feedback=true;
        else if(arg=="--dump-intermediates")dump=true;
        else if(arg=="--save-frames")saveFrames=true;
        else if(arg=="--per-pass-timing")perPassTiming=true;
        else if(arg=="--input-pattern" && i+1<argc) {
            std::string size=argv[++i]; const auto x=size.find('x');
            if(x==std::string::npos) { std::cerr<<"--input-pattern expects WxH"<<std::endl; return 1; }
            patternW=std::stoi(size.substr(0,x)); patternH=std::stoi(size.substr(x+1));
        }
        else if(arg.rfind("--input=",0)==0 || arg=="--input") {
            std::string spec=(arg=="--input")?(i+1<argc?argv[++i]:std::string()):arg.substr(8);
            const auto colon=spec.find(':');
            const auto x=(colon==std::string::npos)?std::string::npos:spec.substr(0,colon).find('x');
            if(colon==std::string::npos||x==std::string::npos) { std::cerr<<"--input expects WxH:FILE"<<std::endl; return 1; }
            const std::string size=spec.substr(0,colon);
            inputW=std::stoi(size.substr(0,x)); inputH=std::stoi(size.substr(size.find('x')+1));
            inputPath=spec.substr(colon+1);
        }
        else if(arg.rfind("--plane-mv=",0)==0) {
            const std::string v=arg.substr(11); const auto comma=v.find(',');
            if(comma==std::string::npos) { std::cerr<<"--plane-mv expects X,Y"<<std::endl; return 1; }
            planeMvX=std::stof(v.substr(0,comma)); planeMvY=std::stof(v.substr(comma+1));
        }
        else if(arg.rfind("--plane-depth=",0)==0) {
            const std::string v=arg.substr(14);
            planeDepth = v=="ramp"?1 : v=="invramp"?2 : v=="zero"?3 : v=="one"?4 : 0;
            if(planeDepth==0) { std::cerr<<"--plane-depth expects ramp|invramp|zero|one"<<std::endl; return 1; }
        }
        // AMD's control flags, verbatim names and defaults (nr_graph.cpp:2362-2367).
        else if(arg=="--style" && i+1<argc) mc.style=std::stoi(argv[++i]);
        else if(arg=="--local-tone-strength" && i+1<argc) mc.tone=std::stof(argv[++i]);
        else if(arg=="--local-structure-strength" && i+1<argc) mc.structure=std::stof(argv[++i]);
        else if(arg=="--skin-structure-strength" && i+1<argc) mc.skin=std::stof(argv[++i]);
        else if(arg=="--auto-mask" && i+1<argc) mc.mask=std::stoi(argv[++i])!=0;
        else { std::cerr<<"unknown/incomplete argument: "<<arg<<std::endl; return 1; }
    }
    try {
        if(path.empty() || (!cpu && out.empty()) || warm<0 || reps<1) throw std::runtime_error("invalid runner arguments; see --help");
        auto plan=read_plan(path);
        if(cpu) {
            for(const auto& b:plan.buffers) if(b.path!="-" && !std::filesystem::is_regular_file(b.path)) throw std::runtime_error("missing input asset "+b.path);
            for(const auto& p:plan.passes) if(!std::filesystem::is_regular_file(p.spv)) throw std::runtime_error("missing shader asset "+p.spv);
            std::cout<<"CPU_PLAN_OK "<<kPlanRows<<" rows, "<<plan.passes.size()<<" passes, "<<plan.buffers.size()<<" buffers; no Vulkan loaded\n"; return 0; }
        {
            const float structure = mc.structure;
            const float mask_skin  = mc.mask ? (mc.skin < 0 ? structure : mc.skin) : -1.0f;
            const float mask_other = mc.mask ? structure : -1.0f;
            const bool any_mask = std::max(mask_skin, mask_other) >= 0.0f;
            std::printf("pre conditioning [style,tone,structure,skin,other]=[%g,%g,%g,%g,%g]\n",
                        float(std::clamp(mc.style,0,2))/128.0f, mc.tone,
                        any_mask ? 1.0f : structure,
                        any_mask ? (mask_skin < 0 ? structure : mask_skin) : -1.0f,
                        any_mask ? (mask_other < 0 ? structure : mask_other) : -1.0f);
            std::fflush(stdout);
        }
        return execute(plan,out,warm,reps,pooled,feedback,dump,saveFrames,perPassTiming,
                       patternW,patternH,inputW,inputH,inputPath,planeMvX,planeMvY,planeDepth,mc);
    } catch(const std::exception& error) { std::cerr<<"PORT_NETWORK_FAIL: "<<error.what()<<std::endl; return 1; }
}
