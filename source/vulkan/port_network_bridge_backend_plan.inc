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

// Windows paths in the adapted runtime are written with single backslashes.
// std::quoted() treats '\' as an escape character and silently strips them,
// turning "E:\NR-..." into "E:NR-...". Read the quoted token verbatim instead;
// this changes path parsing only, not plan semantics, buffers or SPIR-V.
static std::string read_quoted_path(std::istream& in) {
    char ch = 0;
    in >> std::ws;
    if (!(in >> ch) || ch != '"') throw std::runtime_error("expected quoted path");
    std::string value;
    while (in.get(ch)) {
        if (ch == '"') return value;
        value.push_back(ch);
    }
    throw std::runtime_error("unterminated quoted path");
}

static Plan read_plan(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot read network plan " + path);
    std::string word;
    size_t rows, nb, np, no;
    file >> word >> rows >> nb >> np >> no;
    if (word != "PORT_NETWORK_1" || rows != kPlanRows || !nb || !np || !no)
        throw std::runtime_error("network requires a complete prepared plan");
    Plan plan;
    for (size_t i = 0; i < nb; ++i) {
        Buffer buffer;
        file >> word;
        buffer.name = read_quoted_path(file);
        file >> buffer.bytes;
        buffer.path = read_quoted_path(file);
        if (word != "BUFFER" || !buffer.bytes) throw std::runtime_error("invalid BUFFER");
        plan.buffers.push_back(buffer);
    }
    std::vector<bool> covered(rows, false);
    int previous = -1;
    for (size_t i = 0; i < np; ++i) {
        Pass pass;
        size_t nbind, npush;
        file >> word >> pass.row;
        pass.name = read_quoted_path(file);
        pass.spv = read_quoted_path(file);
        file >> pass.subgroup >> pass.shared >> nbind >> npush >> pass.gx >> pass.gy >> pass.gz;
        if (word != "PASS" || pass.row >= rows || int(pass.row) < previous ||
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
    const auto directory=std::filesystem::path(path).parent_path();
    for(auto& buffer:plan.buffers)if(buffer.path!="-")
        buffer.path=(directory/std::filesystem::path(buffer.path)).lexically_normal().string();
    for(auto& pass:plan.passes)
        pass.spv=(directory/std::filesystem::path(pass.spv)).lexically_normal().string();
    return plan;
}

static std::vector<char> read_bytes(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read " + path);
    return std::vector<char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}
