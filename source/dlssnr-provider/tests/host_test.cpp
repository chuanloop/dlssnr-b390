// Offline host for the dlssnr probe: stands in for OptiScaler so the ABI can be
// exercised without launching a game.
//
// It implements NVSDK_NGX_Parameter the same way the caller does (a typed map),
// fills the DLSSNR.* keys OptiScaler is known to write, and drives the D3D12
// entry points. Then it checks the probe log actually captured the width, the
// height and the resource pointers - which is what would otherwise only be
// discoverable after a game restart.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

#include <nvsdk_ngx_params.h>

#include "../dlssnr_abi.h"

// Imported from nvngx.dll_dlssnr.dll; the host is linked against its import lib.
extern "C" {
int __cdecl dlssnr_call_create(void* params, void* out_handle);
int __cdecl dlssnr_call_evaluate(void* params, void* handle);
int __cdecl dlssnr_call_set_float_slot(int slot, float value);
float __cdecl dlssnr_call_probe_float(void* params, void* unused);
float __cdecl dlssnr_query_scaling_ratio(float* out_ratio);
}

namespace {

struct Value {
    enum class Kind { None, UInt, Int, Float, Ptr } kind = Kind::None;
    unsigned long long u = 0;
    float f = 0.f;
    void* p = nullptr;
};

class Param final : public NVSDK_NGX_Parameter {
public:
    void Set(const char* name, unsigned long long v) override { m_[name] = {Value::Kind::UInt, v}; }
    void Set(const char* name, float v) override { store_float(name, v); }
    void Set(const char* name, double v) override { store_float(name, float(v)); }
    void Set(const char* name, unsigned int v) override { m_[name] = {Value::Kind::UInt, v}; }
    void Set(const char* name, int v) override {
        Value x; x.kind = Value::Kind::Int; x.u = static_cast<unsigned long long>(static_cast<long long>(v));
        m_[name] = x;
    }
    void Set(const char* name, ID3D11Resource*) override { (void)name; }
    void Set(const char* name, ID3D12Resource*) override { (void)name; }
    void Set(const char* name, void* v) override {
        Value x; x.kind = Value::Kind::Ptr; x.p = v; m_[name] = x;
    }

    NVSDK_NGX_Result Get(const char* name, unsigned long long* out) const override {
        const Value* v = find(name, Value::Kind::UInt);
        if (!v) return NVSDK_NGX_Result_FAIL_InvalidParameter;
        *out = v->u; return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result Get(const char* name, float* out) const override {
        const Value* v = find(name, Value::Kind::Float);
        if (!v) return NVSDK_NGX_Result_FAIL_InvalidParameter;
        *out = v->f; return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result Get(const char* name, double* out) const override {
        const Value* v = find(name, Value::Kind::Float);
        if (!v) return NVSDK_NGX_Result_FAIL_InvalidParameter;
        *out = double(v->f); return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result Get(const char* name, unsigned int* out) const override {
        const Value* v = find(name, Value::Kind::UInt);
        if (!v) return NVSDK_NGX_Result_FAIL_InvalidParameter;
        *out = unsigned(v->u); return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result Get(const char* name, int* out) const override {
        const Value* v = find(name, Value::Kind::Int);
        if (!v) return NVSDK_NGX_Result_FAIL_InvalidParameter;
        *out = int(v->u); return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result Get(const char* name, ID3D11Resource** out) const override {
        const Value* v = find(name, Value::Kind::Ptr);
        if (!v) return NVSDK_NGX_Result_FAIL_InvalidParameter;
        *out = nullptr; return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result Get(const char* name, ID3D12Resource** out) const override {
        const Value* v = find(name, Value::Kind::Ptr);
        if (!v) return NVSDK_NGX_Result_FAIL_InvalidParameter;
        *out = nullptr; return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result Get(const char* name, void** out) const override {
        const Value* v = find(name, Value::Kind::Ptr);
        if (!v) return NVSDK_NGX_Result_FAIL_InvalidParameter;
        *out = v->p; return NVSDK_NGX_Result_Success;
    }

    void Reset() override { m_.clear(); }

private:
    void store_float(const char* name, float v) {
        Value x; x.kind = Value::Kind::Float; x.f = v; m_[name] = x;
    }
    const Value* find(const char* name, Value::Kind kind) const {
        auto it = m_.find(name);
        if (it == m_.end() || it->second.kind != kind) return nullptr;
        return &it->second;
    }

    mutable std::unordered_map<std::string, Value> m_;
};

bool contains(const std::string& haystack, const char* needle) {
    return haystack.find(needle) != std::string::npos;
}

}  // namespace

int main(int argc, char** argv) {
    const char* log_path = argc > 1 ? argv[1] : "dlssnr_probe.log";

    Param params;
    // The keys OptiScaler writes, with plausible values.
    params.Set("DLSSNR.Width", 1280u);
    params.Set("DLSSNR.Height", 720u);
    params.Set("DLSSNR.Enabled", 1);
    params.Set("DLSSNR.Reset", 0);
    params.Set("DLSSNR.Hint.Render.Preset", 3);
    params.Set("DLSSNR.DepthInverted", 1);
    params.Set("DLSSNR.Intensity", 1.0f);
    params.Set("DLSSNR.ScalingRatio", 1.0f);
    params.Set("DLSSNR.LocalStructureStrength", 0.5f);
    params.Set("DLSSNR.LocalToneStrength", 0.25f);
    params.Set("DLSSNR.SkinStructureStrength", 0.75f);
    params.Set("DLSSNR.OptiScalerFloatProbe", 0.375f);
    params.Set("DLSSNR.MVecScaleX", 1.0f);
    params.Set("DLSSNR.MVecScaleY", 1.0f);
    params.Set("DLSSNR.Color", reinterpret_cast<void*>(0x1000));
    params.Set("DLSSNR.Depth", reinterpret_cast<void*>(0x2000));
    params.Set("DLSSNR.MVec", reinterpret_cast<void*>(0x3000));
    params.Set("DLSSNR.Output", reinterpret_cast<void*>(0x4000));

    void* handle = nullptr;
    const int create_rc = dlssnr_call_create(&params, &handle);
    printf("create rc=%d handle=%p\n", create_rc, handle);

    dlssnr_call_evaluate(&params, handle);
    dlssnr_call_set_float_slot(3, 0.75f);
    const float probe = dlssnr_call_probe_float(&params, nullptr);
    printf("probe_float -> %.4f (expected 0.375)\n", probe);
    printf("ratio      -> %.4f\n", dlssnr_query_scaling_ratio(nullptr));

    FILE* f = fopen(log_path, "r");
    if (!f) { printf("FAIL: log not found at %s\n", log_path); return 1; }
    std::string text;
    char chunk[4096];
    size_t got;
    while ((got = fread(chunk, 1, sizeof(chunk), f)) > 0) text.append(chunk, got);
    fclose(f);

    int failures = 0;
    struct Check { const char* needle; const char* what; };
    const Check checks[] = {
        {"w=1280 h=720", "dimensions read from the parameter object"},
        {"param@arg1", "parameter object located in arg1"},
        {"color=0000000000001000", "colour resource pointer"},
        {"out=0000000000004000", "output resource pointer"},
        {"slot=3 value=0.75000", "float slot captured"},
        {"probe -> returning 0.375000", "float probe echoed"},
    };
    for (const auto& c : checks) {
        const bool ok = contains(text, c.needle);
        printf("%s %s\n", ok ? "ok  " : "FAIL", c.what);
        if (!ok) ++failures;
    }
    printf(failures ? "\n%d check(s) failed\n" : "\nall checks passed\n", failures);
    return failures ? 1 : 0;
}
