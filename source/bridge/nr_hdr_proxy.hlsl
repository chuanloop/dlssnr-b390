// Candidate for the Cyberpunk R11G11B10 -> SDR NR -> R11G11B10 path.
// The original scene-linear source is never clamped in the final composite.
// The _f16 entry points add the same pipeline for an R16G16B16A16_FLOAT
// source: RGB follows the identical formulas; alpha keeps the source value.
Texture2D<float3> GameColor : register(t0);
// The FP16 source variant binds the same t0 slot with its own typed view.
Texture2D<float4> GameColorF16 : register(t0);
// t1 is the game's RG16F motion during prepare and the NR color during composite.
// The descriptor changes only after the prepare fence has completed.
Texture2D<float4> MotionOrNR : register(t1);
RWTexture2D<float4> Proxy : register(u0);
RWTexture2D<float2> NRMotion : register(u1);
RWTexture2D<float3> XeSSInput : register(u2);
// The FP16 variant writes the same u2 slot; the descriptor carries the
// resource's own R16G16B16A16_FLOAT format.
RWTexture2D<float4> XeSSInputF16 : register(u2);
cbuffer Constants : register(b0) {
    // The four float4 blocks come first so they stay 16-byte aligned; every
    // remaining scalar then packs tightly. The host fills 27 dwords in this
    // exact order (root 32-bit constants, b0).
    float4 Curve0; float4 Curve1; float4 Curve2; float4 Curve3;
    uint Width; uint Height; uint UseGameMotion;
    float MotionScaleX; float MotionScaleY;
    float StyleSat; uint StyleMode; float StyleTone;
    float StyleExposure; float StyleContrast; float StyleGain;
};


// ---------------------------------------------------------------- style grade
// Port of the reference NativeStylePost peripheral: a 16-point tone curve for
// style 1 (or its analytic exposure/contrast form when tone != 1), identity for
// styles 0 and 2, then an HSL saturation scale. Style 0 is a passthrough.
float curve_sample(float x) {
    float t = saturate(x) * 15.0f;
    int i = (int)floor(t);
    float w = t - (float)i;
    float lut[16] = { Curve0.x, Curve0.y, Curve0.z, Curve0.w,
                      Curve1.x, Curve1.y, Curve1.z, Curve1.w,
                      Curve2.x, Curve2.y, Curve2.z, Curve2.w,
                      Curve3.x, Curve3.y, Curve3.z, Curve3.w };
    return lerp(lut[i], lut[min(i + 1, 15)], w);
}
float style_curve(float x) {
    // StyleGain scales the deviation from identity: 1 is the reference,
    // 2 doubles it, 0 disables the curve.
    if (StyleMode == 1u) {
        float y = curve_sample(x);
        if (StyleTone < 0.999f) {
            float v = saturate(x * StyleExposure);
            float delta = v * v * (3.0f - 2.0f * v) - v;
            y = v + delta * StyleContrast;
        }
        return saturate(x + (y - x) * StyleGain);
    }
    return saturate(x);
}
float hue_to_rgb(float p, float q, float t) {
    t = frac(t);
    if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
    if (t < 0.5f) return q;
    if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
    return p;
}
float3 style_grade(float3 nr) {
    if (StyleMode == 0u) return nr;
    float3 c = float3(style_curve(nr.r), style_curve(nr.g), style_curve(nr.b));
    float maximum = max(c.r, max(c.g, c.b));
    float minimum = min(c.r, min(c.g, c.b));
    float lightness = (maximum + minimum) * 0.5f;
    float difference = maximum - minimum;
    float hue = 0.0f, saturation = 0.0f;
    if (difference > 0.0f) {
        saturation = (lightness > 0.5f)
            ? difference / max(2.0f - maximum - minimum, 1e-6f)
            : difference / max(maximum + minimum, 1e-6f);
        if (maximum == c.r)
            hue = (c.g - c.b) / difference + ((c.g < c.b) ? 6.0f : 0.0f);
        else if (maximum == c.g)
            hue = (c.b - c.r) / difference + 2.0f;
        else
            hue = (c.r - c.g) / difference + 4.0f;
        hue *= 1.0f / 6.0f;
    }
    saturation = saturate(saturation * (1.0f + (StyleSat - 1.0f) * StyleGain));
    float q = (lightness < 0.5f) ? lightness * (1.0f + saturation)
                                 : lightness + saturation - lightness * saturation;
    float p = 2.0f * lightness - q;
    return saturate(float3(hue_to_rgb(p, q, hue + 1.0f / 3.0f),
                           hue_to_rgb(p, q, hue),
                           hue_to_rgb(p, q, hue - 1.0f / 3.0f)));
}

[numthreads(8,8,1)]
void prepare(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= Width || thread.y >= Height) return;
    uint2 xy = thread.xy;
    float3 scene = GameColor.Load(int3(xy, 0));
    Proxy[xy] = float4(saturate(scene), 1.0f);
    float2 pixels = float2(0.0f, 0.0f);
    if (UseGameMotion != 0) {
        pixels = MotionOrNR.Load(int3(xy, 0)).xy *
            float2(MotionScaleX, MotionScaleY);
        // The shared bridge consumes RG16F. Reject invalid values rather
        // than writing NaNs or infinities into its temporal reprojection.
        if (!all(isfinite(pixels))) pixels = float2(0.0f, 0.0f);
        pixels = clamp(pixels, -65504.0f, 65504.0f);
    }
    NRMotion[xy] = pixels;
}

[numthreads(8,8,1)]
void composite(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= Width || thread.y >= Height) return;
    uint2 xy = thread.xy;
    float3 scene = GameColor.Load(int3(xy, 0));
    float3 sdr = saturate(scene);
    float3 nr = MotionOrNR.Load(int3(xy, 0)).rgb;
    float peak = max(scene.r, max(scene.g, scene.b));
    float weight = saturate((1.2f - peak) / 0.4f);
    // The reference grades the composed image (original + amount*(grade-original)),
    // not just the neural output, so the look must not be scaled by the HDR
    // weight. Grade the tone-mapped part and keep the HDR excess above it.
    float3 composed = max(scene + weight * (nr - sdr), 0.0f);
    float3 base = saturate(composed);
    XeSSInput[xy] = max(composed - base + style_grade(base), 0.0f);
}

// FP16 source variant (R16G16B16A16_FLOAT). RGB follows the exact formulas
// of the R11G11B10 path; alpha carries the source value through unchanged.
[numthreads(8,8,1)]
void prepare_f16(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= Width || thread.y >= Height) return;
    uint2 xy = thread.xy;
    float4 scene4 = GameColorF16.Load(int3(xy, 0));
    float3 scene = scene4.rgb;
    Proxy[xy] = float4(saturate(scene), scene4.a);
    float2 pixels = float2(0.0f, 0.0f);
    if (UseGameMotion != 0) {
        pixels = MotionOrNR.Load(int3(xy, 0)).xy *
            float2(MotionScaleX, MotionScaleY);
        // The shared bridge consumes RG16F. Reject invalid values rather
        // than writing NaNs or infinities into its temporal reprojection.
        if (!all(isfinite(pixels))) pixels = float2(0.0f, 0.0f);
        pixels = clamp(pixels, -65504.0f, 65504.0f);
    }
    NRMotion[xy] = pixels;
}

[numthreads(8,8,1)]
void composite_f16(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= Width || thread.y >= Height) return;
    uint2 xy = thread.xy;
    float4 scene4 = GameColorF16.Load(int3(xy, 0));
    float3 scene = scene4.rgb;
    float3 sdr = saturate(scene);
    float3 nr = MotionOrNR.Load(int3(xy, 0)).rgb;
    float peak = max(scene.r, max(scene.g, scene.b));
    float weight = saturate((1.2f - peak) / 0.4f);
    float3 composed = max(scene + weight * (nr - sdr), 0.0f);
    float3 base = saturate(composed);
    XeSSInputF16[xy] = float4(max(composed - base + style_grade(base), 0.0f), scene4.a);
}
