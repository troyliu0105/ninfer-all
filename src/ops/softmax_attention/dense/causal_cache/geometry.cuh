#pragma once

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

// A causal attention geometry fixes the query heads per KV head (GroupSize) and the small-T split
// scale at compile time. How many KV heads a call attends with is a run-time count: the model's,
// or a tensor-parallel rank's run of them, at most ModelKVHeads. Every kernel takes it from its
// launch grid, which spans the call's KV heads (small-T partials, packed-GQA prompts) or query
// heads (reductions, prompts, merges). ModelQHeads names the model geometry whose device-profile
// routes a call takes.
template <int GroupSizeValue, int ModelKVHeadsValue, int SmallTSplitScaleValue>
struct CausalAttentionGeometry {
    static_assert(GroupSizeValue > 0 && ModelKVHeadsValue > 0 && SmallTSplitScaleValue > 0);

    static constexpr int GroupSize           = GroupSizeValue;
    static constexpr int ModelKVHeads        = ModelKVHeadsValue;
    static constexpr int ModelQHeads         = GroupSize * ModelKVHeads;
    static constexpr int SmallTSplitScale    = SmallTSplitScaleValue;
    static constexpr int SmallTMaximumSplits = 85 * SmallTSplitScale;
};

using CausalD256H24Kv4 = CausalAttentionGeometry<6, 4, 1>;
using CausalD256H16Kv2 = CausalAttentionGeometry<8, 2, 2>;
// The Qwen3.5-2B dense geometry: 8 query heads over 2 KV heads (group 4), half the small-T
// splits of the 16-head model.
using CausalD256H8Kv2 = CausalAttentionGeometry<4, 2, 1>;

// Whether `query_heads` over `kv_heads` is one of those geometries or a run of its KV heads.
[[nodiscard]] constexpr bool causal_attention_geometry_supported(std::int32_t query_heads,
                                                                 std::int32_t kv_heads) {
    if (kv_heads <= 0 || query_heads % kv_heads != 0) { return false; }
    const std::int32_t group = query_heads / kv_heads;
    return (group == CausalD256H24Kv4::GroupSize && kv_heads <= CausalD256H24Kv4::ModelKVHeads) ||
           (group == CausalD256H16Kv2::GroupSize && kv_heads <= CausalD256H16Kv2::ModelKVHeads) ||
           (group == CausalD256H8Kv2::GroupSize && kv_heads <= CausalD256H8Kv2::ModelKVHeads);
}

// Calls `f.template operator()<Geometry>()` with the geometry of `query_heads` over `kv_heads`.
// Throws for any other pair.
template <typename F>
decltype(auto) with_causal_geometry(std::int32_t query_heads, std::int32_t kv_heads, F&& f) {
    if (!causal_attention_geometry_supported(query_heads, kv_heads)) {
        throw std::invalid_argument("causal attention: unsupported head geometry");
    }
    const std::int32_t group = query_heads / kv_heads;
    if (group == CausalD256H24Kv4::GroupSize) {
        return f.template operator()<CausalD256H24Kv4>();
    }
    if (group == CausalD256H8Kv2::GroupSize) {
        return f.template operator()<CausalD256H8Kv2>();
    }
    return f.template operator()<CausalD256H16Kv2>();
}

// The same for a call named by its query heads alone. A call holds whole groups of query heads,
// and the geometries' counts (6, 12, 18 or 24, 4 or 8, and 8 or 16) overlap at 8: the 2B model's
// whole 8 over 2 KV heads and a 16-head model's half over 1. `kv_heads` — every call site holds
// the cache view's count — separates them.
template <typename F>
decltype(auto) with_causal_query_geometry(std::int32_t query_heads, std::int32_t kv_heads,
                                          F&& f) {
    return with_causal_geometry(query_heads, kv_heads, static_cast<F&&>(f));
}

// The query heads of the model whose geometry a call with `query_heads` takes: a tensor-parallel
// share takes its model's measured routes, chunks and tiles.
[[nodiscard]] inline std::int32_t causal_attention_model_query_heads(std::int32_t query_heads,
                                                                     std::int32_t kv_heads) {
    if (!causal_attention_geometry_supported(query_heads, kv_heads)) {
        throw std::invalid_argument("causal attention: unsupported head geometry");
    }
    const std::int32_t group = query_heads / kv_heads;
    if (group == CausalD256H24Kv4::GroupSize) { return CausalD256H24Kv4::ModelQHeads; }
    if (group == CausalD256H8Kv2::GroupSize) { return CausalD256H8Kv2::ModelQHeads; }
    return CausalD256H16Kv2::ModelQHeads;
}

} // namespace ninfer::ops
