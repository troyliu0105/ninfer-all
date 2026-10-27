#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "core/layout.h"
#include "core/arena.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class SelectorRoute { Direct, Lattice };

// Whether a codebook is read as its stored rows (contiguous BF16 words) or gathered through a
// compact FP32 plane: a codebook in its own ggml block format has no BF16 row to read directly, so
// the round decodes exactly the rows it walks.
[[nodiscard]] bool selector_bf16_rows(const Weight& codebook);

struct SelectorPlan {
    SelectorRoute route = SelectorRoute::Direct;
    bool staged         = false;
};

struct SelectorWorkspace {
    Tensor edges;
    // One FP32 row of 256 values per staged codebook row.
    Tensor successor_rows, predecessor_rows;
    // The predecessor rows' token ids, as the gather reads them.
    Tensor predecessor_ids;
};

SelectorPlan candidate_selector_path_plan(int steps, int batch, bool staged);
const char* selector_route_name(SelectorPlan plan);
const char* candidate_selector_path_route_name(int steps, int batch, bool staged);
// Predecessor rows one round gathers: the anchor, then every candidate rank of every earlier step.
[[nodiscard]] int selector_predecessor_rows(int steps, int batch);

template <class Allocator>
SelectorWorkspace allocate_selector_workspace(Allocator& allocator, SelectorPlan plan, int steps,
                                              int batch) {
    SelectorWorkspace out;
    if (plan.route == SelectorRoute::Lattice)
        out.edges = allocator.alloc(DType::FP32, {16, 16, steps, batch}, 256);
    if (plan.staged) {
        out.successor_rows = allocator.alloc(DType::FP32, {16 * steps * batch, 256});
        out.predecessor_rows =
            allocator.alloc(DType::FP32, {selector_predecessor_rows(steps, batch), 256});
        out.predecessor_ids = allocator.alloc(DType::I32, {selector_predecessor_rows(steps, batch)});
    }
    return out;
}

void candidate_selector_path_dispatch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                      const Tensor& projected_hidden, const Tensor& anchors,
                                      const Weight& predecessor_codebook,
                                      const Weight& successor_codebook,
                                      const Tensor& base_positions, const SamplingConfig* configs,
                                      Tensor& drafts, Tensor& proposal_q, WorkspaceArena& workspace,
                                      cudaStream_t stream);

} // namespace ninfer::ops::detail
