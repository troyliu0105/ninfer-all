#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ops/candidate_selector/bf16/candidate_selector_path_plan.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void candidate_selector_path_launch(SelectorPlan plan, const Tensor& candidate_ids,
                                    const Tensor& unary_scores, const Tensor& projected_hidden,
                                    const Tensor& anchors, const Weight& predecessor_codebook,
                                    const Weight& successor_codebook, const Tensor& base_positions,
                                    const SamplingConfig* configs, Tensor& drafts,
                                    Tensor& proposal_q, SelectorWorkspace& workspace,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
