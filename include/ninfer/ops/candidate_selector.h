#pragma once

#include "core/tensor.h"
#include "core/arena.h"
#include "core/weight.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

// Capacity for every K/B pair in the inclusive intervals, K=1..15 and B=1..8. A codebook stored
// as ggml blocks (rather than contiguous BF16 words) adds the compact FP32 row planes the path
// gathers for one round: 16*K*B successor rows and B + 16*(K-1)*B predecessor rows of 256 values,
// beside their row ids.
[[nodiscard]] std::size_t candidate_selector_path_workspace_capacity_bytes(
    const Weight& predecessor_codebook, const Weight& successor_codebook, std::int32_t min_steps,
    std::int32_t max_steps, std::int32_t min_batch, std::int32_t max_batch);

/**
 * Op: candidate_selector_path
 *
 * For K in [1,15] and B in [1,8], the inputs are contiguous candidate_ids I32 [16,K,B],
 * unary_scores FP32 [16,K,B], projected_hidden BF16 [256,K,B], anchors I32 [B],
 * predecessor_codebook and successor_codebook [T,256] (one BF16 word or one stored ggml block row
 * per padded token id), base_positions I32 [B], and a device-resident SamplingConfig[B]. Candidate
 * rank is the fastest axis. The 16 candidate ids in each row are distinct, and all candidate and
 * anchor token ids lie in [0,T); the registered vocabulary, artifact binding, and linear_topk
 * producer establish that trusted value contract.
 *
 * Starting with predecessor=anchors[b], each position i in [0,K) computes:
 *
 *   edge[c] = unary_scores[c,i,b]
 *           + sum_r predecessor_codebook[r,predecessor]
 *                   * projected_hidden[r,i,b]
 *                   * successor_codebook[r,candidate_ids[c,i,b]].
 *
 * A row with configs[b].temperature<=0 selects the lowest candidate rank attaining max(edge) and
 * writes its exact one-hot distribution. A positive-temperature row writes the FP32 softmax of
 * edge/temperature, then draws a candidate with counter key
 * (configs[b].seed,base_positions[b]+i,kSamplePurposeDFlash2Proposal). The selected global id is
 * written to drafts[i,b] and becomes the next predecessor. The Op ignores all other
 * SamplingConfig fields and never updates token_counts.
 *
 * A codebook stored as ggml blocks is decoded a bounded set of rows at a time -- the round's own
 * candidates once, and its predecessors once per candidate rank -- into FP32 planes; the token
 * table is never expanded and no row is decoded twice. The arithmetic is the same either way: a
 * stored row's represented values take part in the same FP64-evaluated formula.
 *
 * drafts is contiguous I32 [K,B] and proposal_q is contiguous FP32 [16,K,B]. Both outputs are
 * completely overwritten. Inputs, outputs, codebooks, and the config array must be pairwise
 * non-overlapping. The Op has no persistent state or internal allocation. Caller workspace is
 * transient and must not overlap any input or output.
 */
void candidate_selector_path(const Tensor& candidate_ids, const Tensor& unary_scores,
                             const Tensor& projected_hidden, const Tensor& anchors,
                             const Weight& predecessor_codebook,
                             const Weight& successor_codebook, const Tensor& base_positions,
                             const SamplingConfig* configs, Tensor& drafts, Tensor& proposal_q,
                             WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
