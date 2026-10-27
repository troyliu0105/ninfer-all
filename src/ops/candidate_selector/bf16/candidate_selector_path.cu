#include "ops/candidate_selector/bf16/candidate_selector_path_kernels.h"
#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/sampling_device.cuh"
#include "ops/linear/gguf/gguf_linear.h"
#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kCandidates = 16, kRank = 256;

// A codebook's rows: contiguous BF16 words, or the compact FP32 plane a round decoded them into.
// `staged` is a compile-time choice, so neither path pays for the other's indirection.
struct DeviceArgs {
    const std::int32_t* ids;
    const float* unary;
    const __nv_bfloat16* hidden;
    const std::int32_t* anchors;
    const __nv_bfloat16* predecessor;
    const __nv_bfloat16* successor;
    const float* predecessor_rows;
    const float* successor_rows;
    const std::int32_t* positions;
    const SamplingConfig* configs;
    std::int32_t* drafts;
    float* q;
    int steps;
    int batch;
};

struct alignas(16) SelectorShared {
    __nv_bfloat16 successors[kCandidates * kRank];
    float product[kRank];
    float edge[kCandidates];
    int predecessor, predecessor_rank, base_position;
    float temperature;
    unsigned long long seed;
};

// The probabilities written here are the same FP32 values consumed by the draw.
__device__ int draw_rank(float edge, float temperature, unsigned long long seed, int position,
                         float* q) {
    const int lane      = threadIdx.x & 31;
    const float maximum = warp_max(edge);
    if (temperature <= 0.0F) {
        const unsigned winners =
            __ballot_sync(kFullWarpMask, lane < kCandidates && edge == maximum);
        const int selected = winners == 0 ? 0 : __ffs(winners) - 1;
        if (lane < kCandidates) q[lane] = lane == selected ? 1.0F : 0.0F;
        return selected;
    }
    const float weight      = lane < kCandidates ? __expf((edge - maximum) / temperature) : 0.0F;
    const float probability = weight / warp_sum(weight);
    if (lane < kCandidates) q[lane] = probability;
    float uniform =
        lane == 0 ? sampling_uniform(seed, position, kSamplePurposeDFlash2Proposal, 0U) : 0.0F;
    uniform          = __shfl_sync(kFullWarpMask, uniform, 0);
    float cumulative = probability;
#pragma unroll
    for (int offset = 1; offset < kCandidates; offset *= 2) {
        const float previous = __shfl_up_sync(kFullWarpMask, cumulative, offset);
        if (lane >= offset) cumulative += previous;
    }
    const unsigned hits = __ballot_sync(kFullWarpMask, lane < kCandidates && uniform < cumulative);
    return hits ? __ffs(hits) - 1 : kCandidates - 1;
}

// The compact predecessor table holds the anchor first, then every candidate rank of every earlier
// step: batch b's rows start at b * (1 + 16 * (K - 1)).
__device__ int predecessor_row(const DeviceArgs& a, int step, int batch, int rank) {
    const int per_batch = 1 + kCandidates * (a.steps - 1);
    return batch * per_batch + (step == 0 ? 0 : 1 + (step - 1) * kCandidates + rank);
}

// `step` only positions the staged predecessor row, and it is read after the barrier below: the
// draw that published the rank this row indexes belongs to the previous iteration.
template <bool Staged>
__device__ void score_row(const DeviceArgs& a, int column, int step, int batch,
                          SelectorShared& shared) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if constexpr (!Staged) {
        int token = lane == 0 ? a.ids[column * kCandidates + warp] : 0;
        token     = __shfl_sync(kFullWarpMask, token, 0);
        cp_async<16, Cache::cg>(&shared.successors[warp * kRank + lane * 8],
                                a.successor + static_cast<std::int64_t>(token) * kRank + lane * 8);
        cp_commit();
    }
    // Publish the preceding draw before reading its token. Successor prefetch is independent.
    __syncthreads();
    if (tid < kRank) {
        const float predecessor =
            Staged
                ? a.predecessor_rows[static_cast<std::int64_t>(
                                         predecessor_row(a, step, batch, shared.predecessor_rank)) *
                                         kRank + tid]
                : __bfloat162float(
                      a.predecessor[static_cast<std::int64_t>(shared.predecessor) * kRank + tid]);
        shared.product[tid] =
            predecessor *
            __bfloat162float(a.hidden[static_cast<std::int64_t>(column) * kRank + tid]);
    }
    if constexpr (!Staged) { cp_wait<0>(); }
    __syncthreads();
    {
        const int c = warp;
        float sum   = 0;
        if constexpr (Staged) {
            const float* row = a.successor_rows +
                               static_cast<std::int64_t>(column * kCandidates + c) * kRank;
            for (int r = lane; r < kRank; r += 32) sum = fmaf(shared.product[r], row[r], sum);
        } else {
            for (int r = lane; r < kRank; r += 32)
                sum = fmaf(shared.product[r],
                           __bfloat162float(shared.successors[c * kRank + r]), sum);
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) shared.edge[c] = a.unary[column * kCandidates + c] + sum;
    }
    __syncthreads();
}

template <bool Staged>
__global__ __launch_bounds__(512, 1) void selector_walk_kernel(DeviceArgs a) {
    pdl::enter();
    __shared__ SelectorShared shared;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, batch = blockIdx.x;
    if (tid == 0) {
        shared.predecessor      = a.anchors[batch];
        shared.predecessor_rank = 0;
        shared.base_position    = a.positions[batch];
        shared.temperature      = a.configs[batch].temperature;
        shared.seed             = a.configs[batch].seed;
    }
#pragma unroll 1
    for (int step = 0; step < a.steps; ++step) {
        const int column = batch * a.steps + step;
        score_row<Staged>(a, column, step, batch, shared);
        if (warp == 0) {
            const float edge   = lane < kCandidates ? shared.edge[lane] : -CUDART_INF_F;
            const int selected = draw_rank(edge, shared.temperature, shared.seed,
                                           shared.base_position + step, a.q + column * kCandidates);
            if (lane == 0) {
                shared.predecessor_rank = selected;
                shared.predecessor      = a.ids[column * kCandidates + selected];
                a.drafts[column]        = shared.predecessor;
            }
        }
    }
}

template <bool Staged>
__global__ __launch_bounds__(512, 2) void selector_lattice_kernel(DeviceArgs a, float* edges) {
    pdl::enter();
    const int column = blockIdx.x, p = blockIdx.y;
    const int step = column % a.steps, batch = column / a.steps;
    if (step == 0 && p != 0) return;
    __shared__ SelectorShared shared;
    if (threadIdx.x == 0) {
        shared.predecessor      = step == 0 ? a.anchors[batch] : a.ids[(column - 1) * kCandidates + p];
        shared.predecessor_rank = p;
    }
    score_row<Staged>(a, column, step, batch, shared);
    if (threadIdx.x < kCandidates)
        edges[(static_cast<std::int64_t>(column) * kCandidates + p) * kCandidates + threadIdx.x] =
            shared.edge[threadIdx.x];
}

__global__ __launch_bounds__(32) void selector_lattice_walk_kernel(DeviceArgs a,
                                                                   const float* edges) {
    pdl::enter();
    const int lane = threadIdx.x, batch = blockIdx.x;
    const auto seed         = a.configs[batch].seed;
    const float temperature = a.configs[batch].temperature;
    const int position      = a.positions[batch];
    int predecessor_rank    = 0;
#pragma unroll 1
    for (int step = 0; step < a.steps; ++step) {
        const int column = batch * a.steps + step;
        const float edge =
            lane < kCandidates
                ? edges[(static_cast<std::int64_t>(column) * kCandidates + predecessor_rank) *
                            kCandidates +
                        lane]
                : -CUDART_INF_F;
        int selected =
            draw_rank(edge, temperature, seed, position + step, a.q + column * kCandidates);
        predecessor_rank = selected;
        if (lane == 0) a.drafts[column] = a.ids[column * kCandidates + selected];
    }
}

// One row of each id, as the FP32 values it represents: the successor plane lists the round's own
// candidates in column order, and the predecessor plane its predecessors in predecessor_row order.
__global__ __launch_bounds__(256) void selector_gather_bf16(const __nv_bfloat16* table,
                                                            const std::int32_t* ids, int rows,
                                                            float* out) {
    pdl::enter();
    const int row = blockIdx.x, lane = threadIdx.x;
    const std::int32_t id = ids[row];
    out[static_cast<std::int64_t>(row) * kRank + lane] =
        __bfloat162float(table[static_cast<std::int64_t>(id) * kRank + lane]);
}

__global__ __launch_bounds__(256) void selector_predecessor_ids(const std::int32_t* candidate_ids,
                                                                const std::int32_t* anchors,
                                                                int steps, int batch,
                                                                std::int32_t* out) {
    pdl::enter();
    const int b         = blockIdx.x;
    const int per_batch = 1 + kCandidates * (steps - 1);
    const int lane      = blockIdx.y * blockDim.x + threadIdx.x;
    if (lane >= per_batch) return;
    if (lane == 0) {
        out[b * per_batch] = anchors[b];
        return;
    }
    const int slot = lane - 1, step = slot / kCandidates + 1, rank = slot % kCandidates;
    out[b * per_batch + lane] =
        candidate_ids[(static_cast<std::int64_t>(b * steps + step - 1)) * kCandidates + rank];
}

void gather_rows(const Weight& codebook, const Tensor& ids, int rows, Tensor& out,
                 cudaStream_t stream) {
    const auto* index = static_cast<const std::int32_t*>(ids.data);
    if (codebook.qtype == QType::BF16) {
        CUDA_CHECK(pdl::launch_consumer(
            {dim3(rows), dim3(kRank), 0, stream}, selector_gather_bf16,
            static_cast<const __nv_bfloat16*>(codebook.qdata), index, rows,
            static_cast<float*>(out.data)));
    } else {
        const auto block = gguf_block_shape(codebook.qtype);
        gguf::dequantize_rows(
            gguf::GgmlType(static_cast<int>(gguf_type(codebook.qtype))), codebook.qdata,
            std::int64_t(codebook.k / block.elements) * block.bytes, codebook.k, index, rows,
            static_cast<float*>(out.data), kRank, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <bool Staged>
void selector_launch(SelectorRoute route, const DeviceArgs& args, const Tensor& candidate_ids,
                     SelectorWorkspace& workspace, cudaStream_t stream) {
    if (route == SelectorRoute::Direct) {
        CUDA_CHECK(pdl::launch_consumer({dim3(candidate_ids.ne[2]), dim3(512), 0, stream},
                                        selector_walk_kernel<Staged>, args));
    } else {
        auto* edges = static_cast<float*>(workspace.edges.data);
        CUDA_CHECK(pdl::launch_consumer(
            {dim3(args.steps * candidate_ids.ne[2], kCandidates), dim3(512), 0, stream},
            selector_lattice_kernel<Staged>, args, edges));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(pdl::launch_consumer({dim3(candidate_ids.ne[2]), dim3(32), 0, stream},
                                        selector_lattice_walk_kernel, args, edges));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void candidate_selector_path_launch(SelectorPlan plan, const Tensor& candidate_ids,
                                    const Tensor& unary_scores, const Tensor& projected_hidden,
                                    const Tensor& anchors, const Weight& predecessor_codebook,
                                    const Weight& successor_codebook, const Tensor& base_positions,
                                    const SamplingConfig* configs, Tensor& drafts,
                                    Tensor& proposal_q, SelectorWorkspace& workspace,
                                    cudaStream_t stream) {
    DeviceArgs args{static_cast<const std::int32_t*>(candidate_ids.data),
                    static_cast<const float*>(unary_scores.data),
                    static_cast<const __nv_bfloat16*>(projected_hidden.data),
                    static_cast<const std::int32_t*>(anchors.data),
                    static_cast<const __nv_bfloat16*>(predecessor_codebook.qdata),
                    static_cast<const __nv_bfloat16*>(successor_codebook.qdata),
                    static_cast<const float*>(workspace.predecessor_rows.data),
                    static_cast<const float*>(workspace.successor_rows.data),
                    static_cast<const std::int32_t*>(base_positions.data),
                    configs,
                    static_cast<std::int32_t*>(drafts.data),
                    static_cast<float*>(proposal_q.data),
                    candidate_ids.ne[1],
                    candidate_ids.ne[2]};
    if (!plan.staged) {
        selector_launch<false>(plan.route, args, candidate_ids, workspace, stream);
        return;
    }
    // The round's own candidate rows, then the rows its predecessor walk can reach.
    const std::int32_t candidates = kCandidates * args.steps * args.batch;
    gather_rows(successor_codebook, candidate_ids, candidates, workspace.successor_rows, stream);
    const int predecessor_rows = selector_predecessor_rows(args.steps, args.batch);
    const dim3 ids_grid(args.batch, (predecessor_rows + 255) / 256);
    CUDA_CHECK(pdl::launch_consumer({ids_grid, dim3(256), 0, stream}, selector_predecessor_ids,
                                    static_cast<const std::int32_t*>(candidate_ids.data),
                                    static_cast<const std::int32_t*>(anchors.data), args.steps,
                                    args.batch,
                                    static_cast<std::int32_t*>(workspace.predecessor_ids.data)));
    CUDA_CHECK(cudaGetLastError());
    gather_rows(predecessor_codebook, workspace.predecessor_ids, predecessor_rows,
                workspace.predecessor_rows, stream);
    selector_launch<true>(plan.route, args, candidate_ids, workspace, stream);
}
} // namespace ninfer::ops::detail
