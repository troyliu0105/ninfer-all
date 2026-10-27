#include "ops/candidate_selector/bf16/candidate_selector_path_plan.h"
#include "ops/candidate_selector/bf16/candidate_selector_path_kernels.h"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

void require_codebook(const Weight& codebook, const char* label) {
    constexpr std::int32_t kRank = 256;
    if (codebook.ndim != 2 || codebook.n <= 0 || codebook.k != kRank ||
        codebook.shape[0] != codebook.n || codebook.shape[1] != kRank || codebook.qdata == nullptr) {
        throw std::invalid_argument(std::string("candidate_selector_path: invalid ") + label);
    }
    if (is_gguf(codebook.qtype)) {
        const auto block = gguf_block_shape(codebook.qtype);
        if (codebook.layout != QuantLayout::GgufBlocks || block.elements == 0 ||
            codebook.k % block.elements != 0 || codebook.k % 32 != 0) {
            throw std::invalid_argument(std::string("candidate_selector_path: invalid ") + label);
        }
        return;
    }
    if (codebook.qtype != QType::BF16 || codebook.layout != QuantLayout::Contiguous ||
        codebook.payload_bytes < std::uint64_t(codebook.n) * kRank * sizeof(std::uint16_t)) {
        throw std::invalid_argument(std::string("candidate_selector_path: invalid ") + label);
    }
}

std::size_t row_bytes(const Weight& codebook) {
    if (codebook.qtype == QType::BF16) { return std::uint64_t(codebook.k) * sizeof(std::uint16_t); }
    const auto block = gguf_block_shape(codebook.qtype);
    return std::uint64_t(codebook.k / block.elements) * block.bytes;
}

} // namespace

bool selector_bf16_rows(const Weight& codebook) {
    return codebook.qtype == QType::BF16 && codebook.layout == QuantLayout::Contiguous;
}

int selector_predecessor_rows(int steps, int batch) {
    return batch * (1 + 16 * (steps - 1));
}

SelectorPlan candidate_selector_path_plan(int steps, int batch, bool staged) {
    if (steps < 1 || steps > 15 || batch < 1 || batch > 8)
        throw std::invalid_argument("invalid selector K/B");
    return {steps <= 4 ? SelectorRoute::Direct : SelectorRoute::Lattice, staged};
}

const char* selector_route_name(SelectorPlan plan) {
    if (plan.route == SelectorRoute::Direct) {
        return plan.staged ? "direct.t512.staged_rows" : "direct.t512";
    }
    return plan.staged ? "lattice.t512.staged_rows" : "lattice.t512";
}

const char* candidate_selector_path_route_name(int steps, int batch, bool staged) {
    return selector_route_name(candidate_selector_path_plan(steps, batch, staged));
}

void candidate_selector_path_dispatch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                      const Tensor& projected_hidden, const Tensor& anchors,
                                      const Weight& predecessor_codebook,
                                      const Weight& successor_codebook,
                                      const Tensor& base_positions, const SamplingConfig* configs,
                                      Tensor& drafts, Tensor& proposal_q, WorkspaceArena& workspace,
                                      cudaStream_t stream) {
    require_codebook(predecessor_codebook, "predecessor_codebook");
    require_codebook(successor_codebook, "successor_codebook");
    const int steps = candidate_ids.ne[1], batch = candidate_ids.ne[2];
    const SelectorPlan plan = candidate_selector_path_plan(
        steps, batch, !selector_bf16_rows(predecessor_codebook) ||
                           !selector_bf16_rows(successor_codebook));
    auto scope        = workspace.scope();
    auto scratch       = allocate_selector_workspace(workspace, plan, steps, batch);
    const std::array<const Tensor*, 10> live{&candidate_ids, &unary_scores, &projected_hidden,
                                             &anchors,        &base_positions, &drafts,
                                             &proposal_q};
    const std::pair<const Tensor*, const char*> planes[] = {
        {&scratch.edges, "edges"}, {&scratch.successor_rows, "successor rows"},
        {&scratch.predecessor_rows, "predecessor rows"},
        {&scratch.predecessor_ids, "predecessor ids"}};
    for (const auto& [work, plane] : planes) {
        if (!work->data) continue;
        const auto begin = reinterpret_cast<std::uintptr_t>(work->data),
                   end   = begin + work->bytes();
        for (std::size_t index = 0; index < live.size(); ++index) {
            const auto* tensor = live[index];
            if (tensor == nullptr || !tensor->data) continue;
            const auto tb = reinterpret_cast<std::uintptr_t>(tensor->data);
            if (begin < tb + tensor->bytes() && tb < end)
                throw std::invalid_argument(std::string("selector workspace plane ") + plane +
                                            " overlaps operand " + std::to_string(index));
        }
        for (const auto* codebook : {&predecessor_codebook, &successor_codebook}) {
            const auto cb = reinterpret_cast<std::uintptr_t>(codebook->qdata);
            if (begin < cb + row_bytes(*codebook) * std::uint64_t(codebook->n) && cb < end)
                throw std::invalid_argument(std::string("selector workspace plane ") + plane +
                                            " overlaps a codebook");
        }
        const auto cb = reinterpret_cast<std::uintptr_t>(configs);
        if (begin < cb + batch * sizeof(SamplingConfig) && cb < end)
            throw std::invalid_argument(std::string("selector workspace plane ") + plane +
                                        " overlaps configs");
    }
    candidate_selector_path_launch(plan, candidate_ids, unary_scores, projected_hidden, anchors,
                                   predecessor_codebook, successor_codebook, base_positions,
                                   configs, drafts, proposal_q, scratch, stream);
}
} // namespace ninfer::ops::detail
