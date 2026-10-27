#include "models/qwen3_5/program/prefix/hybrid_host_layout.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::detail {
namespace {

// Slab records start on page boundaries so every DMA run begins aligned.
constexpr std::size_t kSlabAlignment = 4096;

std::size_t align_slab(std::size_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() - (kSlabAlignment - 1U)) {
        throw std::overflow_error("hybrid Host slab size overflows");
    }
    return (bytes + kSlabAlignment - 1U) / kSlabAlignment * kSlabAlignment;
}

} // namespace

HybridHostLayout plan_hybrid_host_layout(const KVPageGeometry& text, const KVPageGeometry* backend,
                                         std::size_t image_bytes) {
    if (image_bytes == 0) { throw std::invalid_argument("hybrid Host layout has no StateImage"); }
    HybridHostLayout out;
    out.text                = plan_host_kv_page_layout(text);
    std::size_t bytes       = out.text.page_stride;
    out.block_payload_bytes = out.text.page_stride;
    if (backend != nullptr) {
        out.backend        = plan_host_kv_page_layout(*backend);
        out.backend_offset = align_slab(bytes);
        bytes              = out.backend_offset + out.backend->page_stride;
        out.block_payload_bytes += out.backend->page_stride;
    }
    out.slab_bytes          = align_slab(bytes);
    out.image_bytes         = image_bytes;
    const std::size_t slabs = 1U + (image_bytes - 1U) / out.slab_bytes;
    if (slabs > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("hybrid StateImage slab count exceeds uint32");
    }
    out.image_slabs = static_cast<std::uint32_t>(slabs);
    return out;
}

std::uint32_t hybrid_host_slabs(const HybridHostLayout& layout, std::uint64_t budget_bytes) {
    if (budget_bytes == 0) { return 0; }
    if (layout.slab_bytes == 0) { throw std::logic_error("hybrid Host layout is empty"); }
    const std::uint64_t slabs   = budget_bytes / layout.slab_bytes;
    const std::uint64_t minimum = static_cast<std::uint64_t>(layout.image_slabs) + 2U;
    if (slabs < minimum) {
        const std::uint64_t minimum_mib = (minimum * layout.slab_bytes + (1ULL << 20U) - 1U) >> 20U;
        throw std::invalid_argument(
            "--host-cache-mib is too small for the hybrid prefix cache: one state snapshot needs " +
            std::to_string(minimum_mib) + " MiB; use at least that or 0 to disable the Host tier");
    }
    if (slabs > std::numeric_limits<std::uint32_t>::max() / 2U) {
        throw std::overflow_error("hybrid Host slab count is not representable");
    }
    return static_cast<std::uint32_t>(slabs);
}

HybridDiskStagingPlan plan_hybrid_disk_staging(const HybridHostLayout& layout,
                                               std::uint64_t budget_bytes) {
    HybridDiskStagingPlan plan;
    if (budget_bytes == 0 || layout.slab_bytes == 0 || layout.block_payload_bytes == 0 ||
        layout.image_slabs == 0) {
        return plan;
    }
    const std::uint64_t image_segments =
        static_cast<std::uint64_t>(layout.image_slabs) * layout.slab_bytes;
    const std::uint64_t tail        = layout.block_payload_bytes;
    const std::uint64_t slab_minimum =
        (static_cast<std::uint64_t>(layout.image_slabs) + 2U) * layout.slab_bytes;
    // One block payload of staging is the floor: with fewer the tier could never stage a read.
    plan.min_budget_bytes = image_segments + tail + slab_minimum + layout.block_payload_bytes;
    if (budget_bytes <= image_segments + tail) { return plan; }
    const std::uint64_t after_image = budget_bytes - image_segments - tail;
    if (after_image <= slab_minimum) { return plan; }
    const std::uint64_t block_space = after_image - slab_minimum;
    plan.staging_blocks             = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        kHybridDiskMaxStagingBlocks, block_space / layout.block_payload_bytes));
    plan.staging_bytes =
        image_segments + tail +
        static_cast<std::uint64_t>(plan.staging_blocks) * layout.block_payload_bytes;
    plan.slab_budget_bytes = budget_bytes - plan.staging_bytes;
    return plan;
}

HybridDiskBudget hybrid_disk_budget(std::uint64_t capacity_bytes) {
    if (capacity_bytes == 0) { capacity_bytes = std::uint64_t{64} << 30U; }
    HybridDiskBudget budget;
    budget.block_bytes = capacity_bytes / 2U;
    budget.image_bytes = capacity_bytes - budget.block_bytes;
    return budget;
}

HybridDiskRecords plan_hybrid_disk_records(const HybridHostLayout& layout,
                                           std::uint64_t capacity_bytes) {
    HybridDiskRecords records;
    if (layout.block_payload_bytes == 0 || layout.slab_bytes == 0 || layout.image_slabs == 0) {
        throw std::logic_error("hybrid Host layout is empty");
    }
    // The index represents at most kNoId/2 of either descriptor; the same bound its constructor
    // enforces, so the caller can sum the residencies without a second representability rule.
    const std::uint64_t representable = std::numeric_limits<std::uint32_t>::max() / 2U;
    const HybridDiskBudget budget     = hybrid_disk_budget(capacity_bytes);
    const std::uint64_t blocks = budget.block_bytes / layout.block_payload_bytes;
    const std::uint64_t images = budget.image_bytes / layout.slab_bytes / layout.image_slabs;
    if (blocks > representable || images > representable) {
        throw std::overflow_error("hybrid disk record capacity is not representable");
    }
    records.blocks = static_cast<std::uint32_t>(blocks);
    records.images = static_cast<std::uint32_t>(images);
    return records;
}

} // namespace ninfer::models::qwen3_5::detail
