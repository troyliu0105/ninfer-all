#pragma once

#include "core/host_kv_arena.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::models::qwen3_5::detail {

// Host tier geometry of the hybrid prefix cache (docs/maintainer/hybrid-prefix-cache-spec.md
// §5.4). The whole Host budget is one pinned pool of equal slabs. One slab holds one KV block
// bundle: the Main text page packed with its Host page layout, then the backend page when a
// backend pool exists. A StateImage is split over `image_slabs` consecutive-in-image slabs and a
// snapshot tail bundle uses one more slab, so blocks and state images compete for the same bytes
// under the index's eviction policy instead of a fixed split.
struct HybridHostLayout {
    HostKVPageLayout text;
    std::optional<HostKVPageLayout> backend;
    std::size_t backend_offset = 0;
    std::size_t slab_bytes     = 0;
    // Bytes one block bundle actually transfers (the packed page records, without padding).
    std::size_t block_payload_bytes = 0;
    std::size_t image_bytes         = 0;
    std::uint32_t image_slabs       = 0;
};

[[nodiscard]] HybridHostLayout plan_hybrid_host_layout(const KVPageGeometry& text,
                                                       const KVPageGeometry* backend,
                                                       std::size_t image_bytes);

// Slabs a Host budget buys. A zero budget disables the Host tier. A nonzero budget must hold at
// least one complete snapshot (image and tail) and one block, or it is rejected: a smaller tier
// could never back a snapshot.
[[nodiscard]] std::uint32_t hybrid_host_slabs(const HybridHostLayout& layout,
                                              std::uint64_t budget_bytes);

// Block payloads the disk tier's staging may hold (docs/maintainer/hybrid-prefix-cache-disk-tier.md
// §3: "a bounded number of KV copy batches"). The tier's streaming cadence needs far fewer.
inline constexpr std::uint32_t kHybridDiskMaxStagingBlocks = 256;

// How the advertised Host budget splits between the disk tier's pinned staging and the slab pool
// (§3 budgets). Staging is one whole StateImage in its physical slab segments, one block payload
// for the snapshot tail, and up to `kHybridDiskMaxStagingBlocks` block payloads — but the block
// batch takes only what the budget leaves after the slab pool's own minimum geometry
// (`hybrid_host_slabs`), so enabling the disk tier never rejects a Host budget that satisfies the
// slab minimum on its own. `staging_blocks == 0` means the budget cannot hold a single block
// payload beside that minimum: the caller must reject the configuration rather than open a disk
// tier that could never stage a block read.
struct HybridDiskStagingPlan {
    std::uint32_t staging_blocks    = 0;
    std::uint64_t staging_bytes     = 0; // rounded physical bytes the tier pins
    std::uint64_t slab_budget_bytes = 0; // what remains for the slab pool
    // The smallest Host budget that yields one block payload of staging beside the slab pool's
    // own minimum geometry. A budget at or below it returns `staging_blocks == 0`, and the
    // caller reports this figure instead of re-deriving the geometry.
    std::uint64_t min_budget_bytes  = 0;
};

[[nodiscard]] HybridDiskStagingPlan plan_hybrid_disk_staging(const HybridHostLayout& layout,
                                                             std::uint64_t budget_bytes);

} // namespace ninfer::models::qwen3_5::detail
