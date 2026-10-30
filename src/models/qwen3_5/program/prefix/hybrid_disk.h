#pragma once

// Disk orchestration for the hybrid prefix cache. Buffers and physical transfers are supplied by
// the Program; this state machine and its storage executor have no CUDA dependency.
#include "runtime/prefix_cache/disk_worker.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

namespace pc = runtime::prefix_cache;

class HybridDiskTier final {
public:
    struct Layout {
        std::size_t slab_bytes = 0;
        std::size_t block_payload_bytes = 0;
        std::uint32_t image_slabs = 0;
    };
    // Borrowed for the tier's entire lifetime, including shutdown settlement.
    struct Staging {
        std::span<std::byte> blocks;
        std::span<std::byte> image;
        std::span<std::byte> tail;
    };
    // Synchronous requests to the physical owner. Accepted copies retain their index pins until
    // that owner publishes completion; submitting a request never means its bytes are ready.
    struct HostTransfers {
        void* context;
        const std::byte* (*slab_bytes_of)(void*, std::uint32_t);
        bool (*start_snapshot_host_write)(void*, pc::SnapshotRef);
        void (*start_block_host_write)(void*, pc::NodeRef);
    };
    struct Options {
        std::filesystem::path directory;
        std::uint64_t block_capacity_bytes = 0;
        std::uint64_t image_capacity_bytes = 0;
        double host_high_watermark = 0.75;
        double host_low_watermark = 0.50;
        std::uint32_t demotion_batch_records = 16;
        std::uint32_t staging_blocks = 0;
        // Accepted write source bytes, including running and unadopted completions. Zero derives
        // three physical image-and-tail jobs; every individual submission respects this bound.
        std::uint64_t max_inqueue_bytes = 0;
    };
    struct Stats {
        std::uint64_t demoted_blocks = 0;
        std::uint64_t demoted_images = 0;
        std::uint64_t committed_closures = 0;
        std::uint64_t failed_closures = 0;
        std::uint64_t disk_read_bytes = 0;
        std::uint64_t disk_write_bytes = 0;
        std::uint64_t restored_blocks = 0;
        std::uint64_t queue_rejects = 0;
        std::uint64_t released_records = 0;
        std::uint64_t evicted_records = 0;
        std::uint32_t pending_removals = 0;
        std::uint32_t catalog_blocks = 0;
        std::uint32_t catalog_committed = 0;
    };
    HybridDiskTier(Options options, Layout layout, Staging staging, const std::string& fingerprint);
    ~HybridDiskTier();
    HybridDiskTier(const HybridDiskTier&) = delete;
    HybridDiskTier& operator=(const HybridDiskTier&) = delete;

    struct RestoreSummary {
        bool ok = false;
        std::uint64_t blocks = 0;
        std::uint64_t snapshots = 0;
        std::uint64_t kv_bytes = 0;
    };
    // Startup only: replay before starting the I/O executor.
    [[nodiscard]] RestoreSummary restore_into(pc::PrefixCacheIndex& index);
    void maybe_demote(pc::PrefixCacheIndex& index, const HostTransfers& transfers);
    // One Engine cadence: adopt completed I/O, then continue bounded retention. Returns
    // whether completions were adopted; an unchanged poll must not re-arm admission.
    bool poll(pc::PrefixCacheIndex& index, const HostTransfers& transfers);
    bool publish_completions(pc::PrefixCacheIndex& index);
    // Failure/rebuild barrier, never part of serving admission or transfer draining.
    void settle_sources(pc::PrefixCacheIndex& index) {
        worker_->wait_idle();
        publish_completions(index);
    }
    [[nodiscard]] bool busy() const noexcept;
    void shutdown() noexcept;
    [[nodiscard]] Stats stats() const noexcept;
    [[nodiscard]] std::uint64_t completion_revision() const noexcept {
        return worker_ ? worker_->completion_revision() : 0;
    }
    [[nodiscard]] bool reads_pending() const noexcept {
        return !reading_blocks_.empty() || reading_image_.has_value() ||
               reading_tail_.has_value() || !staged_blocks_.empty();
    }
    [[nodiscard]] std::size_t block_payload_bytes() const noexcept { return host_.block_payload_bytes; }

    [[nodiscard]] bool stage_blocks(const std::vector<pc::NodeRef>& nodes,
                                    const pc::PrefixCacheIndex& index);
    struct StagedBlock {
        pc::NodeRef node;
        const std::byte* bytes = nullptr;
        std::uint64_t lease = 0; // protects a reused physical slot from a stale release
    };
    [[nodiscard]] std::vector<StagedBlock> take_staged_blocks();
    void release_staged(const std::vector<StagedBlock>& blocks);
    [[nodiscard]] bool stage_image(pc::SnapshotRef snapshot, const pc::PrefixCacheIndex& index);
    [[nodiscard]] const std::byte* take_staged_image(pc::SnapshotRef snapshot);
    void release_staged_image_except(pc::SnapshotRef snapshot);
    void release_image();
    [[nodiscard]] const std::byte* staged_tail(pc::SnapshotRef snapshot) const noexcept;
    void release_staged_tail_except(pc::SnapshotRef snapshot);
    [[nodiscard]] bool stage_tail(pc::SnapshotRef snapshot, const pc::PrefixCacheIndex& index);
    void release_tail();
    [[nodiscard]] std::uint64_t staging_bytes() const noexcept;

    // Nonblocking, idempotent submission. Slots become reusable only after the worker has
    // journaled the removal; a failed removal is resubmitted on the next completion cadence.
    void release_disk_record(std::uint32_t record) noexcept;
    void release_snapshot_disk_records(std::uint32_t record) noexcept;

private:
    void release_record(std::uint32_t record) noexcept;
    void retry_removals();
    bool reclaim_records(pc::PrefixCacheIndex& index);
    void free_staging_slot(const std::byte* bytes, std::uint64_t lease) noexcept;
    [[nodiscard]] std::byte* block_staging(std::uint32_t slot) const noexcept;

    Options options_;
    Layout host_;
    std::unique_ptr<pc::PrefixDiskCatalog> catalog_;
    std::unique_ptr<pc::PrefixDiskWorker> worker_;
    Staging staging_;
    std::vector<std::uint32_t> free_staging_;
    std::vector<std::uint64_t> staging_leases_; // zero: free; otherwise the current source owner
    std::uint64_t next_staging_lease_ = 1;
    std::vector<pc::NodeRef> reading_blocks_;
    std::optional<pc::SnapshotRef> reading_image_;
    std::optional<pc::SnapshotRef> reading_tail_;
    std::vector<StagedBlock> staged_blocks_;
    pc::SnapshotRef staged_image_;
    pc::SnapshotRef staged_tail_;
    std::vector<std::uint32_t> pending_removals_;
    // Bounded, unpinned interests: finish started closures below the low watermark too.
    std::vector<pc::SnapshotRef> demotion_targets_;
    std::vector<pc::SnapshotRef> demotion_candidates_;
    std::vector<pc::NodeRef> demotion_path_;
    Stats stats_;
    bool stopping_ = false;
    bool demoting_ = false;
};

} // namespace ninfer::models::qwen3_5::detail
