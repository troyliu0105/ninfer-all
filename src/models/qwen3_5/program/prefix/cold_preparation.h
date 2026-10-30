#pragma once

#include "models/qwen3_5/program/prefix/hybrid_disk.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// FIFO-head preparation owns only bounded staging leases, never a lane or context transaction.
// Physical copies remain with the Program; this control path is also executable without CUDA.
class ColdPreparation final {
public:
    enum class State : std::uint8_t { Idle, Preparing, Ready, Fallback };
    using Block = HybridDiskTier::StagedBlock;
    struct Copies {
        void* context;
        std::uint64_t source_revision; // physical Host/Device copy completions, not just disk I/O
        std::uint32_t (*room)(void*);
        // Submit a prefix of ready blocks. Accepted buffers remain leased until their index
        // destinations leave Filling; the callback must not wait for an unfinished disk read.
        std::uint32_t (*submit)(void*, std::span<const Block>, std::span<const pc::NodeRef>);
        // Check image/tail readiness, starting reads only while the preparation budget remains.
        bool (*prepare_auxiliary)(void*, pc::SnapshotRef, bool allow_reads);
    };

    void select(std::uint64_t head, pc::SnapshotRef snapshot, std::span<const pc::NodeRef> path,
                std::uint64_t now_ns, std::uint64_t budget_ns, HybridDiskTier& disk);
    [[nodiscard]] bool progress(pc::PrefixCacheIndex& index, HybridDiskTier& disk,
                                 std::uint64_t now_ns, const Copies& copies);
    void abandon(std::uint64_t head, HybridDiskTier& disk);
    // The physical owner must first settle all accepted I/O and CUDA copies.
    void clear_after_settlement(HybridDiskTier& disk);
    [[nodiscard]] State state() const noexcept { return state_; }
    [[nodiscard]] bool owns(std::uint64_t head) const noexcept { return head_ == head && head != 0; }
    [[nodiscard]] std::uint64_t copied_blocks() const noexcept { return copied_blocks_; }
    [[nodiscard]] bool pending(const HybridDiskTier& disk) const noexcept;
    [[nodiscard]] pc::SnapshotRef snapshot() const noexcept { return snapshot_; }

private:
    void release_unused(HybridDiskTier& disk);
    [[nodiscard]] bool contains(pc::NodeRef node) const;
    std::uint64_t head_ = 0;
    pc::SnapshotRef snapshot_;
    std::vector<pc::NodeRef> path_;
    std::vector<Block> held_;
    std::vector<Block> inflight_;
    std::vector<Block> released_; // reusable non-owning scratch
    std::vector<pc::NodeRef> cold_;
    std::uint64_t copied_blocks_ = 0;
    std::uint64_t deadline_ns_ = 0;
    std::uint64_t observed_completion_ = 0;
    std::uint64_t observed_source_revision_ = 0;
    std::uint32_t observed_room_ = 0;
    bool dirty_ = false;
    State state_ = State::Idle;
};

} // namespace ninfer::models::qwen3_5::detail
