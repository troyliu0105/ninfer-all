// The hybrid prefix cache's disk tier binding — see hybrid_disk.h and the A4 design note in
// local://hybrid-disk-tier-plan.md.
//
// The two threads: the Engine worker picks victims, snapshots their metadata and Host slab
// pointers into immutable jobs, and pins the sources; the I/O worker only moves bytes between
// those pinned slabs and the catalog. Completions return to the Engine worker, which alone
// runs finish_block/commit_snapshot and mutates the index. A closure commits only once every
// one of its block records and its image record landed; a closure whose block write or anchor
// record was lost is never committed, and its image record is reclaimed only after its own
// write returned.
//
// The tier also owns the catalog's reclamation side: the index's release callbacks arrive here
// (release_disk_record/release_snapshot_disk_records) and journal a removal, so a restart never
// resurrects a record the running cache dropped, and a full catalog is recycled with the
// index's bounded disk victim selection instead of failing every later demotion.

#include "models/qwen3_5/program/prefix/hybrid_disk.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ninfer::models::qwen3_5::detail {

namespace {

using pc::CopyState;
using pc::NodeRef;
using pc::NodeView;
using pc::PrefixCacheIndex;
using pc::PrefixDiskCatalog;
using pc::SnapshotRef;
using pc::SnapshotView;

// Disk records a full catalog reclaims in one bounded step: the lowest-value committed closures
// first, then the exclusive blocks they kept (index.evict_disk_entries), plus whatever reserved
// records no write is going to finish (catalog.reclaim_abandoned).
constexpr std::uint32_t kReclaimRecords = 8U;

} // namespace

HybridDiskTier::HybridDiskTier(Options options, const HybridHostLayout& host,
                               const std::string& fingerprint)
    : options_(std::move(options)), host_(host) {
    if (options_.directory.empty()) {
        throw std::invalid_argument("hybrid disk tier directory is empty");
    }
    if (options_.staging_blocks == 0 || host_.block_payload_bytes == 0 ||
        host_.image_bytes == 0 || host_.image_slabs == 0 || host_.slab_bytes == 0) {
        throw std::invalid_argument("hybrid disk tier requires a positive geometry");
    }
    if (!(options_.host_low_watermark >= 0.0 && options_.host_low_watermark <
                                                   options_.host_high_watermark &&
          options_.host_high_watermark <= 1.0)) {
        throw std::invalid_argument("hybrid disk tier Host watermarks are inconsistent");
    }
    // One image segment per slab: a segment write is exactly the slab payload the pin holds.
    PrefixDiskCatalog::Geometry geometry;
    geometry.block_payload_bytes = host_.block_payload_bytes;
    geometry.image_segment_bytes = host_.slab_bytes;
    geometry.image_segments      = host_.image_slabs;
    PrefixDiskCatalog::Options catalog_options;
    catalog_options.directory            = options_.directory;
    catalog_options.block_capacity_bytes = options_.block_capacity_bytes;
    catalog_options.image_capacity_bytes = options_.image_capacity_bytes;
    catalog_ = std::make_unique<PrefixDiskCatalog>(std::move(catalog_options), geometry,
                                                   fingerprint);
    // The staging is pinned transfer storage, not a second loose RAM cache: the Program sized it
    // out of the advertised Host budget (plan_hybrid_disk_staging) before sizing the slab pool.
    // The disk stores whole slabs (image_slabs of slab_bytes); only the first image_bytes are
    // the StateImage, but the staging buffer must hold every physical segment a read writes.
    try {
        block_staging_.emplace(static_cast<std::size_t>(options_.staging_blocks) *
                               host_.block_payload_bytes);
        image_staging_.emplace(static_cast<std::size_t>(host_.image_slabs) * host_.slab_bytes);
        tail_staging_.emplace(host_.block_payload_bytes);
    } catch (const std::exception& error) {
        throw std::runtime_error(
            std::string("pinning the hybrid disk tier's staging failed (it is accounted inside "
                        "the hybrid Host budget): ") +
            error.what());
    }
    free_staging_.resize(options_.staging_blocks);
    staging_slot_free_.assign(options_.staging_blocks, true);
    for (std::uint32_t i = 0; i < options_.staging_blocks; ++i) {
        free_staging_[i] = options_.staging_blocks - 1U - i;
    }
    worker_ = std::thread([this] { worker_loop(); });
}

HybridDiskTier::~HybridDiskTier() { shutdown(); }

bool HybridDiskTier::restore_into(PrefixCacheIndex& index) {
    if (!catalog_->open()) {
        catalog_->wipe();
        return false;
    }
    // Replayed ids are parents-first by construction: a record's finish_block preceded any
    // dependent commit, and the catalog replays in id order.
    std::unordered_map<std::uint32_t, NodeRef> restored;
    for (const PrefixDiskCatalog::ReplayedBlock& block : catalog_->replayed_blocks()) {
        NodeRef parent;
        if (block.identity.parent != PrefixDiskCatalog::kNoRecord) {
            const auto found = restored.find(block.identity.parent);
            if (found == restored.end()) { continue; } // torn dependency: skip this subtree
            parent = found->second;
        }
        const auto node = index.restore_disk_block(
            parent, block.identity.lookup_hash,
            std::span<const TokenId>(block.identity.tokens.data(),
                                     block.identity.tokens.size()),
            block.identity.extra, block.id);
        if (!node) { break; } // node capacity: deeper entries cannot restore either
        restored.emplace(block.id, *node);
    }
    for (const PrefixDiskCatalog::ReplayedSnapshot& snapshot : catalog_->replayed_snapshots()) {
        NodeRef anchor;
        if (snapshot.identity.anchor != PrefixDiskCatalog::kNoRecord) {
            const auto found = restored.find(snapshot.identity.anchor);
            if (found == restored.end()) {
                continue;
            }
            anchor = found->second;
        }
        (void)index.restore_disk_snapshot(
            anchor, snapshot.identity.frontier,
            std::span<const TokenId>(snapshot.identity.tail.data(), snapshot.identity.tail_len),
            snapshot.identity.kind, snapshot.identity.hits, snapshot.id, true);
    }
    return true;
}

// ---- record reclamation (Engine worker) ------------------------------------------------------------

void HybridDiskTier::release_record(std::uint32_t record) noexcept {
    if (record == PrefixDiskCatalog::kNoRecord) { return; }
    bool removed = false;
    try {
        catalog_->remove(record);
        removed = true;
    } catch (...) {
        removed = false;
    }
    std::lock_guard lock(mutex_);
    // Either way the record must not be named by a later commit: a failed removal is owed back
    // to the catalog at the next cadence point, and never resurrected at restart.
    removed_records_.push_back(record);
    if (removed) {
        ++stats_.released_records;
    } else {
        pending_removals_.push_back(record);
    }
    stats_.pending_removals = static_cast<std::uint32_t>(pending_removals_.size());
}

void HybridDiskTier::retry_removals() {
    std::vector<std::uint32_t> owed;
    {
        std::lock_guard lock(mutex_);
        owed.swap(pending_removals_);
    }
    std::vector<std::uint32_t> still;
    for (const std::uint32_t record : owed) {
        try {
            catalog_->remove(record);
            std::lock_guard lock(mutex_);
            ++stats_.released_records;
        } catch (...) { still.push_back(record); }
    }
    std::lock_guard lock(mutex_);
    pending_removals_.insert(pending_removals_.end(), still.begin(), still.end());
    stats_.pending_removals = static_cast<std::uint32_t>(pending_removals_.size());
    if (!still.empty()) {
        std::fprintf(stderr, "hybrid disk tier: %zu catalog removals could not be journaled\n",
                     still.size());
    }
}

void HybridDiskTier::release_disk_record(std::uint32_t record) noexcept {
    release_record(record);
}

void HybridDiskTier::release_snapshot_disk_records(std::uint32_t record) noexcept {
    // A snapshot record carries its image segments and its optional tail; remove() drops them
    // together. The anchor-path block records arrive through release_disk_record.
    release_record(record);
}

// ---- demotion (Engine worker) ----------------------------------------------------------------------

void HybridDiskTier::maybe_demote(PrefixCacheIndex& index, const HybridPrefixCache& cache) {
    retry_removals();
    if (!cache.host_tier()) { return; }
    const std::uint32_t slabs = index.config().host_slabs;
    if (slabs == 0) { return; }
    const auto occupied_slabs = [&] {
        const pc::PrefixIndexStats current = index.stats();
        return slabs - current.host_free_slabs;
    };
    const double high = static_cast<double>(slabs) * options_.host_high_watermark;
    const double low  = static_cast<double>(slabs) * options_.host_low_watermark;
    if (static_cast<double>(occupied_slabs()) < high) {
        return; // below the high watermark
    }
    const auto candidates = index.collect_demotion_candidates(options_.demotion_batch_records);
    for (const SnapshotRef snapshot : candidates) {
        if (static_cast<double>(occupied_slabs()) < low) {
            break; // the pressure cleared: stop selecting more work
        }
        {
            std::lock_guard lock(mutex_);
            if (inqueue_bytes_ >= options_.max_inqueue_bytes) {
                ++stats_.queue_rejects;
                return; // backpressure: stop submitting, never block
            }
        }
        const SnapshotView view = index.snapshot(snapshot);
        if (view.host != CopyState::Resident) { continue; }
        // The closure copies exactly the snapshot's Host slabs: the image grid and, when a tail
        // exists, its slab. An incomplete Host copy (a fill aborted under pressure, or a tail the
        // Host tier never allocated) could not produce a complete disk closure.
        const std::size_t closure_slabs = host_.image_slabs + (view.tail_len != 0 ? 1U : 0U);
        if (view.host_slabs.size() != closure_slabs) { continue; }

        PendingClosure closure;
        closure.snapshot = snapshot;
        closure.identity = PrefixDiskCatalog::SnapIdentityRec{
            .anchor   = PrefixDiskCatalog::kNoRecord,
            .frontier = view.frontier,
            .tail_len = view.tail_len,
            .tail     = {},
            .kind     = view.kind,
            .hits     = view.hits};
        std::copy(view.tail.begin(), view.tail.end(), closure.identity.tail.begin());

        // The anchor chain root→anchor; each block without a disk record becomes a write job.
        std::vector<NodeRef> chain;
        for (NodeRef walk = view.anchor; walk.valid(); walk = index.node(walk).parent) {
            chain.push_back(walk);
        }
        std::reverse(chain.begin(), chain.end());
        bool abort_closure = false;
        index.pin_snapshot(snapshot);
        for (const NodeRef node : chain) {
            closure.chain_nodes.push_back(node);
            const NodeView node_view = index.node(node);
            if (node_view.disk == CopyState::Resident || node_view.disk == CopyState::Filling) {
                // Already disk-backed (or a sibling closure's write is in flight): the anchor
                // chain keeps those records without recopying. An in-flight write is a real
                // dependency — it stays in the wait set so a sibling failure fails this
                // closure too; a committed record never completes again and must not.
                if (node_view.disk_ref != pc::kNoId) {
                    closure.chain_records.push_back(node_view.disk_ref);
                    if (node_view.disk == CopyState::Filling) {
                        closure.records.push_back(node_view.disk_ref);
                    }
                }
                continue;
            }
            if (node_view.host != CopyState::Resident) {
                abort_closure = true; // no Host source to read
                break;
            }
            const pc::BlockIdentity identity = index.block_identity(node);
            const std::uint32_t parent_record =
                closure.chain_records.empty() ? PrefixDiskCatalog::kNoRecord
                                              : closure.chain_records.back();
            if (const auto existing = catalog_->find_block(parent_record, identity.lookup_hash,
                                                           identity.tokens, identity.extra)) {
                closure.chain_records.push_back(*existing);
                continue; // a committed ancestor is reused, not recopied
            }
            std::uint32_t record = catalog_->begin_block_write();
            if (record == PrefixDiskCatalog::kNoRecord) {
                // The record directory is full: recycle the lowest-value committed closures and
                // collect reserved records nothing will finish, then try once more before
                // treating this cycle as backpressure (docs §4.2 step 1/3).
                reclaim_records(index);
                record = catalog_->begin_block_write();
            }
            if (record == PrefixDiskCatalog::kNoRecord) {
                abort_closure = true; // still full: stop demoting this cycle
                break;
            }
            index.pin_node(node);
            index.begin_node_disk_fill(node, record);
            BlockJob job;
            job.node     = node;
            job.record   = record;
            job.identity = PrefixDiskCatalog::BlockIdentityRec{
                .parent      = parent_record,
                .lookup_hash = identity.lookup_hash,
                .extra       = identity.extra,
                .tokens      = {}};
            std::copy(identity.tokens.begin(), identity.tokens.end(),
                      job.identity.tokens.begin());
            job.source = cache.slab_bytes_of(node_view.host_slab);
            {
                std::lock_guard lock(mutex_);
                write_blocks_.push_back(job);
                inqueue_bytes_ += host_.block_payload_bytes;
                wake_.notify_one();
            }
            closure.records.push_back(record);
            closure.chain_records.push_back(record);
        }
        if (abort_closure) {
            // Queued write jobs keep their pins: publish_completions unpins each node as its
            // write lands, so unpinning here would release a pin the job still owns.
            index.unpin_snapshot(snapshot);
            continue;
        }

        // The image (and tail) job; the snapshot stays pinned until the closure commits.
        std::uint32_t record = catalog_->begin_snapshot_write();
        if (record == PrefixDiskCatalog::kNoRecord) {
            reclaim_records(index);
            record = catalog_->begin_snapshot_write();
        }
        if (record == PrefixDiskCatalog::kNoRecord) {
            // Same ownership as the abort path: the queued writes release their own pins.
            index.unpin_snapshot(snapshot);
            std::lock_guard lock(mutex_);
            ++stats_.queue_rejects;
            continue;
        }
        index.begin_snapshot_disk_fill(snapshot, record);
        ImageJob job;
        job.snapshot = snapshot;
        job.record   = record;
        for (const std::uint32_t slab : view.host_slabs) {
            job.segments.push_back(cache.slab_bytes_of(slab));
        }
        // The tail bundle is the snapshot's last host slab (blocks and images share the pool and
        // the tail occupies the slab after the image run); the slab guard above proved it exists.
        if (view.tail_len != 0) { job.tail = cache.slab_bytes_of(view.host_slabs.back()); }
        job.identity = closure.identity;
        // The committed identity names the chain's last record: the deepest block this
        // closure named, whether it wrote the record or a sibling's write lands it.
        job.identity.anchor = closure.chain_records.empty()
                                  ? PrefixDiskCatalog::kNoRecord
                                  : closure.chain_records.back();
        closure.identity.anchor = job.identity.anchor;
        closure.image_record = record;
        {
            std::lock_guard lock(mutex_);
            write_images_.push_back(std::move(job));
            inqueue_bytes_ += host_.image_bytes;
            wake_.notify_one();
        }
        pending_closures_.push_back(std::move(closure));
    }
}

void HybridDiskTier::publish_completions(PrefixCacheIndex& index) {
    retry_removals();
    std::vector<CompletedBlock> blocks;
    std::vector<CompletedImage> images;
    {
        std::lock_guard lock(mutex_);
        blocks = std::move(completed_blocks_);
        images = std::move(completed_images_);
        completed_blocks_.clear();
        completed_images_.clear();
    }
    // Block records: finish, mark the node disk-backed, release the source pin.
    for (const CompletedBlock& done : blocks) {
        const bool node_ok = index.valid(done.node);
        if (!done.ok || !node_ok) {
            if (node_ok) {
                index.unpin_node(done.node);
                if (index.node(done.node).disk == CopyState::Filling) {
                    index.abort_node_disk_fill(done.node);
                }
            }
            if (done.record != PrefixDiskCatalog::kNoRecord) {
                // The write returned: the record's payload slot is free to reclaim. A closure
                // that still awaits this record can never commit, but its own image record stays
                // reserved until that write returns too (see below).
                try {
                    catalog_->abandon(done.record);
                } catch (...) { release_record(done.record); }
                for (PendingClosure& closure : pending_closures_) {
                    if (std::find(closure.records.begin(), closure.records.end(), done.record) !=
                        closure.records.end()) {
                        closure.failed = true;
                    }
                }
            }
            continue;
        }
        try {
            catalog_->finish_block(done.record, done.identity);
        } catch (...) {
            index.unpin_node(done.node);
            if (index.node(done.node).disk == CopyState::Filling) {
                index.abort_node_disk_fill(done.node);
            }
            release_record(done.record);
            for (PendingClosure& closure : pending_closures_) {
                if (std::find(closure.records.begin(), closure.records.end(), done.record) !=
                    closure.records.end()) {
                    closure.failed = true;
                }
            }
            continue;
        }
        try {
            index.mark_disk_backed(done.node, done.record);
        } catch (...) {
            // The index no longer expects a fill for this node: the finished record has no
            // owner in the running cache, so drop it rather than leave an orphan a restart
            // would replay, and leave the node disk-Absent rather than Filling for ever.
            if (index.node(done.node).disk == CopyState::Filling) {
                index.abort_node_disk_fill(done.node);
            }
            index.unpin_node(done.node);
            release_record(done.record);
            continue;
        }
        {
            std::lock_guard lock(mutex_);
            ++stats_.demoted_blocks;
            stats_.disk_write_bytes += host_.block_payload_bytes;
        }
        index.unpin_node(done.node);
    }
    // Image records: their payload landed (or failed); a closure's own record is only reclaimed
    // here, after its write returned, so a queued write never recreates a deleted payload.
    for (const CompletedImage& done : images) {
        const auto found = std::find_if(pending_closures_.begin(), pending_closures_.end(),
                                        [&](const PendingClosure& closure) {
                                            return closure.image_record == done.record;
                                        });
        if (found == pending_closures_.end()) {
            // No closure owns this record: the index dropped the snapshot before the write
            // returned. Abandon it now that the write is done.
            if (done.record != PrefixDiskCatalog::kNoRecord) {
                try {
                    catalog_->abandon(done.record);
                } catch (...) { release_record(done.record); }
            }
            continue;
        }
        PendingClosure& closure = *found;
        const bool snapshot_ok  = index.valid(closure.snapshot);
        if (!done.ok || closure.failed || !snapshot_ok) {
            if (snapshot_ok) {
                // Releases the image record through the backend's release callback.
                index.abort_snapshot_disk_fill(closure.snapshot);
            } else if (done.record != PrefixDiskCatalog::kNoRecord) {
                // The index dropped the snapshot before the write returned: the record has no
                // owner in the running cache, so reclaim it now that its write is done.
                release_record(done.record);
            }
            {
                std::lock_guard lock(mutex_);
                ++stats_.failed_closures;
            }
            finish_closure(index, closure, false);
            pending_closures_.erase(found);
            continue;
        }
        {
            std::lock_guard lock(mutex_);
            stats_.disk_write_bytes += host_.image_bytes;
        }
        closure.image_landed   = true;
        closure.image_returned = true;
    }
    // A closure waits only on its queued block writes: drop the ones that returned successfully
    // in this pass so an empty list means the whole closure is durable.
    for (PendingClosure& closure : pending_closures_) {
        std::erase_if(closure.records, [&](std::uint32_t record) {
            return std::any_of(blocks.begin(), blocks.end(), [&](const CompletedBlock& done) {
                return done.ok && done.record == record;
            });
        });
    }
    // Commit every closure whose wait list emptied and whose image landed. The records vector
    // held the awaited block writes; committed ancestors were never queued, so an empty list
    // with the image landed means the whole closure is durable. A record the index released in
    // the meantime (eviction) leaves the closure uncommittable instead of publishing a dangling
    // anchor.
    for (std::size_t i = 0; i < pending_closures_.size();) {
        PendingClosure& closure = pending_closures_[i];
        if (closure.failed) {
            // The image write may still own its record; wait for its completion, which fails the
            // closure through the branch above.
            if (!closure.image_returned) {
                ++i;
                continue;
            }
            // A block failure killed this closure, so it can never commit: its snapshot's disk
            // fill is still open and its image record still reserved. Return both now instead of
            // leaving the snapshot Filling with a record nothing will ever own.
            abort_open_snapshot_fill(index, closure);
            {
                std::lock_guard lock(mutex_);
                ++stats_.failed_closures;
            }
            finish_closure(index, closure, false);
            pending_closures_.erase(pending_closures_.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }
        const bool record_lost =
            std::any_of(closure.chain_records.begin(), closure.chain_records.end(),
                        [&](std::uint32_t record) {
                            return std::find(removed_records_.begin(), removed_records_.end(),
                                             record) != removed_records_.end();
                        });
        if (!closure.records.empty() || !closure.image_landed) {
            ++i;
            continue;
        }
        const SnapshotRef snapshot = closure.snapshot;
        bool committed             = false;
        if (!record_lost && index.valid(snapshot)) {
            try {
                catalog_->commit_snapshot(closure.image_record, closure.identity);
                committed = true;
            } catch (...) { committed = false; }
            if (committed) {
                try {
                    index.mark_snapshot_disk_backed(snapshot, closure.image_record);
                } catch (...) {
                    // A committed closure the index cannot publish: drop the record again and
                    // leave the two sides agreeing, instead of a closure only one of them knows.
                    committed = false;
                }
            }
        }
        if (!committed) {
            // No commit: the snapshot's disk fill — and the image record it reserved — must not
            // outlive the closure. A failed commit, a lost chain record and an unreachable
            // snapshot all land here, and the closure ends with its Host copy intact.
            abort_open_snapshot_fill(index, closure);
        }
        finish_closure(index, closure, committed);
        pending_closures_.erase(pending_closures_.begin() + static_cast<std::ptrdiff_t>(i));
    }
    // Prune the removal log to the records the remaining closures still name.
    std::erase_if(removed_records_, [&](std::uint32_t record) {
        return std::none_of(pending_closures_.begin(), pending_closures_.end(),
                            [&](const PendingClosure& closure) {
                                return std::find(closure.chain_records.begin(),
                                                 closure.chain_records.end(),
                                                 record) != closure.chain_records.end();
                            });
    });
}

bool HybridDiskTier::reclaim_records(PrefixCacheIndex& index) {
    // The index removes whole committed closures (their image, tail and exclusive anchor blocks)
    // in GDSF order, then disk-backed blocks no committed closure needs. Each removal comes back
    // through release_disk_record, so the catalog journals it and a restart cannot resurrect it.
    const std::uint32_t evicted = index.evict_disk_entries(kReclaimRecords);
    std::uint32_t collected     = 0;
    try {
        collected = catalog_->reclaim_abandoned();
    } catch (...) { collected = 0; }
    if (collected != 0) {
        std::lock_guard lock(mutex_);
        stats_.evicted_records += collected;
    }
    if (evicted != 0) {
        std::lock_guard lock(mutex_);
        stats_.evicted_records += evicted;
    }
    return evicted != 0 || collected != 0;
}

void HybridDiskTier::finish_closure(PrefixCacheIndex& index, PendingClosure& closure,
                                    bool committed) {
    if (committed) {
        // The committed closure duplicates the Host copy; releasing it is the tier's whole point,
        // and it happens now rather than under the next eviction pressure. The anchor chain's Host
        // copies follow when no snapshot still needs them (its own closure is committed now).
        index.release_snapshot_host_copy(closure.snapshot);
    }
    if (index.valid(closure.snapshot)) { index.unpin_snapshot(closure.snapshot); }
    if (committed) {
        index.release_redundant_host_copies(closure.chain_nodes);
        std::lock_guard lock(mutex_);
        ++stats_.demoted_images;
        ++stats_.committed_closures;
    }
}

void HybridDiskTier::abort_open_snapshot_fill(PrefixCacheIndex& index,
                                              const PendingClosure& closure) {
    if (index.valid(closure.snapshot)) {
        // The fill is only open while the closure is pending: a snapshot whose record the index
        // already dropped has nothing to return.
        if (index.snapshot(closure.snapshot).disk == CopyState::Filling) {
            index.abort_snapshot_disk_fill(closure.snapshot);
        }
        return;
    }
    if (closure.image_record != PrefixDiskCatalog::kNoRecord) {
        // The snapshot left the index before its write returned; the record has no owner in the
        // running cache, so drop it now rather than let it resurrect at restart.
        release_record(closure.image_record);
    }
}

// ---- cold preparation (Engine worker) --------------------------------------------------------------

std::uint64_t HybridDiskTier::staging_bytes() const noexcept {
    return static_cast<std::uint64_t>(options_.staging_blocks) * host_.block_payload_bytes +
           static_cast<std::uint64_t>(host_.image_slabs) * host_.slab_bytes +
           host_.block_payload_bytes;
}

bool HybridDiskTier::busy() const noexcept {
    std::lock_guard lock(mutex_);
    return !write_blocks_.empty() || !write_images_.empty() || !read_blocks_.empty() ||
           read_image_.has_value() || read_tail_.has_value() || !completed_blocks_.empty() ||
           !completed_images_.empty() || !pending_closures_.empty() || !staged_blocks_.empty();
}

HybridDiskTier::Stats HybridDiskTier::stats() const noexcept {
    Stats out;
    {
        std::lock_guard lock(mutex_);
        out = stats_; // the I/O worker writes counters under the same lock
    }
    const auto catalog    = catalog_->stats();
    out.catalog_blocks    = catalog.blocks;
    out.catalog_committed = catalog.committed;
    return out;
}

void HybridDiskTier::shutdown() noexcept {
    if (stopping_.exchange(true, std::memory_order_acq_rel)) { return; }
    {
        std::lock_guard lock(mutex_);
        wake_.notify_all();
    }
    if (worker_.joinable()) { worker_.join(); }
    // Every accepted write has settled. The Engine worker's cadence will not run again, so the
    // records of closures that never committed are dropped here (a reserved record must not
    // survive as a half-closure a restart would resurrect), removals still owed are retried, and
    // the journal is flushed. A failure is reported, never thrown at a destructor or at the
    // noexcept shutdown path.
    try {
        catalog_->reclaim_abandoned();
        for (const std::uint32_t record : pending_removals_) { catalog_->remove(record); }
        pending_removals_.clear();
        catalog_->checkpoint();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "hybrid disk tier: shutdown could not checkpoint the catalog: %s\n",
                     error.what());
    } catch (...) {
        std::fprintf(stderr, "hybrid disk tier: shutdown could not checkpoint the catalog\n");
    }
    std::lock_guard lock(mutex_);
    stats_.pending_removals = 0;
}

void HybridDiskTier::reset() {
    shutdown();
    catalog_->wipe();
    {
        std::lock_guard lock(mutex_);
        stats_ = Stats{};
    }
    inqueue_bytes_ = 0;
    pending_closures_.clear();
    removed_records_.clear();
    pending_removals_.clear();
    staged_blocks_.clear();
    write_blocks_.clear();
    write_images_.clear();
    read_blocks_.clear();
    read_image_.reset();
    read_tail_.reset();
    reads_running_.clear();
    image_read_running_ = false;
    tail_read_running_  = false;
    completed_blocks_.clear();
    completed_images_.clear();
    staged_image_ = SnapshotRef{};
    staged_tail_  = SnapshotRef{};
    free_staging_.clear();
    std::fill(staging_slot_free_.begin(), staging_slot_free_.end(), true);
    free_staging_.resize(options_.staging_blocks);
    for (std::uint32_t i = 0; i < options_.staging_blocks; ++i) {
        free_staging_[i] = options_.staging_blocks - 1U - i;
    }
    stopping_.store(false, std::memory_order_release);
    worker_ = std::thread([this] { worker_loop(); });
}

bool HybridDiskTier::stage_blocks(const std::vector<NodeRef>& nodes,
                                  const PrefixCacheIndex& index) {
    std::lock_guard lock(mutex_);
    std::vector<ReadJob> jobs;
    // Bounded streaming: queue exactly the slots free now; the caller's next cadence stages more.
    // An all-or-nothing request could never stage a cold chain longer than the pool, which is the
    // normal case for a long conversation.
    const std::size_t slots = free_staging_.size();
    for (const NodeRef node : nodes) {
        if (jobs.size() >= slots) { break; }
        if (!index.valid(node)) { continue; }
        const auto queued = std::find_if(read_blocks_.begin(), read_blocks_.end(),
                                         [&](const ReadJob& job) { return job.node == node; });
        if (queued != read_blocks_.end()) { continue; } // already reading
        if (std::find(reads_running_.begin(), reads_running_.end(), node) !=
            reads_running_.end()) {
            continue; // the worker popped it; landing is imminent
        }
        if (std::find_if(staged_blocks_.begin(), staged_blocks_.end(),
                         [&](const StagedBlock& staged) { return staged.node == node; }) !=
            staged_blocks_.end()) {
            continue; // already landed
        }
        const NodeView view = index.node(node);
        if (view.disk != CopyState::Resident || view.device == CopyState::Resident ||
            view.host == CopyState::Resident) {
            continue; // only disk-only blocks stage
        }
        ReadJob job;
        job.node     = node;
        job.disk_ref = view.disk_ref;
        job.slot     = free_staging_.back();
        free_staging_.pop_back();
        staging_slot_free_[job.slot] = false;
        jobs.push_back(job);
    }
    if (jobs.empty()) { return false; }
    for (const ReadJob& job : jobs) { read_blocks_.push_back(job); }
    wake_.notify_one();
    return true;
}

std::vector<HybridDiskTier::StagedBlock> HybridDiskTier::take_staged_blocks() {
    std::lock_guard lock(mutex_);
    return std::move(staged_blocks_);
}

void HybridDiskTier::release_staged(const std::vector<StagedBlock>& blocks) {
    std::lock_guard lock(mutex_);
    for (const StagedBlock& block : blocks) { free_staging_slot(block.bytes); }
}

bool HybridDiskTier::stage_image(SnapshotRef snapshot, const PrefixCacheIndex& index) {
    std::lock_guard lock(mutex_);
    if (read_image_.has_value() || image_read_running_ || staged_image_.valid()) {
        return false;
    }
    if (!index.valid(snapshot)) { return false; }
    const SnapshotView view = index.snapshot(snapshot);
    if (view.disk != CopyState::Resident) { return false; }
    read_image_ = ImageReadJob{snapshot, view.disk_ref};
    wake_.notify_one();
    return true;
}

const std::byte* HybridDiskTier::take_staged_image(SnapshotRef snapshot) {
    std::lock_guard lock(mutex_);
    return staged_image_ == snapshot ? image_staging_.value().data() : nullptr;
}

void HybridDiskTier::release_staged_image_except(SnapshotRef snapshot) {
    std::lock_guard lock(mutex_);
    if (staged_image_.valid() && !(staged_image_ == snapshot)) { staged_image_ = SnapshotRef{}; }
}

void HybridDiskTier::release_image() {
    std::lock_guard lock(mutex_);
    staged_image_ = SnapshotRef{};
}

const std::byte* HybridDiskTier::staged_tail(SnapshotRef snapshot) const noexcept {
    std::lock_guard lock(mutex_);
    return staged_tail_ == snapshot ? tail_staging_.value().data() : nullptr;
}

void HybridDiskTier::release_staged_tail_except(SnapshotRef snapshot) {
    std::lock_guard lock(mutex_);
    if (staged_tail_.valid() && !(staged_tail_ == snapshot)) { staged_tail_ = SnapshotRef{}; }
}

void HybridDiskTier::release_tail() {
    std::lock_guard lock(mutex_);
    staged_tail_ = SnapshotRef{};
}

bool HybridDiskTier::stage_tail(SnapshotRef snapshot, const PrefixCacheIndex& index) {
    std::lock_guard lock(mutex_);
    if (read_tail_.has_value() || tail_read_running_ || staged_tail_.valid()) { return false; }
    if (!index.valid(snapshot)) { return false; }
    const SnapshotView view = index.snapshot(snapshot);
    if (view.tail_len == 0 || view.disk != CopyState::Resident ||
        view.host == CopyState::Resident) {
        return false;
    }
    read_tail_ = TailReadJob{snapshot, view.disk_ref};
    wake_.notify_one();
    return true;
}

// ---- staging slots -----------------------------------------------------------------------------------

std::byte* HybridDiskTier::block_staging(std::uint32_t slot) const noexcept {
    return static_cast<std::byte*>(block_staging_->data()) +
           static_cast<std::size_t>(slot) * host_.block_payload_bytes;
}

void HybridDiskTier::free_staging_slot(const std::byte* bytes) noexcept {
    if (!block_staging_.has_value() || bytes == nullptr) { return; }
    const auto base  = reinterpret_cast<std::uintptr_t>(block_staging_->data());
    const auto value = reinterpret_cast<std::uintptr_t>(bytes);
    const std::size_t extent =
        static_cast<std::size_t>(options_.staging_blocks) * host_.block_payload_bytes;
    if (value < base || value >= base + extent) { return; } // not a block staging pointer
    const std::size_t offset = value - base;
    if (offset % host_.block_payload_bytes != 0) { return; }
    const std::uint32_t slot = static_cast<std::uint32_t>(offset / host_.block_payload_bytes);
    if (slot >= staging_slot_free_.size() || staging_slot_free_[slot]) {
        return; // already returned: a duplicate release must not hand out the slot twice
    }
    staging_slot_free_[slot] = true;
    free_staging_.push_back(slot);
}

// ---- I/O worker ----------------------------------------------------------------------------------------

void HybridDiskTier::worker_loop() {
    for (;;) {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [&] {
            return stopping_.load(std::memory_order_acquire) || !write_blocks_.empty() ||
                   !write_images_.empty() || !read_blocks_.empty() || read_image_.has_value() ||
                   read_tail_.has_value();
        });
        if (write_blocks_.empty() && write_images_.empty() && read_blocks_.empty() &&
            !read_image_.has_value() && !read_tail_.has_value()) {
            if (stopping_.load(std::memory_order_acquire)) { return; }
            continue;
        }
        if (!write_blocks_.empty()) {
            const BlockJob job = write_blocks_.front();
            write_blocks_.pop_front();
            lock.unlock();
            run_write_block(job);
            lock.lock();
        } else if (!write_images_.empty()) {
            const ImageJob job = std::move(write_images_.front());
            write_images_.pop_front();
            lock.unlock();
            run_write_image(job);
            lock.lock();
        } else if (!read_blocks_.empty()) {
            const ReadJob job = read_blocks_.front();
            read_blocks_.pop_front();
            reads_running_.push_back(job.node);
            lock.unlock();
            run_read(job);
            lock.lock();
            std::erase(reads_running_, job.node);
        } else if (read_image_.has_value()) {
            const ImageReadJob job = *read_image_;
            read_image_.reset();
            image_read_running_ = true;
            lock.unlock();
            run_image_read(job);
            lock.lock();
            image_read_running_ = false;
        } else {
            const TailReadJob job = *read_tail_;
            read_tail_.reset();
            tail_read_running_ = true;
            lock.unlock();
            run_tail_read(job);
            lock.lock();
            tail_read_running_ = false;
        }
    }
}

void HybridDiskTier::run_write_block(const BlockJob& job) {
    bool ok = false;
    try {
        ok = catalog_->write_block(
            job.record, std::span<const std::byte>(job.source, host_.block_payload_bytes));
    } catch (...) { ok = false; }
    std::lock_guard lock(mutex_);
    completed_blocks_.push_back(
        CompletedBlock{job.node, job.record, job.identity, ok});
    inqueue_bytes_ = inqueue_bytes_ >= host_.block_payload_bytes
                         ? inqueue_bytes_ - host_.block_payload_bytes
                         : 0U;
}

void HybridDiskTier::run_write_image(const ImageJob& job) {
    bool ok = true;
    try {
        if (job.segments.size() < host_.image_slabs) {
            // The snapshot's Host copy lost slabs between submission and the write (a
            // concurrent reclaim under pressure cannot happen — the pin holds — but a
            // partially-filled snapshot reaches here when its fill aborted).
            ok = false;
        }
        for (std::uint32_t segment = 0; ok && segment != host_.image_slabs; ++segment) {
            ok = catalog_->write_image_segment(
                job.record, segment,
                std::span<const std::byte>(job.segments[segment], host_.slab_bytes));
        }
        if (ok && job.tail != nullptr) {
            ok = catalog_->write_tail(
                job.record, std::span<const std::byte>(job.tail, host_.block_payload_bytes));
        }
    } catch (...) { ok = false; }
    std::lock_guard lock(mutex_);
    completed_images_.push_back(CompletedImage{job.snapshot, job.record, job.identity, ok});
    inqueue_bytes_ = inqueue_bytes_ >= host_.image_bytes ? inqueue_bytes_ - host_.image_bytes
                                                         : 0U;
}

void HybridDiskTier::run_read(const ReadJob& job) {
    std::byte* destination = block_staging(job.slot);
    bool ok = false;
    try {
        catalog_->read_block(job.disk_ref,
                             std::span<std::byte>(destination, host_.block_payload_bytes));
        ok = true;
    } catch (...) { ok = false; }
    std::lock_guard lock(mutex_);
    if (ok) {
        staged_blocks_.push_back(StagedBlock{job.node, destination});
        stats_.disk_read_bytes += host_.block_payload_bytes;
        ++stats_.restored_blocks;
    } else {
        if (job.slot < staging_slot_free_.size() && !staging_slot_free_[job.slot]) {
            staging_slot_free_[job.slot] = true;
            free_staging_.push_back(job.slot);
        }
    }
}

void HybridDiskTier::run_image_read(const ImageReadJob& job) {
    bool ok = true;
    std::byte* staging = image_staging_.value().data();
    for (std::uint32_t segment = 0; segment != host_.image_slabs && ok; ++segment) {
        try {
            catalog_->read_image_segment(
                job.disk_ref, segment,
                std::span<std::byte>(staging + static_cast<std::size_t>(segment) * host_.slab_bytes,
                                    host_.slab_bytes));
        } catch (...) { ok = false; }
    }
    std::lock_guard lock(mutex_);
    if (ok) {
        staged_image_ = job.snapshot;
        stats_.disk_read_bytes += host_.image_bytes;
    }
}

void HybridDiskTier::run_tail_read(const TailReadJob& job) {
    bool ok = false;
    try {
        catalog_->read_tail(job.disk_ref,
                            std::span<std::byte>(tail_staging_.value().data(),
                                                 host_.block_payload_bytes));
        ok = true;
    } catch (...) { ok = false; }
    std::lock_guard lock(mutex_);
    if (ok) {
        staged_tail_ = job.snapshot;
        stats_.disk_read_bytes += host_.block_payload_bytes;
    }
}

} // namespace ninfer::models::qwen3_5::detail
