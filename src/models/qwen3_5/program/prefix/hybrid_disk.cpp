#include "models/qwen3_5/program/prefix/hybrid_disk.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ninfer::models::qwen3_5::detail {
namespace {
using pc::CopyState;
using pc::NodeRef;
using pc::PrefixCacheIndex;
using pc::PrefixDiskCatalog;
using pc::PrefixDiskWorker;
using pc::SnapshotRef;
constexpr std::uint32_t kReclaimRecords = 8;
constexpr std::uint64_t kSupersededGraceTicks = 2;
constexpr double kSupersededGraceEscapeOccupancy = 0.95;
}

HybridDiskTier::HybridDiskTier(Options options, Layout layout, Staging staging,
                               const std::string& fingerprint)
    : options_(std::move(options)), host_(layout), staging_(staging) {
    if (options_.directory.empty() || options_.staging_blocks == 0 ||
        host_.block_payload_bytes == 0 || host_.image_slabs == 0 || host_.slab_bytes == 0) {
        throw std::invalid_argument("hybrid disk tier requires a directory and positive geometry");
    }
    if (!(options_.host_low_watermark >= 0.0 &&
          options_.host_low_watermark < options_.host_high_watermark &&
          options_.host_high_watermark <= 1.0)) {
        throw std::invalid_argument("hybrid disk tier Host watermarks are inconsistent");
    }
    const auto image_job_bytes = host_.image_slabs * host_.slab_bytes + host_.block_payload_bytes;
    if (options_.max_inqueue_bytes == 0) { options_.max_inqueue_bytes = 3 * image_job_bytes; }
    if (options_.max_inqueue_bytes < image_job_bytes) {
        throw std::invalid_argument("hybrid disk write bound must hold one complete image and tail");
    }
    demotion_targets_.reserve(options_.demotion_batch_records);
    demotion_candidates_.reserve(2 * options_.demotion_batch_records);
    PrefixDiskCatalog::Options catalog_options;
    catalog_options.directory = options_.directory;
    catalog_options.block_capacity_bytes = options_.block_capacity_bytes;
    catalog_options.image_capacity_bytes = options_.image_capacity_bytes;
    catalog_ = std::make_unique<PrefixDiskCatalog>(
        std::move(catalog_options),
        PrefixDiskCatalog::Geometry{host_.block_payload_bytes, host_.slab_bytes, host_.image_slabs},
        fingerprint);
    if (staging_.blocks.size() != options_.staging_blocks * host_.block_payload_bytes ||
        staging_.image.size() != host_.image_slabs * host_.slab_bytes ||
        staging_.tail.size() != host_.block_payload_bytes) {
        throw std::invalid_argument("hybrid disk staging does not match its accounted geometry");
    }
    staging_leases_.assign(options_.staging_blocks, 0);
    for (std::uint32_t i = 0; i < options_.staging_blocks; ++i) {
        free_staging_.push_back(options_.staging_blocks - 1 - i);
    }
}

HybridDiskTier::~HybridDiskTier() { shutdown(); }

HybridDiskTier::RestoreSummary HybridDiskTier::restore_into(PrefixCacheIndex& index) {
    if (worker_) { throw std::logic_error("hybrid disk tier was already opened"); }
    RestoreSummary out;
    out.ok = catalog_->open();
    if (!out.ok) { catalog_->wipe(); }
    std::unordered_map<std::uint32_t, NodeRef> restored;
    for (const auto& block : catalog_->replayed_blocks()) {
        NodeRef parent;
        if (block.identity.parent != PrefixDiskCatalog::kNoRecord) {
            const auto found = restored.find(block.identity.parent);
            if (found == restored.end()) { continue; }
            parent = found->second;
        }
        const auto node = index.restore_disk_block(parent, block.identity.lookup_hash,
                                                   block.identity.tokens, block.identity.extra,
                                                   block.id);
        if (!node) { break; }
        restored.emplace(block.id, *node);
    }
    for (const auto& snapshot : catalog_->replayed_snapshots()) {
        NodeRef anchor;
        if (snapshot.identity.anchor != PrefixDiskCatalog::kNoRecord) {
            const auto found = restored.find(snapshot.identity.anchor);
            if (found == restored.end()) { continue; }
            anchor = found->second;
        }
        if (index.restore_disk_snapshot(
                anchor, snapshot.identity.frontier,
                {snapshot.identity.tail.data(), snapshot.identity.tail_len}, snapshot.identity.kind,
                snapshot.identity.hits, snapshot.id, true)) {
            ++out.snapshots;
        }
    }
    out.blocks = restored.size();
    out.kv_bytes = out.blocks * host_.block_payload_bytes;
    worker_ = std::make_unique<PrefixDiskWorker>(
        *catalog_, PrefixDiskCatalog::Geometry{host_.block_payload_bytes, host_.slab_bytes,
                                              host_.image_slabs},
        options_.max_inqueue_bytes, options_.staging_blocks + 2);
    worker_->start();
    return out;
}

void HybridDiskTier::release_record(std::uint32_t record) noexcept {
    if (record == pc::kNoId || stopping_ || !worker_) { return; }
    try { worker_->remove(record); }
    catch (...) {
        // Only queue allocation can fail here; no filesystem operation runs on this thread.
        try { pending_removals_.push_back(record); }
        catch (...) { std::terminate(); }
    }
}
void HybridDiskTier::release_disk_record(std::uint32_t record) noexcept { release_record(record); }
void HybridDiskTier::release_snapshot_disk_records(std::uint32_t record) noexcept {
    release_record(record);
}
void HybridDiskTier::retry_removals() {
    std::vector<std::uint32_t> pending;
    pending.swap(pending_removals_);
    for (const auto record : pending) { release_record(record); }
    stats_.pending_removals = static_cast<std::uint32_t>(pending_removals_.size());
}
bool HybridDiskTier::reclaim_records(PrefixCacheIndex& index) {
    const auto loose = index.release_disk_record_range(kReclaimRecords);
    const auto closures = loose < kReclaimRecords ? index.evict_disk_entries(kReclaimRecords) : 0;
    stats_.evicted_records += loose + closures;
    // Removal is asynchronous. New reservations retry on a later cadence after the worker has
    // journaled the removals. Never reclaim reservations belonging to accepted write jobs.
    return loose + closures != 0;
}

void HybridDiskTier::maybe_demote(PrefixCacheIndex& index, const HostTransfers& transfers) {
    if (!worker_ || stopping_) { return; }
    const auto slabs = index.config().host_slabs;
    const auto occupied = [&] { return slabs - index.stats().host_free_slabs; };
    if (slabs == 0) { return; }
    if (occupied() <= slabs * options_.host_low_watermark) { demoting_ = false; }
    if (occupied() >= slabs * options_.host_high_watermark) { demoting_ = true; }
    std::erase_if(demotion_targets_, [&](SnapshotRef ref) {
        return !index.valid(ref) || index.snapshot(ref).disk_committed;
    });
    if (!demoting_ && demotion_targets_.empty()) { return; }
    auto& candidates = demotion_candidates_;
    candidates.assign(demotion_targets_.begin(), demotion_targets_.end());
    if (demoting_) {
        for (const auto ref : index.collect_demotion_candidates(options_.demotion_batch_records)) {
            if (std::find(candidates.begin(), candidates.end(), ref) == candidates.end()) {
                candidates.push_back(ref);
            }
        }
    }
    for (const auto snapshot : candidates) {
        if (occupied() <= slabs * options_.host_low_watermark) { demoting_ = false; }
        if (!index.valid(snapshot)) { continue; }
        const bool continuing = std::find(demotion_targets_.begin(), demotion_targets_.end(),
                                           snapshot) != demotion_targets_.end();
        if ((!demoting_ && !continuing) ||
            (!continuing && demotion_targets_.size() >= options_.demotion_batch_records)) {
            continue;
        }
        const auto remember = [&] {
            if (std::find(demotion_targets_.begin(), demotion_targets_.end(), snapshot) ==
                demotion_targets_.end()) { demotion_targets_.push_back(snapshot); }
        };
        const auto view = index.snapshot(snapshot);
        if (view.pins != 0 || view.disk != CopyState::Absent) { continue; }
        if (!continuing && view.superseded &&
            occupied() < slabs * kSupersededGraceEscapeOccupancy &&
            index.latest_superseded_tick() - view.superseded_tick < kSupersededGraceTicks) {
            continue;
        }
        if (view.host == CopyState::Absent &&
            transfers.start_snapshot_host_write(transfers.context, snapshot)) {
            remember();
            continue;
        }
        // A Device-only image may not yet fit on Host. Its resident ancestor blocks can still
        // stream to Disk and release enough slabs for the next image-fill attempt.
        if (view.host == CopyState::Filling ||
            (view.host == CopyState::Resident &&
             view.host_slabs.size() != host_.image_slabs + (view.tail_len != 0 ? 1U : 0U))) {
            continue;
        }
        auto& chain = demotion_path_;
        chain.clear();
        for (auto node = view.anchor; node.valid(); node = index.node(node).parent) {
            chain.push_back(node);
        }
        std::reverse(chain.begin(), chain.end());
        bool ready = true;
        std::uint32_t parent_record = pc::kNoId;
        for (const auto node : chain) {
            const auto block = index.node(node);
            if (block.disk == CopyState::Resident) {
                parent_record = block.disk_ref;
                continue;
            }
            ready = false;
            if (block.disk == CopyState::Filling) {
                parent_record = block.disk_ref;
                continue;
            }
            if (block.host != CopyState::Resident) {
                if (block.device == CopyState::Resident && block.host == CopyState::Absent) {
                    // One reserved slab funds a repair even at full ordinary Host occupancy.
                    transfers.start_block_host_write(transfers.context, node);
                    if (index.node(node).host == CopyState::Filling) { remember(); }
                }
                break; // no child record may name a missing parent
            }
            if (!worker_->can_write(host_.block_payload_bytes)) {
                ++stats_.queue_rejects;
                break;
            }
            const auto record = catalog_->begin_block_write();
            if (record == pc::kNoId) {
                (void)reclaim_records(index);
                ++stats_.queue_rejects;
                break;
            }
            const auto identity = index.block_identity(node);
            PrefixDiskWorker::BlockWrite job;
            job.node = node;
            job.record = record;
            job.identity.parent = parent_record;
            job.identity.lookup_hash = identity.lookup_hash;
            job.identity.extra = identity.extra;
            std::copy(identity.tokens.begin(), identity.tokens.end(), job.identity.tokens.begin());
            job.source = {transfers.slab_bytes_of(transfers.context, block.host_slab),
                          host_.block_payload_bytes};
            index.pin_node(node);
            index.begin_node_disk_fill(node, record);
            try {
                if (!worker_->submit(std::move(job))) {
                    index.abort_node_disk_fill(node);
                    index.unpin_node(node);
                    release_record(record);
                    ++stats_.queue_rejects;
                    break;
                }
            } catch (...) {
                index.abort_node_disk_fill(node);
                index.unpin_node(node);
                release_record(record);
                throw;
            }
            remember();
            parent_record = record;
        }
        // Partial blocks land independently and release their leased Host copies. No snapshot
        // pin or reservation spans this wait, so backpressure cannot strand a repair pin.
        if (!ready) { continue; }
        if (view.host != CopyState::Resident) {
            // A started closure must retry its image after other jobs return Host space, even
            // below the low watermark. Keeping this bounded, unpinned interest leases no bytes.
            continue;
        }
        const auto bytes = host_.image_slabs * host_.slab_bytes +
                           (view.tail_len != 0 ? host_.block_payload_bytes : 0);
        if (!worker_->can_write(bytes)) { ++stats_.queue_rejects; continue; }
        const auto record = catalog_->begin_snapshot_write(view.tail_len != 0);
        if (record == pc::kNoId) {
            (void)reclaim_records(index);
            ++stats_.queue_rejects;
            continue;
        }
        PrefixDiskWorker::ImageWrite job;
        job.snapshot = snapshot;
        job.record = record;
        job.identity.anchor = parent_record;
        job.identity.frontier = view.frontier;
        job.identity.tail_len = view.tail_len;
        job.identity.kind = view.kind;
        job.identity.hits = view.hits;
        std::copy(view.tail.begin(), view.tail.end(), job.identity.tail.begin());
        try {
            job.segments.reserve(host_.image_slabs);
            for (std::uint32_t i = 0; i < host_.image_slabs; ++i) {
                job.segments.push_back(transfers.slab_bytes_of(transfers.context, view.host_slabs[i]));
            }
        } catch (...) { release_record(record); throw; }
        if (view.tail_len != 0) {
            job.tail = {transfers.slab_bytes_of(transfers.context, view.host_slabs.back()),
                        host_.block_payload_bytes};
        }
        index.pin_snapshot(snapshot);
        index.begin_snapshot_disk_fill(snapshot, record);
        try {
            if (!worker_->submit(std::move(job))) {
                index.abort_snapshot_disk_fill(snapshot);
                index.unpin_snapshot(snapshot);
                ++stats_.queue_rejects;
            }
        } catch (...) {
            index.abort_snapshot_disk_fill(snapshot);
            index.unpin_snapshot(snapshot);
            throw;
        }
    }
}

void HybridDiskTier::publish_completions(PrefixCacheIndex& index) {
    if (!worker_) { return; }
    retry_removals();
    for (const auto& done : worker_->take_completions()) {
        using Kind = PrefixDiskWorker::Kind;
        if (done.kind == Kind::Remove) {
            if (done.ok) { ++stats_.released_records; }
            else { pending_removals_.push_back(done.record); }
        } else if (done.kind == Kind::WriteBlock) {
            if (index.valid(done.node)) {
                const auto view = index.node(done.node);
                if (view.disk == CopyState::Filling && view.disk_ref == done.record) {
                    if (done.ok && !view.disk_invalidated) {
                        index.mark_disk_backed(done.node, done.record);
                        ++stats_.demoted_blocks;
                        stats_.disk_write_bytes += done.bytes;
                    } else {
                        index.abort_node_disk_fill(done.node);
                        release_record(done.record);
                    }
                } else { release_record(done.record); }
                // A completed block is a valid mixed-tier source before its snapshot commits.
                // Releasing it now funds the next bounded Device-to-Host repair.
                index.release_redundant_host_copies({&done.node, 1});
                index.unpin_node(done.node);
            } else { release_record(done.record); }
        } else if (done.kind == Kind::WriteImage) {
            bool published = false;
            if (index.valid(done.snapshot)) {
                const auto view = index.snapshot(done.snapshot);
                if (view.disk == CopyState::Filling && view.disk_ref == done.record) {
                    bool path_valid = true;
                    for (auto node = view.anchor; node.valid(); node = index.node(node).parent) {
                        if (index.node(node).disk != CopyState::Resident) { path_valid = false; break; }
                    }
                    if (done.ok && path_valid && !view.disk_invalidated) {
                        index.mark_snapshot_disk_backed(done.snapshot, done.record);
                        index.release_snapshot_host_copy(done.snapshot);
                        ++stats_.demoted_images;
                        ++stats_.committed_closures;
                        stats_.disk_write_bytes += done.bytes;
                        published = true;
                    } else { index.abort_snapshot_disk_fill(done.snapshot); }
                }
                index.unpin_snapshot(done.snapshot);
            }
            if (!published) { release_record(done.record); ++stats_.failed_closures; }
            std::erase(demotion_targets_, done.snapshot);
        } else if (done.kind == Kind::ReadBlock) {
            std::erase(reading_blocks_, done.node);
            const bool valid = index.valid(done.node) &&
                               index.node(done.node).disk == CopyState::Resident &&
                               index.node(done.node).disk_ref == done.record;
            if (done.ok && valid) {
                staged_blocks_.push_back({done.node, block_staging(done.slot), 0,
                                          staging_leases_[done.slot]});
                stats_.disk_read_bytes += done.bytes;
                ++stats_.restored_blocks;
            } else {
                free_staging_slot(block_staging(done.slot), staging_leases_[done.slot]);
                if (!done.ok) {
                    index.invalidate_disk_block(done.node, done.record);
                    release_record(done.record);
                }
            }
        } else {
            const bool image = done.kind == Kind::ReadImage;
            if (image) { reading_image_.reset(); }
            else { reading_tail_.reset(); }
            const bool valid = index.valid(done.snapshot) &&
                               index.snapshot(done.snapshot).disk == CopyState::Resident &&
                               index.snapshot(done.snapshot).disk_ref == done.record;
            if (done.ok && valid) {
                if (image) { staged_image_ = done.snapshot; }
                else { staged_tail_ = done.snapshot; }
                stats_.disk_read_bytes += done.bytes;
            } else if (!done.ok) {
                index.invalidate_disk_snapshot(done.snapshot, done.record);
                release_record(done.record);
            }
        }
    }
    stats_.pending_removals = static_cast<std::uint32_t>(pending_removals_.size());
}

bool HybridDiskTier::stage_blocks(const std::vector<NodeRef>& nodes, const PrefixCacheIndex& index) {
    bool started = false;
    for (const auto node : nodes) {
        if (free_staging_.empty()) { break; }
        if (!index.valid(node) ||
            std::find(reading_blocks_.begin(), reading_blocks_.end(), node) != reading_blocks_.end() ||
            std::any_of(staged_blocks_.begin(), staged_blocks_.end(),
                        [&](const auto& staged) { return staged.node == node; })) { continue; }
        const auto view = index.node(node);
        if (view.disk != CopyState::Resident || view.device != CopyState::Absent ||
            view.host != CopyState::Absent) { continue; }
        const auto slot = free_staging_.back();
        reading_blocks_.push_back(node);
        try {
            if (!worker_->submit(PrefixDiskWorker::Read{
                    PrefixDiskWorker::Kind::ReadBlock, node, {}, view.disk_ref, slot,
                    {block_staging(slot), host_.block_payload_bytes}})) {
                reading_blocks_.pop_back();
                break;
            }
        } catch (...) {
            reading_blocks_.pop_back();
            throw;
        }
        free_staging_.pop_back();
        staging_leases_[slot] = next_staging_lease_++;
        started = true;
    }
    return started;
}
std::vector<HybridDiskTier::StagedBlock> HybridDiskTier::take_staged_blocks() {
    std::vector<StagedBlock> out;
    out.swap(staged_blocks_);
    return out;
}
void HybridDiskTier::release_staged(const std::vector<StagedBlock>& blocks) {
    for (const auto& block : blocks) { free_staging_slot(block.bytes, block.lease); }
}
bool HybridDiskTier::stage_image(SnapshotRef snapshot, const PrefixCacheIndex& index) {
    if (reading_image_ || staged_image_.valid() || !index.valid(snapshot)) { return false; }
    const auto view = index.snapshot(snapshot);
    if (view.disk != CopyState::Resident || !view.disk_committed) { return false; }
    if (!worker_->submit(PrefixDiskWorker::Read{
            PrefixDiskWorker::Kind::ReadImage, {}, snapshot, view.disk_ref, 0,
            staging_.image})) {
        return false;
    }
    reading_image_ = snapshot;
    return true;
}
const std::byte* HybridDiskTier::take_staged_image(SnapshotRef snapshot) {
    return staged_image_ == snapshot ? staging_.image.data() : nullptr;
}
void HybridDiskTier::release_staged_image_except(SnapshotRef snapshot) {
    if (!(staged_image_ == snapshot)) { release_image(); }
}
void HybridDiskTier::release_image() { staged_image_ = {}; }
const std::byte* HybridDiskTier::staged_tail(SnapshotRef snapshot) const noexcept {
    return staged_tail_ == snapshot ? staging_.tail.data() : nullptr;
}
void HybridDiskTier::release_staged_tail_except(SnapshotRef snapshot) {
    if (!(staged_tail_ == snapshot)) { release_tail(); }
}
void HybridDiskTier::release_tail() { staged_tail_ = {}; }
bool HybridDiskTier::stage_tail(SnapshotRef snapshot, const PrefixCacheIndex& index) {
    if (reading_tail_ || staged_tail_.valid() || !index.valid(snapshot)) { return false; }
    const auto view = index.snapshot(snapshot);
    if (view.tail_len == 0 || view.disk != CopyState::Resident || !view.disk_committed ||
        view.host == CopyState::Resident) { return false; }
    if (!worker_->submit(PrefixDiskWorker::Read{
            PrefixDiskWorker::Kind::ReadTail, {}, snapshot, view.disk_ref, 0,
            staging_.tail})) {
        return false;
    }
    reading_tail_ = snapshot;
    return true;
}
std::byte* HybridDiskTier::block_staging(std::uint32_t slot) const noexcept {
    return staging_.blocks.data() + slot * host_.block_payload_bytes;
}
void HybridDiskTier::free_staging_slot(const std::byte* bytes, std::uint64_t lease) noexcept {
    const auto base = reinterpret_cast<std::uintptr_t>(staging_.blocks.data());
    const auto value = reinterpret_cast<std::uintptr_t>(bytes);
    const auto extent = options_.staging_blocks * host_.block_payload_bytes;
    if (value < base || value >= base + extent) { return; }
    const auto offset = value - base;
    if (offset % host_.block_payload_bytes != 0) { return; }
    const auto slot = static_cast<std::uint32_t>(offset / host_.block_payload_bytes);
    if (lease == 0 || staging_leases_[slot] != lease) { return; }
    staging_leases_[slot] = 0;
    free_staging_.push_back(slot);
}
std::uint64_t HybridDiskTier::staging_bytes() const noexcept {
    return options_.staging_blocks * host_.block_payload_bytes +
           host_.image_slabs * host_.slab_bytes + host_.block_payload_bytes;
}
bool HybridDiskTier::busy() const noexcept {
    return worker_ && (worker_->busy() || !staged_blocks_.empty() || !pending_removals_.empty());
}
HybridDiskTier::Stats HybridDiskTier::stats() const noexcept {
    Stats out = stats_;
    const auto current = catalog_->stats();
    out.catalog_blocks = current.blocks;
    out.catalog_committed = current.committed;
    return out;
}
void HybridDiskTier::shutdown() noexcept {
    if (stopping_) { return; }
    stopping_ = true;
    if (!worker_) { return; }
    worker_->shutdown();
    // No accepted job can now recreate a reclaimed reservation. A normal shutdown may retain
    // completed closures even if the Engine has not adopted their completion yet.
    try {
        for (const auto& done : worker_->take_completions()) {
            if (!done.ok) { catalog_->remove(done.record); }
        }
        for (const auto record : pending_removals_) { catalog_->remove(record); }
        catalog_->reclaim_abandoned();
        catalog_->checkpoint();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "hybrid disk tier: shutdown checkpoint failed: %s\n", error.what());
    } catch (...) {
        std::fprintf(stderr, "hybrid disk tier: shutdown checkpoint failed\n");
    }
}

} // namespace ninfer::models::qwen3_5::detail
