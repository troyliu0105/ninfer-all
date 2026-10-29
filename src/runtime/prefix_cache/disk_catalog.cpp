// Persistent metadata of the hybrid prefix cache's disk tier — see disk_catalog.h and
// docs/maintainer/hybrid-prefix-cache-disk-tier.md §6.
//
// File formats. The journal and the checkpoint share one record encoding:
//
//   header: magic "NHPCD1\0\0", u32 version, u32 fingerprint length, fingerprint bytes, u32
//           next_id, u64 covered journal sequence  (the last two are checkpoint only; the journal
//           inherits both from the checkpoint)
//   record: u8 kind (RecordKind), u32 ref, u32 body length, u64 journal sequence, body bytes, u32
//           CRC of the body
//
// The journal sequence numbers every journal record this directory has ever written. A checkpoint
// stores the highest sequence its state reflects, so a journal record at or below it is already
// covered: the window between replacing the checkpoint and removing the journal it covers leaves a
// stale journal behind, and replaying it as covered is what keeps a record the checkpoint dropped
// (a removal, whose payload is gone) from coming back.
//
// The journal is append-only; each checkpoint rewrites the whole state and truncates the journal.
// A torn journal tail (a short body or a CRC mismatch) ends the replay and is cut off during
// open(), so the next append continues at the last complete record. The checkpoint's records never
// count as journal records, so a journal written after a restart still starts with its own header;
// a journal whose header never landed is discarded when a checkpoint already covers it.

#include "runtime/prefix_cache/disk_catalog.h"

#include "core/disk_kv_store.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>

namespace ninfer::runtime::prefix_cache {
namespace {

constexpr std::uint32_t kFormatVersion   = 2;
constexpr std::array<char, 8> kMagic     = {'N', 'H', 'P', 'C', 'D', '1', '\0', '\0'};
// A record prefix is the kind, the ref and the body length, then the journal sequence.
constexpr std::size_t kRecordSequenceOffset = 1U + 4U + 4U;
constexpr std::size_t kRecordPrefixBytes    = kRecordSequenceOffset + 8U;

[[nodiscard]] std::filesystem::path journal_path(const std::filesystem::path& directory) {
    return directory / "catalog.journal";
}

[[nodiscard]] std::filesystem::path checkpoint_path(const std::filesystem::path& directory) {
    return directory / "catalog.ckpt";
}

[[nodiscard]] std::filesystem::path block_store_path(const std::filesystem::path& directory) {
    return directory / "blocks.bin";
}

[[nodiscard]] std::filesystem::path image_store_path(const std::filesystem::path& directory) {
    return directory / "images.bin";
}

[[nodiscard]] bool read_exact(std::FILE* file, void* out, std::size_t count) {
    return std::fread(out, 1, count, file) == count;
}

[[nodiscard]] bool write_all(std::FILE* file, const void* data, std::size_t count) {
    return std::fwrite(data, 1, count, file) == count;
}

} // namespace

// The two payload stores. The block store holds block bundles (a snapshot tail is one block
// record); the image store holds fixed image segments.
class PrefixDiskCatalog::Store {
public:
    Store(const Options& options, const Geometry& geometry) {
        DiskKVStore::Options blocks;
        blocks.path               = block_store_path(options.directory).string();
        blocks.slot_size          = geometry.block_payload_bytes;
        blocks.max_slots          = geometry.block_payload_bytes == 0
                                        ? 0U
                                        : static_cast<std::uint32_t>(options.block_capacity_bytes /
                                                                     geometry.block_payload_bytes);
        blocks.defer_index_updates = true;
        blocks_                    = std::make_unique<DiskKVStore>(std::move(blocks));
        DiskKVStore::Options images;
        images.path                = image_store_path(options.directory).string();
        images.slot_size           = geometry.image_segment_bytes;
        images.max_slots           = geometry.image_segment_bytes == 0
                                         ? 0U
                                         : static_cast<std::uint32_t>(
                                               options.image_capacity_bytes /
                                               geometry.image_segment_bytes);
        images.defer_index_updates = true;
        images_                    = std::make_unique<DiskKVStore>(std::move(images));
    }

    // Payload keys are record ids (segment 0..31 packed beside the snapshot id): exact token
    // identity lives in the catalog's in-memory tables, so the store never needs digests.
    [[nodiscard]] static DiskKVIdentity block_key(std::uint32_t ref) noexcept {
        return DiskKVIdentity{.lo = ref, .hi = ~std::uint64_t{ref}, .tag = 1, .frontier = 0};
    }

    [[nodiscard]] static DiskKVIdentity segment_key(std::uint32_t ref,
                                                    std::uint32_t segment) noexcept {
        // Eight bits of segment index: the Host layout may split one image over up to 256
        // slabs, and ids stay unique for every (record, segment) pair.
        const std::uint64_t id = (std::uint64_t{ref} << 8U) | segment;
        return DiskKVIdentity{.lo = id, .hi = ~id, .tag = 2, .frontier = 0};
    }

    [[nodiscard]] std::uint32_t block_slots_free() const { return blocks_->free_slot_count(); }
    [[nodiscard]] std::uint32_t segment_slots_free() const { return images_->free_slot_count(); }

    // One persisted payload key decoded back to the record it belongs to. Payload keys are the
    // only durable trace of a record whose metadata was lost, so open() reads them back.
    struct PayloadKey {
        enum class Kind : std::uint8_t { Block, Segment };
        Kind kind             = Kind::Block;
        std::uint32_t ref     = 0;
        std::uint32_t segment = 0;
    };

    // Keys of this format only; a slot left by another geometry (any other tag, or an identity
    // that is not a complemented id) is unreachable by definition. A block key and a snapshot
    // tail key share one encoding: the tail is the snapshot's block-sized payload.
    [[nodiscard]] static std::optional<PayloadKey> decode_key(const DiskKVIdentity& key) noexcept {
        if (key.hi != ~key.lo || key.frontier != 0) { return std::nullopt; }
        const bool block = key.tag == 1;
        if (!block && key.tag != 2) { return std::nullopt; }
        const std::uint64_t ref64 = block ? key.lo : key.lo >> 8U;
        if (ref64 == 0 || ref64 >= PrefixDiskCatalog::kNoRecord) { return std::nullopt; }
        if (block) {
            return PayloadKey{.kind = PayloadKey::Kind::Block,
                              .ref  = static_cast<std::uint32_t>(ref64)};
        }
        return PayloadKey{.kind    = PayloadKey::Kind::Segment,
                          .ref     = static_cast<std::uint32_t>(ref64),
                          .segment = static_cast<std::uint32_t>(key.lo & 0xFFU)};
    }

    // Makes both indexes cover the slots their data files hold: a payload written under deferred
    // publication survives a crash in the file but not in the index.
    void adopt_unlisted_pages() {
        (void)blocks_->adopt_unlisted_pages();
        (void)images_->adopt_unlisted_pages();
    }
    // Every live payload key, in each store. Open reads them to recover the allocator and to tell
    // a reachable payload from an orphan a crash left behind.
    [[nodiscard]] std::vector<DiskKVIdentity> block_keys() const {
        return blocks_->live_identities();
    }
    [[nodiscard]] std::vector<DiskKVIdentity> image_keys() const {
        return images_->live_identities();
    }
    // Looks up and drops a payload by the key it was stored under, leaving the index publication
    // to flush(): recovery reclaims a whole batch with one index rewrite.
    void drop_block_key(const DiskKVIdentity& key) { (void)blocks_->evict_deferred(key); }
    void drop_image_key(const DiskKVIdentity& key) { (void)images_->evict_deferred(key); }

    bool write_block(std::uint32_t ref, std::span<const std::byte> payload) {
        return blocks_->upsert_page_reserved(block_key(ref), payload);
    }

    bool write_segment(std::uint32_t ref, std::uint32_t segment,
                       std::span<const std::byte> payload) {
        return images_->upsert_page_reserved(segment_key(ref, segment), payload);
    }

    void read_block(std::uint32_t ref, std::span<std::byte> out) const {
        if (!blocks_->read_page(block_key(ref), out)) {
            throw std::runtime_error("disk catalog block record " + std::to_string(ref) +
                                     " is missing or damaged");
        }
    }

    void read_segment(std::uint32_t ref, std::uint32_t segment, std::span<std::byte> out) const {
        if (!images_->read_page(segment_key(ref, segment), out)) {
            throw std::runtime_error("disk catalog image record " + std::to_string(ref) +
                                     " segment " + std::to_string(segment) +
                                     " is missing or damaged");
        }
    }

    void evict_block(std::uint32_t ref) { (void)blocks_->evict(block_key(ref)); }

    void evict_snapshot(std::uint32_t ref, std::uint32_t segments, bool has_tail) {
        for (std::uint32_t segment = 0; segment < segments; ++segment) {
            (void)images_->evict(segment_key(ref, segment));
        }
        if (has_tail) { (void)blocks_->evict(block_key(ref)); }
    }

    void protect_all(const std::vector<std::uint32_t>& blocks,
                     const std::vector<std::pair<std::uint32_t, bool>>& snapshots,
                     std::uint32_t segments) {
        for (const std::uint32_t ref : blocks) { (void)blocks_->protect(block_key(ref)); }
        for (const auto& [ref, has_tail] : snapshots) {
            for (std::uint32_t segment = 0; segment < segments; ++segment) {
                (void)images_->protect(segment_key(ref, segment));
            }
            if (has_tail) { (void)blocks_->protect(block_key(ref)); }
        }
    }

    void flush() {
        blocks_->flush_index();
        images_->flush_index();
    }

private:
    std::unique_ptr<DiskKVStore> blocks_;
    std::unique_ptr<DiskKVStore> images_;
};

PrefixDiskCatalog::PrefixDiskCatalog(Options options, Geometry geometry,
                                     const std::string& fingerprint)
    : options_(std::move(options)), geometry_(geometry), fingerprint_(fingerprint) {
    if (options_.directory.empty()) {
        throw std::invalid_argument("disk catalog directory is empty");
    }
    // The segment count is the Host layout's image_slabs (an image split over equal slabs);
    // a large state image over a small slab lands in the dozens.
    if (geometry_.block_payload_bytes == 0 || geometry_.image_segment_bytes == 0 ||
        geometry_.image_segments == 0 || geometry_.image_segments > 256U) {
        throw std::invalid_argument("disk catalog geometry is incomplete");
    }
    std::error_code error;
    std::filesystem::create_directories(options_.directory, error);
    if (error) {
        throw std::runtime_error("cannot create " + options_.directory.string() + ": " +
                                 error.message());
    }
    store_ = std::make_unique<Store>(options_, geometry_);
}

PrefixDiskCatalog::~PrefixDiskCatalog() {
    std::lock_guard io_lock(io_mutex_);
    if (opened_ && !foreign_) {
        try {
            const CheckpointCapture capture = capture_checkpoint_locked();
            write_checkpoint_file(capture);
            publish_checkpoint_locked(capture.covered_sequence);
            store_->flush();
        } catch (...) { // a destructor never throws
        }
    }
}

void PrefixDiskCatalog::reset_state() {
    blocks_.clear();
    snapshots_.clear();
    children_.clear();
    next_id_             = 1;
    journal_records_     = 0;
    journal_since_ckpt_  = 0;
    journal_valid_bytes_ = 0;
    journal_sequence_    = 0;
    block_budget_        = 0;
    segment_budget_      = 0;
    replayed_blocks_.clear();
    replayed_snapshots_.clear();
}

bool PrefixDiskCatalog::open() {
    std::lock_guard io_lock(io_mutex_);
    std::lock_guard lock(mutex_);
    reset_state();

    // A foreign or newer-format file is reported, not interpreted: the caller wipes.
    foreign_               = false;
    bool checkpoint_loaded = false;
    if (std::FILE* check = std::fopen(checkpoint_path(options_.directory).string().c_str(), "rb");
        check != nullptr) {
        const ReplayOutcome outcome = replay_stream(check, true, nullptr);
        checkpoint_loaded           = outcome == ReplayOutcome::kOk;
        foreign_                    = outcome == ReplayOutcome::kForeign;
        std::fclose(check);
    }
    if (!foreign_) {
        const std::filesystem::path path = journal_path(options_.directory);
        if (std::FILE* journal = std::fopen(path.string().c_str(), "rb"); journal != nullptr) {
            std::uint64_t valid_bytes = 0;
            const ReplayOutcome outcome = replay_stream(journal, false, &valid_bytes);
            std::fclose(journal);
            if (outcome == ReplayOutcome::kForeign) {
                foreign_ = true;
            } else {
                if (outcome == ReplayOutcome::kUnusable) {
                    // The journal's header never landed, so nothing in it can be interpreted. A
                    // loaded checkpoint already covers the journal's records; without one, the
                    // directory carries state this format cannot read and is reported foreign.
                    if (!checkpoint_loaded) { foreign_ = true; }
                } else {
                    journal_valid_bytes_ = valid_bytes;
                }
                if (!foreign_) {
                    // Cut off a torn tail (or the remains of an append that failed): the next
                    // append rewrites from the last complete record instead of landing behind
                    // garbage, where no replay would ever read it.
                    std::error_code error;
                    const auto size = std::filesystem::file_size(path, error);
                    if (!error && size > journal_valid_bytes_) {
                        std::filesystem::resize_file(path, journal_valid_bytes_, error);
                    }
                }
            }
        }
        journal_since_ckpt_ = journal_records_;
    }
    if (foreign_) { return false; }

    // Written-but-unpublished payload slots (a crash between the payload write and the deferred
    // index publication) become live keys first: recovery decides on the stores' real contents.
    store_->adopt_unlisted_pages();

    // Recover what the payload stores still hold before publishing this session's state: ids
    // above every live key, no unreachable payload, and each closure's tail presence.
    recover_payloads();

    // Re-own every replayed payload (a store reopen loses writer ownership) and surface the
    // replay tables: blocks parents-first by construction of the ids, snapshots after commit.
    std::vector<std::uint32_t> block_ids;
    block_ids.reserve(blocks_.size());
    std::vector<std::pair<std::uint32_t, bool>> snapshot_ids;
    snapshot_ids.reserve(snapshots_.size());
    for (const auto& [ref, record] : blocks_) {
        if (record.finished) { block_ids.push_back(ref); }
    }
    for (const auto& [ref, record] : snapshots_) {
        if (record.committed) { snapshot_ids.emplace_back(ref, record.has_tail); }
    }
    store_->protect_all(block_ids, snapshot_ids, geometry_.image_segments);
    for (const std::uint32_t ref : block_ids) {
        replayed_blocks_.push_back(ReplayedBlock{ref, blocks_.at(ref).identity});
    }
    std::sort(replayed_blocks_.begin(), replayed_blocks_.end(),
              [](const ReplayedBlock& left, const ReplayedBlock& right) {
                  return left.id < right.id;
              });
    for (const auto& [ref, has_tail] : snapshot_ids) {
        (void)has_tail;
        replayed_snapshots_.push_back(ReplayedSnapshot{ref, snapshots_.at(ref).identity});
    }
    block_budget_ = store_->block_slots_free();
    segment_budget_ = store_->segment_slots_free();
    opened_ = true;
    return true;
}

// The payload stores outlive the metadata that named their records: a crash between a payload
// write and its journal record leaves bytes behind that no replayed record reaches. Their keys
// still carry the record id, so recovery reads every live key, moves the allocator above each one —
// a new record never takes an id the store still answers to, which is what makes its "a live
// identity keeps its bytes" upsert harmless — and reclaims the keys no record reaches. A closure's
// tail has no persisted flag: the block-sized payload under the snapshot's id is its record, so a
// surviving tail restores `has_tail` before reachability is decided, and a tail the crash lost
// stays absent (a later read misses).
void PrefixDiskCatalog::recover_payloads() {
    const auto block_keys = store_->block_keys();
    const auto image_keys = store_->image_keys();
    const auto next_above = [this](std::uint32_t ref) {
        const auto next = std::uint64_t{ref} + 1;
        if (next > next_id_) {
            next_id_ = next >= kNoRecord ? kNoRecord : static_cast<std::uint32_t>(next);
        }
    };
    for (const auto& key : block_keys) {
        const auto decoded = Store::decode_key(key);
        if (!decoded || decoded->kind != Store::PayloadKey::Kind::Block) { continue; }
        next_above(decoded->ref);
        if (const auto block = blocks_.find(decoded->ref); block != blocks_.end()) {
            block->second.payload_landed = true;
        } else if (const auto snapshot = snapshots_.find(decoded->ref);
                   snapshot != snapshots_.end() && snapshot->second.identity.tail_len != 0) {
            snapshot->second.has_tail = true;
        }
    }
    for (const auto& key : image_keys) {
        const auto decoded = Store::decode_key(key);
        if (!decoded || decoded->kind != Store::PayloadKey::Kind::Segment) { continue; }
        next_above(decoded->ref);
        const auto snapshot = snapshots_.find(decoded->ref);
        if (snapshot == snapshots_.end() || decoded->segment >= geometry_.image_segments) { continue; }
        const auto bit = std::uint64_t{1} << (decoded->segment % 64U);
        auto& record = snapshot->second;
        if ((record.segment_mask[decoded->segment / 64U] & bit) == 0) {
            record.segment_mask[decoded->segment / 64U] |= bit;
            ++record.segments_landed;
        }
    }
    // Metadata whose payload or ancestry disappeared is a miss before the tree is rebuilt.
    // Stable ids order parents before children, so one pass removes every broken descendant.
    std::vector<std::uint32_t> ids;
    ids.reserve(blocks_.size());
    for (const auto& [ref, record] : blocks_) { (void)record; ids.push_back(ref); }
    std::sort(ids.begin(), ids.end());
    for (const auto ref : ids) {
        const auto& record = blocks_.at(ref);
        const auto parent = record.identity.parent;
        if (!record.payload_landed ||
            (parent != kNoRecord && (parent >= ref || !blocks_.contains(parent)))) {
            blocks_.erase(ref);
        }
    }
    for (auto& [ref, record] : snapshots_) {
        (void)ref;
        record.tail_reserved = record.identity.tail_len != 0;
    }
    std::erase_if(snapshots_, [&](const auto& item) {
        const auto& record = item.second;
        return record.identity.tail_len >= kBlockTokens ||
               record.segments_landed != geometry_.image_segments ||
               (record.tail_reserved && !record.has_tail) ||
               (record.identity.anchor != kNoRecord && !blocks_.contains(record.identity.anchor));
    });
    std::uint32_t reclaimed = 0;
    for (const auto& key : block_keys) {
        const auto decoded = Store::decode_key(key);
        if (decoded && decoded->kind == Store::PayloadKey::Kind::Block &&
            (blocks_.contains(decoded->ref) ||
             (snapshots_.contains(decoded->ref) && snapshots_.at(decoded->ref).has_tail))) { continue; }
        store_->drop_block_key(key);
        ++reclaimed;
    }
    for (const auto& key : image_keys) {
        const auto decoded = Store::decode_key(key);
        if (decoded && decoded->kind == Store::PayloadKey::Kind::Segment &&
            snapshots_.contains(decoded->ref) && decoded->segment < geometry_.image_segments) {
            continue;
        }
        store_->drop_image_key(key);
        ++reclaimed;
    }
    children_.clear();
    for (const auto& [ref, record] : blocks_) {
        children_[std::uint64_t{record.identity.parent} << 40U ^ record.identity.lookup_hash]
            .push_back(ref);
    }
    if (reclaimed != 0) { store_->flush(); }
}

void PrefixDiskCatalog::wipe() {
    std::lock_guard io_lock(io_mutex_);
    std::lock_guard lock(mutex_);
    store_.reset();
    std::error_code error;
    for (const char* name :
         {"catalog.journal", "catalog.ckpt", "blocks.bin", "blocks.bin.idx", "images.bin",
          "images.bin.idx"}) {
        std::filesystem::remove(options_.directory / name, error);
    }
    reset_state();
    store_   = std::make_unique<Store>(options_, geometry_);
    foreign_ = false;
    block_budget_ = store_->block_slots_free();
    segment_budget_ = store_->segment_slots_free();
    opened_  = true;
}


std::optional<std::uint32_t> PrefixDiskCatalog::find_block(std::uint32_t parent,
                                                           std::uint64_t lookup_hash,
                                                           std::span<const TokenId> tokens,
                                                           std::uint64_t extra) const {
    std::lock_guard lock(mutex_);
    if (tokens.size() != kBlockTokens) { return std::nullopt; }
    const auto family = children_.find(std::uint64_t{parent} << 40U ^ lookup_hash);
    if (family == children_.end()) { return std::nullopt; }
    for (const std::uint32_t ref : family->second) {
        const auto record = blocks_.find(ref);
        if (record == blocks_.end() || !record->second.finished || record->second.removing) {
            continue;
        }
        const BlockIdentityRec& identity = record->second.identity;
        if (identity.parent != parent || identity.lookup_hash != lookup_hash ||
            identity.extra != extra ||
            !std::equal(tokens.begin(), tokens.end(), identity.tokens.begin())) {
            continue;
        }
        return ref;
    }
    return std::nullopt;
}

std::uint32_t PrefixDiskCatalog::begin_block_write() {
    std::lock_guard lock(mutex_);
    if (block_budget_ == 0 || next_id_ == kNoRecord) { return kNoRecord; }
    const std::uint32_t ref = next_id_;
    BlockRecord record;
    record.id = ref;
    record.reserved = true;
    blocks_.emplace(ref, record);
    ++next_id_;
    --block_budget_;
    return ref;
}

bool PrefixDiskCatalog::write_block(std::uint32_t ref, std::span<const std::byte> payload) {
    std::lock_guard io_lock(io_mutex_);
    {
        std::lock_guard lock(mutex_);
        const auto record = blocks_.find(ref);
        if (record == blocks_.end() || !record->second.reserved || record->second.removing) {
            return false;
        }
    }
    const bool landed = store_->write_block(ref, payload);
    if (landed) {
        std::lock_guard lock(mutex_);
        blocks_.at(ref).payload_landed = true;
    }
    return landed;
}

void PrefixDiskCatalog::finish_block(std::uint32_t ref, const BlockIdentityRec& identity) {
    std::lock_guard io_lock(io_mutex_);
    {
        std::lock_guard lock(mutex_);
        const auto record = blocks_.find(ref);
        if (record == blocks_.end() || !record->second.reserved || record->second.removing ||
            !record->second.payload_landed) {
            throw std::runtime_error("disk catalog finishes an unknown or unwritten block");
        }
        if (identity.parent != kNoRecord) {
            const auto parent = blocks_.find(identity.parent);
            if (identity.parent >= ref || parent == blocks_.end() ||
                !parent->second.finished || parent->second.removing) {
                throw std::runtime_error("disk catalog block lacks its finished parent");
            }
        }
        // Allocate the lookup entry before the journal append. Unfinished entries are never
        // findable, and retries do not accumulate duplicates.
        auto& family = children_[std::uint64_t{identity.parent} << 40U ^ identity.lookup_hash];
        if (std::find(family.begin(), family.end(), ref) == family.end()) { family.push_back(ref); }
        record->second.identity = identity;
    }
    const auto sequence = journal_bytes(RecordKind::Block, ref, &identity, sizeof(identity));
    {
        std::lock_guard lock(mutex_);
        auto& record = blocks_.at(ref);
        record.sequence = sequence;
        record.finished = true;
        record.reserved = false;
    }
    checkpoint_if_needed();
}

void PrefixDiskCatalog::erase_block_locked(std::uint32_t ref) {
    const auto record = blocks_.find(ref);
    if (record == blocks_.end()) { return; }
    const auto& identity = record->second.identity;
    const auto family = children_.find(std::uint64_t{identity.parent} << 40U ^ identity.lookup_hash);
    if (family != children_.end()) {
        std::erase(family->second, ref);
        if (family->second.empty()) { children_.erase(family); }
    }
    blocks_.erase(record);
    ++block_budget_;
}

void PrefixDiskCatalog::erase_snapshot_locked(std::uint32_t ref) {
    const auto record = snapshots_.find(ref);
    if (record == snapshots_.end()) { return; }
    if (record->second.tail_reserved) { ++block_budget_; }
    segment_budget_ += geometry_.image_segments;
    snapshots_.erase(record);
}

void PrefixDiskCatalog::abandon(std::uint32_t ref) {
    std::lock_guard io_lock(io_mutex_);
    bool block = false;
    bool tail = false;
    {
        std::lock_guard lock(mutex_);
        if (const auto record = blocks_.find(ref); record != blocks_.end()) {
            if (record->second.finished || record->second.removing) { return; }
            block = true;
        } else if (const auto record = snapshots_.find(ref); record != snapshots_.end()) {
            if (record->second.committed || record->second.removing) { return; }
            tail = record->second.tail_reserved;
        } else { return; }
    }
    if (block) { store_->evict_block(ref); }
    else { store_->evict_snapshot(ref, geometry_.image_segments, tail); }
    std::lock_guard lock(mutex_);
    if (block) { erase_block_locked(ref); }
    else { erase_snapshot_locked(ref); }
}

std::uint32_t PrefixDiskCatalog::begin_snapshot_write(bool has_tail) {
    std::lock_guard lock(mutex_);
    if (segment_budget_ < geometry_.image_segments || (has_tail && block_budget_ == 0) ||
        next_id_ == kNoRecord) { return kNoRecord; }
    const auto ref = next_id_;
    SnapshotRecord record;
    record.id = ref;
    record.reserved = true;
    record.tail_reserved = has_tail;
    snapshots_.emplace(ref, record);
    ++next_id_;
    segment_budget_ -= geometry_.image_segments;
    if (has_tail) { --block_budget_; }
    return ref;
}

bool PrefixDiskCatalog::write_image_segment(std::uint32_t ref, std::uint32_t segment,
                                            std::span<const std::byte> payload) {
    if (segment >= geometry_.image_segments) {
        throw std::out_of_range("disk catalog image segment index");
    }
    std::lock_guard io_lock(io_mutex_);
    {
        std::lock_guard lock(mutex_);
        const auto record = snapshots_.find(ref);
        if (record == snapshots_.end() || !record->second.reserved || record->second.removing) {
            return false;
        }
    }
    const bool landed = store_->write_segment(ref, segment, payload);
    if (landed) {
        std::lock_guard lock(mutex_);
        auto& record = snapshots_.at(ref);
        const auto bit = std::uint64_t{1} << (segment % 64U);
        if ((record.segment_mask[segment / 64U] & bit) == 0) {
            record.segment_mask[segment / 64U] |= bit;
            ++record.segments_landed;
        }
    }
    return landed;
}

bool PrefixDiskCatalog::write_tail(std::uint32_t ref, std::span<const std::byte> payload) {
    std::lock_guard io_lock(io_mutex_);
    {
        std::lock_guard lock(mutex_);
        const auto record = snapshots_.find(ref);
        if (record == snapshots_.end() || !record->second.reserved ||
            !record->second.tail_reserved || record->second.removing) { return false; }
    }
    const bool landed = store_->write_block(ref, payload);
    if (landed) {
        std::lock_guard lock(mutex_);
        snapshots_.at(ref).has_tail = true;
    }
    return landed;
}

void PrefixDiskCatalog::commit_snapshot(std::uint32_t ref, const SnapIdentityRec& identity) {
    std::lock_guard io_lock(io_mutex_);
    {
        std::lock_guard lock(mutex_);
        const auto record = snapshots_.find(ref);
        if (record == snapshots_.end() || !record->second.reserved || record->second.removing ||
            record->second.segments_landed != geometry_.image_segments ||
            record->second.tail_reserved != (identity.tail_len != 0) ||
            (record->second.tail_reserved && !record->second.has_tail)) {
            throw std::runtime_error("disk catalog commits an incomplete snapshot");
        }
        for (auto ancestor = identity.anchor; ancestor != kNoRecord;) {
            const auto node = blocks_.find(ancestor);
            if (node == blocks_.end() || !node->second.finished || node->second.removing) {
                throw std::runtime_error("disk catalog snapshot lacks its finished anchor chain");
            }
            ancestor = node->second.identity.parent;
        }
        record->second.identity = identity;
    }
    const auto sequence = journal_bytes(RecordKind::SnapshotCommit, ref, &identity, sizeof(identity));
    {
        std::lock_guard lock(mutex_);
        auto& record = snapshots_.at(ref);
        record.sequence = sequence;
        record.committed = true;
        record.reserved = false;
    }
    checkpoint_if_needed();
}

void PrefixDiskCatalog::remove(std::uint32_t ref) {
    std::lock_guard io_lock(io_mutex_);
    bool block = false;
    bool tail = false;
    bool journaled = false;
    {
        std::lock_guard lock(mutex_);
        if (const auto record = blocks_.find(ref); record != blocks_.end()) {
            record->second.removing = true;
            journaled = record->second.removal_journaled;
            block = true;
        } else if (const auto record = snapshots_.find(ref); record != snapshots_.end()) {
            record->second.removing = true;
            journaled = record->second.removal_journaled;
            tail = record->second.tail_reserved;
        } else { return; }
    }
    if (!journaled) {
        (void)journal_bytes(RecordKind::Remove, ref, nullptr, 0);
        std::lock_guard lock(mutex_);
        if (block) { blocks_.at(ref).removal_journaled = true; }
        else { snapshots_.at(ref).removal_journaled = true; }
    }
    if (block) { store_->evict_block(ref); }
    else { store_->evict_snapshot(ref, geometry_.image_segments, tail); }
    {
        std::lock_guard lock(mutex_);
        if (block) { erase_block_locked(ref); }
        else { erase_snapshot_locked(ref); }
    }
    checkpoint_if_needed();
}

std::uint32_t PrefixDiskCatalog::reclaim_abandoned() {
    // Only the stopped executor may call this bulk operation. Runtime abandonment names the
    // exact record explicitly, so an accepted job's reservation can never be swept away.
    std::vector<std::uint32_t> records;
    {
        std::lock_guard lock(mutex_);
        for (const auto& [ref, record] : blocks_) {
            if (record.reserved && !record.finished) { records.push_back(ref); }
        }
        for (const auto& [ref, record] : snapshots_) {
            if (record.reserved && !record.committed) { records.push_back(ref); }
        }
    }
    for (const auto ref : records) { abandon(ref); }
    return static_cast<std::uint32_t>(records.size());
}

void PrefixDiskCatalog::read_block(std::uint32_t ref, std::span<std::byte> out) const {
    std::lock_guard io_lock(io_mutex_);
    {
        std::lock_guard lock(mutex_);
        const auto record = blocks_.find(ref);
        if (record == blocks_.end() || !record->second.finished || record->second.removing) {
            throw std::runtime_error("disk catalog block is unavailable");
        }
    }
    store_->read_block(ref, out);
}

void PrefixDiskCatalog::read_image_segment(std::uint32_t ref, std::uint32_t segment,
                                           std::span<std::byte> out) const {
    std::lock_guard io_lock(io_mutex_);
    {
        std::lock_guard lock(mutex_);
        const auto record = snapshots_.find(ref);
        if (record == snapshots_.end() || !record->second.committed || record->second.removing ||
            segment >= geometry_.image_segments) {
            throw std::runtime_error("disk catalog image is unavailable");
        }
    }
    store_->read_segment(ref, segment, out);
}

void PrefixDiskCatalog::read_tail(std::uint32_t ref, std::span<std::byte> out) const {
    std::lock_guard io_lock(io_mutex_);
    {
        std::lock_guard lock(mutex_);
        const auto record = snapshots_.find(ref);
        if (record == snapshots_.end() || !record->second.committed || record->second.removing ||
            !record->second.has_tail) {
            throw std::runtime_error("disk catalog tail is unavailable");
        }
    }
    store_->read_block(ref, out);
}

bool PrefixDiskCatalog::has_tail(std::uint32_t ref) const {
    std::lock_guard lock(mutex_);
    const auto record = snapshots_.find(ref);
    return record != snapshots_.end() && !record->second.removing && record->second.has_tail;
}

const std::vector<PrefixDiskCatalog::ReplayedBlock>&
PrefixDiskCatalog::replayed_blocks() const noexcept {
    return replayed_blocks_;
}

const std::vector<PrefixDiskCatalog::ReplayedSnapshot>&
PrefixDiskCatalog::replayed_snapshots() const noexcept {
    return replayed_snapshots_;
}

PrefixDiskCatalog::DiskCatalogStats PrefixDiskCatalog::stats() const {
    std::lock_guard lock(mutex_);
    DiskCatalogStats out;
    out.blocks            = static_cast<std::uint32_t>(blocks_.size());
    out.snapshots         = static_cast<std::uint32_t>(snapshots_.size());
    out.abandoned         = 0;
    for (const auto& [ref, record] : blocks_) {
        (void)ref;
        if (record.reserved && !record.finished) { ++out.abandoned; }
    }
    for (const auto& [ref, record] : snapshots_) {
        (void)ref;
        if (record.committed) { ++out.committed; }
        if (record.reserved && !record.committed) { ++out.abandoned; }
    }
    // The budgets are the reservations' own accounting: exactly the store slots the record
    // tables hold from their reservations. Reading the store would wait behind payload I/O,
    // which metadata callers (Engine find/reserve/stats) must never do.
    out.block_slots_free   = block_budget_;
    out.segment_slots_free = segment_budget_;
    out.journal_records    = journal_records_;
    out.next_id            = next_id_;
    return out;
}

void PrefixDiskCatalog::checkpoint() {
    // I/O mutex across the whole operation; the metadata mutex only around the capture and the
    // journal republish. The state written is a consistent snapshot; records journaled after the
    // capture keep the journal alive so no sequence is dropped between checkpoint and journal.
    std::lock_guard io_lock(io_mutex_);
    const CheckpointCapture capture = capture_checkpoint_locked();
    write_checkpoint_file(capture);
    publish_checkpoint_locked(capture.covered_sequence);
    store_->flush();
}

// ---- journal and checkpoint ---------------------------------------------------------------------

std::uint64_t PrefixDiskCatalog::journal_bytes(RecordKind kind, std::uint32_t ref, const void* body,
                                               std::size_t body_bytes) {
    // Caller holds io_mutex_ (never the metadata mutex): the append is real I/O. Counter
    // updates take the metadata mutex briefly; the automatic checkpoint runs inside the same
    // I/O serialization so no second journal appender can interleave.
    if (!opened_ || foreign_) {
        throw std::runtime_error("disk catalog writes before a successful open");
    }
    const std::filesystem::path path = journal_path(options_.directory);
    // Every append continues the journal's durable prefix — its header plus the records that
    // parsed — so a fresh journal writes its header and the offset the catalog remembers always
    // starts the next record. Open() already cut off any torn tail, and a write that fails
    // halfway leaves the offset where it was, so the next record overwrites its remains instead
    // of landing behind them (where a replay would never read it).
    const bool fresh = journal_valid_bytes_ == 0;
    std::FILE* file = std::fopen(path.string().c_str(), fresh ? "wb" : "r+b");
    if (file == nullptr) {
        throw std::runtime_error("cannot open " + path.string() + " for append");
    }
    if (!fresh && std::fseek(file, static_cast<long>(journal_valid_bytes_), SEEK_SET) != 0) {
        std::fclose(file);
        throw std::runtime_error("disk catalog journal seek failed");
    }
    std::uint64_t record_offset = journal_valid_bytes_;
    if (fresh) {
        const std::string fingerprint = fingerprint_;
        std::array<char, 8 + 4U + 4U> head{};
        std::memcpy(head.data(), kMagic.data(), kMagic.size());
        std::memcpy(head.data() + 8, &kFormatVersion, 4U);
        const std::uint32_t length = static_cast<std::uint32_t>(fingerprint.size());
        std::memcpy(head.data() + 12, &length, 4U);
        if (!write_all(file, head.data(), head.size()) ||
            !write_all(file, fingerprint.data(), fingerprint.size())) {
            std::fclose(file);
            throw std::runtime_error("disk catalog journal header write failed");
        }
        record_offset = head.size() + fingerprint.size();
    }
    // The sequence only ever grows and a checkpoint stores its current value, so every record that
    // landed before that checkpoint is at or below it, and the journal it covers replays as
    // covered. A write that fails still consumes its sequence, which only widens that margin.
    std::uint64_t sequence = 0;
    {
        std::lock_guard lock(mutex_);
        sequence = ++journal_sequence_;
    }
    const std::uint32_t crc =
        body_bytes != 0 ? DiskKVStore::crc32(std::span<const std::byte>(
                              static_cast<const std::byte*>(body), body_bytes))
                        : 0U;
    std::array<unsigned char, kRecordPrefixBytes> prefix{};
    prefix[0] = static_cast<unsigned char>(kind);
    std::memcpy(prefix.data() + 1, &ref, 4U);
    const std::uint32_t length = static_cast<std::uint32_t>(body_bytes);
    std::memcpy(prefix.data() + 5, &length, 4U);
    std::memcpy(prefix.data() + kRecordSequenceOffset, &sequence, 8U);
    const bool ok = write_all(file, prefix.data(), prefix.size()) &&
                    (body_bytes == 0 || write_all(file, body, body_bytes)) &&
                    write_all(file, &crc, 4U) && std::fflush(file) == 0;
    std::fclose(file);
    if (!ok) { throw std::runtime_error("disk catalog journal write failed"); }
    {
        std::lock_guard lock(mutex_);
        journal_valid_bytes_ = record_offset + kRecordPrefixBytes + body_bytes + 4U;
        ++journal_records_;
        ++journal_since_ckpt_;
    }
    return sequence;
}

void PrefixDiskCatalog::checkpoint_if_needed() {
    {
        std::lock_guard lock(mutex_);
        if (journal_since_ckpt_ < options_.journal_flush_records) { return; }
    }
    const auto capture = capture_checkpoint_locked();
    write_checkpoint_file(capture);
    publish_checkpoint_locked(capture.covered_sequence);
    store_->flush();
}

PrefixDiskCatalog::CheckpointCapture PrefixDiskCatalog::capture_checkpoint_locked() {
    CheckpointCapture capture;
    {
        std::lock_guard lock(mutex_);
        capture.next_id = next_id_;
        capture.covered_sequence = journal_sequence_;
        capture.blocks.reserve(blocks_.size());
        for (const auto& [ref, record] : blocks_) {
            (void)ref;
            if (record.finished && !record.removing) { capture.blocks.push_back(record); }
        }
        capture.snapshots.reserve(snapshots_.size());
        for (const auto& [ref, record] : snapshots_) {
            (void)ref;
            if (record.committed && !record.removing) { capture.snapshots.push_back(record); }
        }
    }
    std::sort(capture.blocks.begin(), capture.blocks.end(),
              [](const auto& a, const auto& b) { return a.id < b.id; });
    std::sort(capture.snapshots.begin(), capture.snapshots.end(),
              [](const auto& a, const auto& b) { return a.id < b.id; });
    return capture;
}

void PrefixDiskCatalog::write_checkpoint_file(const CheckpointCapture& capture) {
    // Caller holds io_mutex_, never the metadata mutex: this is pure disk work.
    const std::filesystem::path path      = checkpoint_path(options_.directory);
    const std::filesystem::path temporary = path.string() + ".tmp";
    std::FILE* file = std::fopen(temporary.string().c_str(), "wb");
    if (file == nullptr) {
        throw std::runtime_error("cannot create " + temporary.string());
    }
    const std::string fingerprint = fingerprint_;
    std::array<char, 8 + 4U + 4U> head{};
    std::memcpy(head.data(), kMagic.data(), kMagic.size());
    std::memcpy(head.data() + 8, &kFormatVersion, 4U);
    const std::uint32_t length = static_cast<std::uint32_t>(fingerprint.size());
    std::memcpy(head.data() + 12, &length, 4U);
    const std::uint32_t next_id = capture.next_id;
    // The state below reflects every journal record up to the captured covered sequence; a stale
    // journal left by an interrupted replacement replays those as covered instead of applying
    // them again.
    const std::uint64_t covered_sequence = capture.covered_sequence;
    bool ok = write_all(file, head.data(), head.size()) &&
              write_all(file, fingerprint.data(), fingerprint.size()) &&
              write_all(file, &next_id, 4U) && write_all(file, &covered_sequence, 8U);
    auto emit = [&](RecordKind kind, std::uint32_t ref, std::uint64_t sequence, const void* body,
                    std::size_t body_bytes) {
        const std::uint32_t crc =
            body_bytes != 0 ? DiskKVStore::crc32(std::span<const std::byte>(
                                  static_cast<const std::byte*>(body), body_bytes))
                            : 0U;
        std::array<unsigned char, kRecordPrefixBytes> prefix{};
        prefix[0] = static_cast<unsigned char>(kind);
        std::memcpy(prefix.data() + 1, &ref, 4U);
        const std::uint32_t size = static_cast<std::uint32_t>(body_bytes);
        std::memcpy(prefix.data() + 5, &size, 4U);
        std::memcpy(prefix.data() + kRecordSequenceOffset, &sequence, 8U);
        ok = ok && write_all(file, prefix.data(), prefix.size()) &&
             (body_bytes == 0 || write_all(file, body, body_bytes)) && write_all(file, &crc, 4U);
    };
    for (const auto& record : capture.blocks) {
        emit(RecordKind::Block, record.id, record.sequence, &record.identity,
             sizeof(record.identity));
    }
    for (const auto& record : capture.snapshots) {
        emit(RecordKind::SnapshotCommit, record.id, record.sequence, &record.identity,
             sizeof(record.identity));
    }
    ok = ok && std::fflush(file) == 0;
    std::fclose(file);
    if (!ok) {
        std::error_code error;
        std::filesystem::remove(temporary, error);
        throw std::runtime_error("disk catalog checkpoint write failed");
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        throw std::runtime_error("cannot replace " + path.string() + ": " + error.message());
    }
}

void PrefixDiskCatalog::publish_checkpoint_locked(std::uint64_t covered_sequence) {
    // Caller holds io_mutex_; the metadata mutex comes up only for the counter updates. When
    // journal records landed between the capture and this republish, the new checkpoint does not
    // contain them: keeping the journal (and its later records appending behind the covered
    // prefix) is what stops the checkpoint from dropping sequences it never stored.
    // io_mutex_ excludes appenders. The unlink itself must not hold the Engine's metadata lock.
    std::error_code error;
    std::filesystem::remove(journal_path(options_.directory), error);
    std::lock_guard lock(mutex_);
    if (journal_sequence_ > covered_sequence) { return; }
    journal_records_     = 0;
    journal_since_ckpt_  = 0;
    journal_valid_bytes_ = 0;
}

PrefixDiskCatalog::ReplayOutcome PrefixDiskCatalog::replay_stream(std::FILE* file, bool checkpoint,
                                                                  std::uint64_t* valid_bytes) {
    // A header that never landed (or belongs to another build) decides the whole directory: a
    // checkpoint reports it foreign, while a journal reports itself unusable so that a checkpoint
    // which already covers its records still stands.
    const ReplayOutcome header_failure = checkpoint ? ReplayOutcome::kForeign
                                                    : ReplayOutcome::kUnusable;
    std::array<char, 8> magic{};
    if (!read_exact(file, magic.data(), magic.size()) || magic != kMagic) {
        return header_failure;
    }
    std::uint32_t version = 0;
    std::uint32_t length  = 0;
    if (!read_exact(file, &version, 4U) || !read_exact(file, &length, 4U) ||
        version != kFormatVersion || length > 4096U) {
        return header_failure;
    }
    std::string stored(length, '\0');
    if (length != 0 && !read_exact(file, stored.data(), length)) { return header_failure; }
    if (stored != fingerprint_) { return header_failure; }
    if (checkpoint) {
        std::uint32_t next_id = 0;
        std::uint64_t covered = 0;
        if (!read_exact(file, &next_id, 4U) || !read_exact(file, &covered, 8U)) {
            return ReplayOutcome::kForeign;
        }
        next_id_          = next_id;
        journal_sequence_ = covered;
    }
    // Records after this point are the journal's: a checkpoint's own records only restore state,
    // they never count as (or take the place of) journal records — and they always apply, since
    // the checkpoint is the state its covered sequence describes. A journal record at or below
    // `journal_sequence_` is already covered by the checkpoint and is replayed as covered: the
    // crash window between replacing the checkpoint and removing the journal leaves exactly those
    // records behind, and applying them again would undo a removal the checkpoint holds.
    std::uint64_t valid = 0;
    std::uint64_t previous_sequence = 0;
    for (;;) {
        std::array<unsigned char, kRecordPrefixBytes> prefix{};
        if (!read_exact(file, prefix.data(), prefix.size())) { break; } // clean EOF
        std::uint32_t ref = 0;
        std::uint32_t body_bytes = 0;
        std::uint64_t sequence = 0;
        std::memcpy(&ref, prefix.data() + 1, 4U);
        std::memcpy(&body_bytes, prefix.data() + 5, 4U);
        std::memcpy(&sequence, prefix.data() + kRecordSequenceOffset, 8U);
        if (body_bytes > sizeof(BlockIdentityRec) + 64U) { break; } // torn or corrupt tail
        if (!checkpoint) {
            // A journal's sequences only ever grow: a prefix that disagrees is damaged, and
            // stopping is safer than reading it as covered (its record would be dropped without a
            // word).
            if (sequence <= previous_sequence) { break; }
            previous_sequence = sequence;
        }
        std::vector<std::byte> body(body_bytes);
        if (body_bytes != 0 && !read_exact(file, body.data(), body_bytes)) { break; }
        std::uint32_t crc = 0;
        if (!read_exact(file, &crc, 4U)) { break; }
        if (crc != 0U &&
            crc != DiskKVStore::crc32(std::span<const std::byte>(body.data(), body_bytes))) {
            break; // torn tail: everything before it stands
        }
        if (!checkpoint && sequence <= journal_sequence_) {
            valid = static_cast<std::uint64_t>(std::ftell(file));
            ++journal_records_;
            continue; // already part of the checkpoint's state
        }
        const auto kind = static_cast<RecordKind>(prefix[0]);
        if (kind == RecordKind::Block && body_bytes == sizeof(BlockIdentityRec)) {
            BlockRecord record;
            std::memcpy(&record.identity, body.data(), body_bytes);
            record.id       = ref;
            record.sequence = sequence;
            record.finished = true;
            if (ref >= next_id_) { next_id_ = ref + 1U; }
            blocks_[ref]    = record;
            children_[std::uint64_t{record.identity.parent} << 40U ^ record.identity.lookup_hash]
                .push_back(ref);
        } else if (kind == RecordKind::SnapshotCommit &&
                   body_bytes == sizeof(SnapIdentityRec)) {
            SnapshotRecord record;
            std::memcpy(&record.identity, body.data(), body_bytes);
            record.id        = ref;
            record.sequence  = sequence;
            record.committed = true;
            if (ref >= next_id_) { next_id_ = ref + 1U; }
            snapshots_[ref]  = record;
        } else if (kind == RecordKind::Remove) {
            blocks_.erase(ref);
            snapshots_.erase(ref);
        }
        if (!checkpoint) {
            const long position = std::ftell(file);
            if (position > 0) { valid = static_cast<std::uint64_t>(position); }
            ++journal_records_;
            journal_sequence_ = sequence;
        }
    }
    if (!checkpoint && valid_bytes != nullptr) { *valid_bytes = valid; }
    return ReplayOutcome::kOk;
}

} // namespace ninfer::runtime::prefix_cache
