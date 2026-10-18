// Persistent metadata of the hybrid prefix cache's disk tier — see disk_catalog.h and
// docs/maintainer/hybrid-prefix-cache-disk-tier.md §6.
//
// File formats. The journal and the checkpoint share one record encoding:
//
//   header: magic "NHPCD1\0\0", u32 version, u32 fingerprint length, fingerprint bytes, u32
//           next_id  (checkpoint only; the journal inherits the checkpoint's next_id)
//   record: u8 kind (RecordKind), u32 ref, u32 body length, body bytes, u32 CRC of the body
//
// The journal is append-only; each checkpoint rewrites the whole state and truncates the
// journal. A torn journal tail (a short body or a CRC mismatch) ends the replay.

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

constexpr std::uint32_t kFormatVersion   = 1;
constexpr std::array<char, 8> kMagic     = {'N', 'H', 'P', 'C', 'D', '1', '\0', '\0'};
constexpr std::size_t kRecordPrefixBytes = 1U + 4U + 4U;

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
    std::lock_guard lock(mutex_);
    if (opened_ && !foreign_) {
        try {
            write_checkpoint_locked();
            store_->flush();
        } catch (...) { // a destructor never throws
        }
    }
}

void PrefixDiskCatalog::reset_state() {
    blocks_.clear();
    snapshots_.clear();
    children_.clear();
    next_id_            = 1;
    journal_records_    = 0;
    journal_since_ckpt_ = 0;
    replayed_blocks_.clear();
    replayed_snapshots_.clear();
}

bool PrefixDiskCatalog::open() {
    std::lock_guard lock(mutex_);
    reset_state();

    // A foreign or newer-format file is reported, not interpreted: the caller wipes.
    foreign_ = false;
    bool loaded = false;
    if (std::FILE* check = std::fopen(checkpoint_path(options_.directory).string().c_str(), "rb");
        check != nullptr) {
        replay_stream(check, true);
        loaded = !foreign_;
        std::fclose(check);
    }
    if (loaded || std::filesystem::exists(journal_path(options_.directory))) {
        if (std::FILE* journal =
                std::fopen(journal_path(options_.directory).string().c_str(), "rb");
            journal != nullptr) {
            if (!foreign_) { replay_stream(journal, false); }
            std::fclose(journal);
        }
    }
    if (foreign_) { return false; }

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
    opened_ = true;
    return true;
}

void PrefixDiskCatalog::wipe() {
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
    opened_  = true;
}

std::uint32_t PrefixDiskCatalog::allocate_id() {
    if (next_id_ == kNoRecord) { throw std::runtime_error("disk catalog id space exhausted"); }
    return next_id_++;
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
        if (record == blocks_.end() || !record->second.finished) { continue; }
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
    std::uint32_t outstanding = 0;
    for (const auto& [ref, record] : blocks_) {
        (void)ref;
        if (record.reserved && !record.finished) { ++outstanding; }
    }
    if (outstanding >= store_->block_slots_free()) { return kNoRecord; }
    const std::uint32_t ref = allocate_id();
    blocks_[ref]            = BlockRecord{.id = ref, .reserved = true};
    return ref;
}

bool PrefixDiskCatalog::write_block(std::uint32_t ref, std::span<const std::byte> payload) {
    std::lock_guard lock(mutex_);
    return store_->write_block(ref, payload);
}

void PrefixDiskCatalog::finish_block(std::uint32_t ref, const BlockIdentityRec& identity) {
    std::lock_guard lock(mutex_);
    const auto record = blocks_.find(ref);
    if (record == blocks_.end() || !record->second.reserved) {
        throw std::runtime_error("disk catalog finishes an unknown block record");
    }
    record->second.identity = identity;
    record->second.finished = true;
    record->second.reserved = false;
    children_[std::uint64_t{identity.parent} << 40U ^ identity.lookup_hash].push_back(ref);
    journal_bytes(RecordKind::Block, ref, &record->second.identity,
                  sizeof(record->second.identity));
}

void PrefixDiskCatalog::abandon(std::uint32_t ref) {
    std::lock_guard lock(mutex_);
    if (const auto record = blocks_.find(ref);
        record != blocks_.end() && !record->second.finished) {
        store_->evict_block(ref);
        blocks_.erase(record);
        return;
    }
    if (const auto snapshot = snapshots_.find(ref);
        snapshot != snapshots_.end() && !snapshot->second.committed) {
        store_->evict_snapshot(ref, geometry_.image_segments, snapshot->second.has_tail);
        snapshots_.erase(snapshot);
    }
}

std::uint32_t PrefixDiskCatalog::begin_snapshot_write() {
    std::lock_guard lock(mutex_);
    std::uint32_t outstanding = 0;
    for (const auto& [ref, record] : snapshots_) {
        (void)ref;
        if (record.reserved && !record.committed) { ++outstanding; }
    }
    if (store_->segment_slots_free() <
        (outstanding + 1U) * geometry_.image_segments) {
        return kNoRecord;
    }
    const std::uint32_t ref = allocate_id();
    snapshots_[ref]         = SnapshotRecord{.id = ref, .reserved = true};
    return ref;
}

bool PrefixDiskCatalog::write_image_segment(std::uint32_t ref, std::uint32_t segment,
                                            std::span<const std::byte> payload) {
    std::lock_guard lock(mutex_);
    if (segment >= geometry_.image_segments) {
        throw std::out_of_range("disk catalog image segment index");
    }
    return store_->write_segment(ref, segment, payload);
}

bool PrefixDiskCatalog::write_tail(std::uint32_t ref, std::span<const std::byte> payload) {
    std::lock_guard lock(mutex_);
    const auto record = snapshots_.find(ref);
    if (record == snapshots_.end()) { return false; }
    if (!store_->write_block(ref, payload)) { return false; }
    record->second.has_tail = true;
    return true;
}

void PrefixDiskCatalog::commit_snapshot(std::uint32_t ref, const SnapIdentityRec& identity) {
    std::lock_guard lock(mutex_);
    const auto record = snapshots_.find(ref);
    if (record == snapshots_.end() || !record->second.reserved) {
        throw std::runtime_error("disk catalog commits an unknown snapshot record");
    }
    if (identity.anchor != kNoRecord) {
        const auto anchor = blocks_.find(identity.anchor);
        if (anchor == blocks_.end() || !anchor->second.finished) {
            throw std::runtime_error("disk catalog snapshot commit lacks its anchor record");
        }
    }
    record->second.identity = identity;
    record->second.committed = true;
    record->second.reserved  = false;
    journal_bytes(RecordKind::SnapshotCommit, ref, &record->second.identity,
                  sizeof(record->second.identity));
}

void PrefixDiskCatalog::remove(std::uint32_t ref) {
    std::lock_guard lock(mutex_);
    if (const auto record = blocks_.find(ref); record != blocks_.end()) {
        store_->evict_block(ref);
        const BlockIdentityRec& identity = record->second.identity;
        if (const auto family =
                children_.find(std::uint64_t{identity.parent} << 40U ^ identity.lookup_hash);
            family != children_.end()) {
            std::erase(family->second, ref);
        }
        blocks_.erase(record);
        journal_bytes(RecordKind::Remove, ref, nullptr, 0);
        return;
    }
    if (const auto snapshot = snapshots_.find(ref); snapshot != snapshots_.end()) {
        store_->evict_snapshot(ref, geometry_.image_segments, snapshot->second.has_tail);
        snapshots_.erase(snapshot);
        journal_bytes(RecordKind::Remove, ref, nullptr, 0);
    }
}

std::uint32_t PrefixDiskCatalog::reclaim_abandoned() {
    std::lock_guard lock(mutex_);
    std::uint32_t reclaimed = 0;
    for (auto it = blocks_.begin(); it != blocks_.end();) {
        if (it->second.reserved && !it->second.finished) {
            store_->evict_block(it->first);
            it = blocks_.erase(it);
            ++reclaimed;
            continue;
        }
        ++it;
    }
    for (auto it = snapshots_.begin(); it != snapshots_.end();) {
        if (it->second.reserved && !it->second.committed) {
            store_->evict_snapshot(it->first, geometry_.image_segments, it->second.has_tail);
            it = snapshots_.erase(it);
            ++reclaimed;
            continue;
        }
        ++it;
    }
    return reclaimed;
}

void PrefixDiskCatalog::read_block(std::uint32_t ref, std::span<std::byte> out) const {
    std::lock_guard lock(mutex_);
    store_->read_block(ref, out);
}

void PrefixDiskCatalog::read_image_segment(std::uint32_t ref, std::uint32_t segment,
                                           std::span<std::byte> out) const {
    std::lock_guard lock(mutex_);
    store_->read_segment(ref, segment, out);
}

void PrefixDiskCatalog::read_tail(std::uint32_t ref, std::span<std::byte> out) const {
    std::lock_guard lock(mutex_);
    if (!snapshots_.at(ref).has_tail) {
        throw std::runtime_error("disk catalog snapshot has no tail record");
    }
    store_->read_block(ref, out);
}

bool PrefixDiskCatalog::has_tail(std::uint32_t ref) const {
    std::lock_guard lock(mutex_);
    const auto record = snapshots_.find(ref);
    return record != snapshots_.end() && record->second.has_tail;
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
    out.block_slots_free   = store_->block_slots_free();
    out.segment_slots_free = store_->segment_slots_free();
    out.journal_records    = journal_records_;
    out.next_id            = next_id_;
    return out;
}

void PrefixDiskCatalog::checkpoint() {
    std::lock_guard lock(mutex_);
    write_checkpoint_locked();
    store_->flush();
}

// ---- journal and checkpoint ---------------------------------------------------------------------

void PrefixDiskCatalog::journal_bytes(RecordKind kind, std::uint32_t ref, const void* body,
                                      std::size_t body_bytes) {
    if (!opened_ || foreign_) {
        throw std::runtime_error("disk catalog writes before a successful open");
    }
    const std::filesystem::path path = journal_path(options_.directory);
    std::FILE* file = std::fopen(path.string().c_str(), journal_records_ == 0 ? "wb" : "ab");
    if (file == nullptr) {
        throw std::runtime_error("cannot open " + path.string() + " for append");
    }
    if (journal_records_ == 0) {
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
    bool ok = write_all(file, prefix.data(), prefix.size()) &&
              (body_bytes == 0 || write_all(file, body, body_bytes)) &&
              write_all(file, &crc, 4U);
    std::fclose(file);
    if (!ok) { throw std::runtime_error("disk catalog journal write failed"); }
    ++journal_records_;
    if (++journal_since_ckpt_ >= options_.journal_flush_records) {
        write_checkpoint_locked();
        store_->flush();
    }
}

void PrefixDiskCatalog::write_checkpoint_locked() {
    const std::filesystem::path path     = checkpoint_path(options_.directory);
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
    const std::uint32_t next_id = next_id_;
    bool ok = write_all(file, head.data(), head.size()) &&
              write_all(file, fingerprint.data(), fingerprint.size()) &&
              write_all(file, &next_id, 4U);
    auto emit = [&](RecordKind kind, std::uint32_t ref, const void* body,
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
        ok = ok && write_all(file, prefix.data(), prefix.size()) &&
             (body_bytes == 0 || write_all(file, body, body_bytes)) && write_all(file, &crc, 4U);
    };
    for (const auto& [ref, record] : blocks_) {
        if (record.finished) { emit(RecordKind::Block, ref, &record.identity, sizeof(record.identity)); }
    }
    for (const auto& [ref, record] : snapshots_) {
        if (record.committed) {
            emit(RecordKind::SnapshotCommit, ref, &record.identity, sizeof(record.identity));
        }
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
    // The checkpoint covers everything: the journal restarts empty.
    std::filesystem::remove(journal_path(options_.directory), error);
    journal_records_    = 0;
    journal_since_ckpt_ = 0;
}

void PrefixDiskCatalog::replay_stream(std::FILE* file, bool checkpoint) {
    std::array<char, 8> magic{};
    if (!read_exact(file, magic.data(), magic.size()) || magic != kMagic) {
        if (checkpoint) { foreign_ = true; }
        return;
    }
    std::uint32_t version = 0;
    std::uint32_t length  = 0;
    if (!read_exact(file, &version, 4U) || !read_exact(file, &length, 4U) ||
        version != kFormatVersion || length > 4096U) {
        foreign_ = true;
        return;
    }
    std::string stored(length, '\0');
    if (length != 0 && !read_exact(file, stored.data(), length)) {
        foreign_ = true;
        return;
    }
    if (stored != fingerprint_) {
        foreign_ = true;
        return;
    }
    if (checkpoint) {
        if (!read_exact(file, &next_id_, 4U)) { foreign_ = true; }
    }
    for (;;) {
        std::array<unsigned char, kRecordPrefixBytes> prefix{};
        if (!read_exact(file, prefix.data(), prefix.size())) { break; } // clean EOF
        std::uint32_t ref = 0;
        std::uint32_t body_bytes = 0;
        std::memcpy(&ref, prefix.data() + 1, 4U);
        std::memcpy(&body_bytes, prefix.data() + 5, 4U);
        if (body_bytes > sizeof(BlockIdentityRec) + 64U) { break; } // torn or corrupt tail
        std::vector<std::byte> body(body_bytes);
        if (body_bytes != 0 && !read_exact(file, body.data(), body_bytes)) { break; }
        std::uint32_t crc = 0;
        if (!read_exact(file, &crc, 4U)) { break; }
        if (crc != 0U &&
            crc != DiskKVStore::crc32(std::span<const std::byte>(body.data(), body_bytes))) {
            break; // torn tail: everything before it stands
        }
        const auto kind = static_cast<RecordKind>(prefix[0]);
        if (kind == RecordKind::Block && body_bytes == sizeof(BlockIdentityRec)) {
            BlockRecord record;
            std::memcpy(&record.identity, body.data(), body_bytes);
            record.id       = ref;
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
            record.committed = true;
            if (ref >= next_id_) { next_id_ = ref + 1U; }
            snapshots_[ref]  = record;
        } else if (kind == RecordKind::Remove) {
            blocks_.erase(ref);
            snapshots_.erase(ref);
        }
        ++journal_records_;
    }
}

} // namespace ninfer::runtime::prefix_cache
