#pragma once

// Persistent metadata of the hybrid prefix cache's disk tier
// (docs/maintainer/hybrid-prefix-cache-disk-tier.md §6).
//
// The catalog owns the model-independent identity of everything the tier stores: block-bundle
// records with their exact tokens, and snapshot closures (image segments plus an optional tail
// record) that become restartable only when their whole closure and metadata commit landed.
// Payload bytes live in two writer-owned DiskKVStores (blocks and tails in one, image segments
// in the other); the catalog is the only writer, so no automatic LRU ever removes a live record.
//
// Durability model: an append journal (`catalog.journal`) plus a periodic full checkpoint
// (`catalog.ckpt`). Every journal record is checksummed; a torn tail is ignored on replay and cut
// off, so the next append starts at the last complete record. The checkpoint and the journal keep
// separate replay state: a checkpoint's records are never mistaken for journal records, and a
// journal always begins with its own header. Each record carries a journal sequence and each
// checkpoint the sequence it covers, so a crash between replacing the checkpoint and removing the
// journal it covers replays that journal as already covered — never resurrecting a record the
// checkpoint has since dropped. No fsync: a crash may lose the last unpublished records, never
// publish a partial closure.
//
// Identity: record ids are monotonically allocated u32 values, persisted through `next_id` and
// never reused, so a stale journal entry can never alias a live record. Payload store keys are
// derived from the record id, not from token digests: exact token identity is decided here, in
// memory, against the stored tokens (§6 "Format and identity"). A crash can lose `next_id` while
// payload bytes survive, so open() recovers the allocator above every live payload key (a block, an
// image segment or a snapshot tail) and reclaims the payloads no replayed record reaches: a record
// id is never handed out while the payload store still answers to its key. A closure's optional
// tail needs no persisted flag — its payload is its record — so a surviving tail payload restores
// `has_tail` on replay.
//
// Threading: one I/O worker performs the writes and reads; the Engine worker owns metadata
// decisions. An internal mutex keeps those two roles from interleaving half-updated state.

#include "runtime/prefix_cache/block_hash.h"
#include "runtime/prefix_cache/prefix_index.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::runtime::prefix_cache {

class PrefixDiskCatalog {
public:
    // Payload geometries. `block_payload_bytes` covers one block bundle (all enabled pools); a
    // snapshot tail is one block record. `image_segment_bytes` is one image segment slot and
    // `image_segments` the fixed count a whole state image takes.
    struct Geometry {
        std::uint64_t block_payload_bytes = 0;
        std::uint64_t image_segment_bytes = 0;
        std::uint32_t image_segments      = 1;
    };

    struct Options {
        std::filesystem::path directory;
        std::uint64_t block_capacity_bytes  = 0;
        std::uint64_t image_capacity_bytes  = 0;
        // Checkpoint after this many journal records.
        std::uint32_t journal_flush_records = 256;
    };

    static constexpr std::uint32_t kNoRecord = 0xFFFFFFFFU;

    struct BlockIdentityRec {
        std::uint32_t parent                  = kNoRecord;
        std::uint64_t lookup_hash             = 0;
        std::uint64_t extra                   = 0;
        std::array<TokenId, kBlockTokens> tokens{};
    };

    struct SnapIdentityRec {
        std::uint32_t anchor                  = kNoRecord;
        std::uint32_t frontier                = 0;
        std::uint32_t tail_len                = 0;
        std::array<TokenId, kBlockTokens - 1> tail{};
        SnapshotKind kind                     = SnapshotKind::Tap;
        std::uint32_t hits                    = 0;
    };

    struct ReplayedBlock {
        std::uint32_t id = kNoRecord;
        BlockIdentityRec identity;
    };

    struct ReplayedSnapshot {
        std::uint32_t id = kNoRecord;
        SnapIdentityRec identity;
    };

    struct DiskCatalogStats {
        std::uint32_t blocks          = 0;
        std::uint32_t snapshots       = 0;
        std::uint32_t committed       = 0;
        std::uint32_t abandoned       = 0;
        std::uint32_t block_slots_free  = 0;
        std::uint32_t segment_slots_free = 0;
        std::uint64_t journal_records = 0;
        std::uint32_t next_id         = 1;
    };

    // Opens or creates the catalog under `options.directory`. `fingerprint` is the hybrid cache
    // fingerprint: a store written by a different fingerprint, layout or version opens as
    // foreign (open() returns false); the caller wipes before reuse — the directory is
    // artifact-scoped, so foreign data is never interpreted under this format. Throws on I/O
    // errors or a payload store whose geometry changed.
    PrefixDiskCatalog(Options options, Geometry geometry, const std::string& fingerprint);
    ~PrefixDiskCatalog();

    PrefixDiskCatalog(const PrefixDiskCatalog&)            = delete;
    PrefixDiskCatalog& operator=(const PrefixDiskCatalog&) = delete;

    // Replays header, checkpoint and journal into the replayed tables. False when the stored
    // fingerprint differs: the caller treats the directory as empty (wipe() before reuse).
    [[nodiscard]] bool open();

    // Erases the catalog and payload files, then reopens empty. For a foreign store.
    void wipe();

    // ---- block records -----------------------------------------------------------------------
    // Exact-token lookup among finished blocks: dedupes shared ancestors across closures.
    [[nodiscard]] std::optional<std::uint32_t> find_block(std::uint32_t parent,
                                                          std::uint64_t lookup_hash,
                                                          std::span<const TokenId> tokens,
                                                          std::uint64_t extra) const;
    // Reserves a record id and a payload slot. kNoRecord when the block store is full.
    [[nodiscard]] std::uint32_t begin_block_write();
    // Writes the payload bytes (one block bundle). False when the store is full; the record
    // stays reserved and must be abandoned.
    bool write_block(std::uint32_t ref, std::span<const std::byte> payload);
    // Publishes the record: journals it and makes it findable. The payload must have landed.
    void finish_block(std::uint32_t ref, const BlockIdentityRec& identity);
    // Drops an unfinished record (its payload slot returns to the pool). Idempotent.
    void abandon(std::uint32_t ref);

    // ---- snapshot closures ---------------------------------------------------------------------
    // Reserves a snapshot record id. kNoRecord when neither capacity nor id space remains.
    [[nodiscard]] std::uint32_t begin_snapshot_write();
    // False when the image store is full (the record stays reserved; abandon it).
    bool write_image_segment(std::uint32_t ref, std::uint32_t segment,
                             std::span<const std::byte> payload);
    bool write_tail(std::uint32_t ref, std::span<const std::byte> payload);
    // Journals the closure and marks it committed (restartable). The anchor path, image
    // segments and tail must have landed; their records must be finished/committed.
    void commit_snapshot(std::uint32_t ref, const SnapIdentityRec& identity);
    // Removes a record and its payloads (a snapshot takes its image segments and tail with
    // it). Journals the removal so a replay never resurrects it. No-op for unknown records.
    void remove(std::uint32_t ref);
    // Reclaims provisional (reserved, never finished) records: bounded abandoned-record
    // collection (§6 step 5). Returns the number reclaimed.
    std::uint32_t reclaim_abandoned();

    // ---- reads ----------------------------------------------------------------------------------
    // CRC-checked. Throw on a missing record or damaged payload.
    void read_block(std::uint32_t ref, std::span<std::byte> out) const;
    void read_image_segment(std::uint32_t ref, std::uint32_t segment,
                            std::span<std::byte> out) const;
    void read_tail(std::uint32_t ref, std::span<std::byte> out) const;
    [[nodiscard]] bool has_tail(std::uint32_t ref) const;

    // ---- state ----------------------------------------------------------------------------------
    // Replay results, valid after a successful open() until the first write.
    [[nodiscard]] const std::vector<ReplayedBlock>& replayed_blocks() const noexcept;
    [[nodiscard]] const std::vector<ReplayedSnapshot>& replayed_snapshots() const noexcept;
    [[nodiscard]] DiskCatalogStats stats() const;
    // Writes a fresh checkpoint and truncates the journal.
    void checkpoint();

private:
    struct BlockRecord {
        std::uint32_t id   = kNoRecord;
        BlockIdentityRec identity;
        // Journal sequence that published this record; a checkpoint re-emits it.
        std::uint64_t sequence = 0;
        bool finished          = false;
        bool reserved          = false;
    };

    struct SnapshotRecord {
        std::uint32_t id   = kNoRecord;
        SnapIdentityRec identity;
        std::uint64_t sequence = 0;
        bool committed         = false;
        bool reserved          = false;
        bool has_tail          = false;
    };

    enum class RecordKind : std::uint8_t { Block = 1, SnapshotCommit = 2, Remove = 3 };

    // What replay made of one stream: it parsed (`valid_bytes` receives the journal's durable
    // prefix length), its header never landed (`kUnusable`), or it belongs to another version or
    // execution fingerprint (`kForeign`).
    enum class ReplayOutcome : std::uint8_t { kOk, kUnusable, kForeign };

    class Store;

    [[nodiscard]] std::uint32_t allocate_id();
    // Appends one record and returns the journal sequence it was written under.
    [[nodiscard]] std::uint64_t journal_bytes(RecordKind kind, std::uint32_t ref, const void* body,
                                              std::size_t body_bytes);
    void write_checkpoint_locked();
    ReplayOutcome replay_stream(std::FILE* file, bool checkpoint, std::uint64_t* valid_bytes);
    // Recovers the allocator from the payload stores and reclaims the payloads that no replayed
    // record reaches, so a restart never reuses an id whose bytes survived the crash.
    void recover_payloads();
    void reset_state();

    Options options_;
    Geometry geometry_;
    std::string fingerprint_;

    mutable std::mutex mutex_;
    std::unique_ptr<Store> store_;
    std::unordered_map<std::uint32_t, BlockRecord> blocks_;
    std::unordered_map<std::uint32_t, SnapshotRecord> snapshots_;
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> children_;
    std::uint32_t next_id_           = 1;
    // Journal state: complete records in the journal, records since the last checkpoint, the byte
    // length of the journal's durable prefix (header plus complete records) — the offset the next
    // append rewrites from, which also drops any torn tail a crash left behind — and the highest
    // sequence the in-memory state already reflects (a checkpoint's covered sequence, or the last
    // journal record applied). A journal record at or below it is replayed as covered.
    std::uint64_t journal_records_     = 0;
    std::uint64_t journal_since_ckpt_  = 0;
    std::uint64_t journal_valid_bytes_ = 0;
    std::uint64_t journal_sequence_    = 0;
    bool opened_                       = false;
    bool foreign_                      = false;
    std::vector<ReplayedBlock> replayed_blocks_;
    std::vector<ReplayedSnapshot> replayed_snapshots_;
};

} // namespace ninfer::runtime::prefix_cache
