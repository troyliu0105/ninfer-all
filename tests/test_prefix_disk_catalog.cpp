#include "runtime/prefix_cache/disk_catalog.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::runtime::prefix_cache;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::filesystem::path fresh_directory() {
    const auto root = std::filesystem::temp_directory_path() /
                      ("ninfer_prefix_disk_catalog_" + std::to_string(std::random_device{}()));
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    return root;
}

constexpr std::uint64_t kBlockBytes  = 4096;
constexpr std::uint64_t kSegmentBytes = 8192;
constexpr std::uint32_t kSegments     = 2;

PrefixDiskCatalog::Options options_of(const std::filesystem::path& directory) {
    PrefixDiskCatalog::Options options;
    options.directory             = directory;
    options.block_capacity_bytes  = 16 * kBlockBytes;
    options.image_capacity_bytes  = 16 * kSegmentBytes;
    options.journal_flush_records = 4;
    return options;
}

PrefixDiskCatalog::Geometry geometry() {
    return PrefixDiskCatalog::Geometry{
        .block_payload_bytes = kBlockBytes,
        .image_segment_bytes = kSegmentBytes,
        .image_segments      = kSegments,
    };
}

std::vector<std::byte> payload(std::size_t bytes, std::uint32_t seed) {
    std::vector<std::byte> out(bytes);
    for (std::size_t i = 0; i < bytes; ++i) {
        out[i] = static_cast<std::byte>((i * 197U + seed) & 0xFFU);
    }
    return out;
}

std::vector<TokenId> block_tokens(std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<TokenId> tokens(kBlockTokens);
    for (TokenId& token : tokens) { token = static_cast<TokenId>(rng() % 50000U); }
    return tokens;
}

PrefixDiskCatalog::BlockIdentityRec block_identity(std::uint32_t parent, std::uint32_t seed) {
    PrefixDiskCatalog::BlockIdentityRec identity;
    identity.parent      = parent;
    identity.lookup_hash = 0x1234'5678'9ABC'DEF0ULL ^ seed;
    const auto tokens    = block_tokens(seed);
    std::copy(tokens.begin(), tokens.end(), identity.tokens.begin());
    return identity;
}

// A block record that finished, with its payload verified through a read.
std::uint32_t write_block(PrefixDiskCatalog& catalog, std::uint32_t parent, std::uint32_t seed) {
    const auto identity = block_identity(parent, seed);
    const std::uint32_t ref = catalog.begin_block_write();
    check(ref != PrefixDiskCatalog::kNoRecord, "block reservation succeeds");
    check(catalog.write_block(ref, payload(kBlockBytes, seed)), "block payload lands");
    catalog.finish_block(ref, identity);
    return ref;
}

// A session that ends abruptly: the process is killed, so no destructor runs — no shutdown
// checkpoint, no index flush. The sessions stay alive (deliberately leaked through a function
// pointer, never destroyed) so that the next open sees exactly what a killed process left.
PrefixDiskCatalog& crash_session(const PrefixDiskCatalog::Options& options,
                                 const PrefixDiskCatalog::Geometry& geometry,
                                 const std::string& fingerprint) {
    static auto* sessions = new std::vector<std::unique_ptr<PrefixDiskCatalog>>();
    sessions->push_back(std::make_unique<PrefixDiskCatalog>(options, geometry, fingerprint));
    PrefixDiskCatalog& catalog = *sessions->back();
    check(catalog.open(), "an abruptly stopped session opens");
    return catalog;
}

// A committed closure: an anchor block, every image segment, a partial-block tail and the commit.
std::uint32_t write_tailed_snapshot(PrefixDiskCatalog& catalog, std::uint32_t seed) {
    const std::uint32_t anchor = write_block(catalog, PrefixDiskCatalog::kNoRecord, seed);
    const std::uint32_t snap   = catalog.begin_snapshot_write();
    check(snap != PrefixDiskCatalog::kNoRecord, "snapshot reservation succeeds");
    for (std::uint32_t segment = 0; segment < kSegments; ++segment) {
        check(catalog.write_image_segment(snap, segment,
                                          payload(kSegmentBytes, seed + 1U + segment)),
              "image segment lands");
    }
    check(catalog.write_tail(snap, payload(kBlockBytes, seed + 3U)), "tail lands");
    PrefixDiskCatalog::SnapIdentityRec identity;
    identity.anchor   = anchor;
    identity.frontier = 2U * kBlockTokens + 5U;
    identity.tail_len = 5;
    identity.kind     = SnapshotKind::Boundary;
    identity.hits     = seed;
    const auto tokens = block_tokens(seed);
    std::copy_n(tokens.begin(), 5, identity.tail.begin());
    catalog.commit_snapshot(snap, identity);
    check(catalog.has_tail(snap), "the tail is recorded");
    return snap;
}

// The full lifecycle: write blocks and a snapshot closure, read them back, reopen, verify the
// replay reconstructs exactly the committed state.
void lifecycle_and_restart(const std::filesystem::path& root) {
    const auto directory = root / "lifecycle";
    const std::string fingerprint = "artifact=/m.ninfer;size=1;signature=abc;kv=bf16;build=1";
    std::uint32_t block_a = 0;
    std::uint32_t block_b = 0;
    std::uint32_t snap    = 0;
    {
        PrefixDiskCatalog catalog(options_of(directory), geometry(), fingerprint);
        check(catalog.open(), "an empty directory opens");
        block_a = write_block(catalog, PrefixDiskCatalog::kNoRecord, 1);
        block_b = write_block(catalog, block_a, 2);

        // find_block dedupes: the same identity under the same parent resolves to the record.
        const auto identity = block_identity(block_a, 2);
        const auto found    = catalog.find_block(
            block_a, identity.lookup_hash,
            std::span<const TokenId>(identity.tokens.data(), identity.tokens.size()),
            identity.extra);
        check(found.has_value() && *found == block_b, "find_block resolves exact tokens");

        const auto bytes_a = payload(kBlockBytes, 1);
        std::vector<std::byte> read_back(kBlockBytes);
        catalog.read_block(block_a, read_back);
        check(std::memcmp(read_back.data(), bytes_a.data(), kBlockBytes) == 0,
              "block payload reads back byte for byte");

        // A snapshot closure: image segments plus a tail, then the commit.
        snap = catalog.begin_snapshot_write();
        check(snap != PrefixDiskCatalog::kNoRecord, "snapshot reservation succeeds");
        check(catalog.write_image_segment(snap, 0, payload(kSegmentBytes, 3)),
              "image segment 0 lands");
        check(catalog.write_image_segment(snap, 1, payload(kSegmentBytes, 4)),
              "image segment 1 lands");
        check(catalog.write_tail(snap, payload(kBlockBytes, 5)), "tail lands");
        PrefixDiskCatalog::SnapIdentityRec snapshot;
        snapshot.anchor   = block_b;
        snapshot.frontier = 2 * kBlockTokens + 5;
        snapshot.tail_len = 5;
        snapshot.kind     = SnapshotKind::Boundary;
        snapshot.hits     = 7;
        const auto tokens = block_tokens(2);
        std::copy_n(tokens.begin(), 5, snapshot.tail.begin());
        catalog.commit_snapshot(snap, snapshot);
        const auto stats = catalog.stats();
        check(stats.committed == 1 && stats.blocks == 2, "stats see the committed closure");
        check(catalog.has_tail(snap), "the tail is recorded");
        catalog.checkpoint();

        std::vector<std::byte> segment(kSegmentBytes);
        catalog.read_image_segment(snap, 1, segment);
        check(std::memcmp(segment.data(), payload(kSegmentBytes, 4).data(), kSegmentBytes) == 0,
              "image segment reads back");
        std::vector<std::byte> tail(kBlockBytes);
        catalog.read_tail(snap, tail);
        check(std::memcmp(tail.data(), payload(kBlockBytes, 5).data(), kBlockBytes) == 0,
              "tail reads back");
    }
    {
        PrefixDiskCatalog catalog(options_of(directory), geometry(), fingerprint);
        check(catalog.open(), "reopen with the same fingerprint succeeds");
        const auto& blocks    = catalog.replayed_blocks();
        const auto& snapshots = catalog.replayed_snapshots();
        check(blocks.size() == 2, "both blocks replayed");
        check(snapshots.size() == 1 && snapshots.front().id == snap,
              "the committed closure replays");
        check(snapshots.front().identity.hits == 7 &&
                  snapshots.front().identity.kind == SnapshotKind::Boundary,
              "snapshot identity survives");
        check(blocks.front().identity.parent == PrefixDiskCatalog::kNoRecord &&
                  blocks.front().id < blocks.back().id,
              "blocks replay parents first");
        check(blocks.back().identity.parent == blocks.front().id,
              "the parent link survives the restart");

        // find_block still resolves the child under its parent after the restart.
        const auto identity = block_identity(blocks.back().id, 2);
        const auto found    = catalog.find_block(
            blocks.front().id, identity.lookup_hash,
            std::span<const TokenId>(identity.tokens.data(), identity.tokens.size()),
            identity.extra);
        check(found.has_value() && *found == blocks.back().id,
              "find_block works after the restart");

        std::vector<std::byte> read_back(kBlockBytes);
        catalog.read_block(blocks.front().id, read_back);
        check(std::memcmp(read_back.data(), payload(kBlockBytes, 1).data(), kBlockBytes) == 0,
              "payloads survive the restart");
    }
}

// A torn journal tail (truncated mid-record) replays everything before it.
void torn_journal_tail(const std::filesystem::path& root) {
    const auto directory   = root / "torn";
    const std::string fingerprint = "fingerprint-torn";
    std::uint32_t first    = 0;
    const auto journal     = directory / "catalog.journal";
    const auto mid_journal = directory / "journal.mid";
    {
        PrefixDiskCatalog::Options options = options_of(directory);
        options.journal_flush_records      = 1000; // keep every record in the journal
        PrefixDiskCatalog catalog(options, geometry(), fingerprint);
        check(catalog.open(), "open");
        first = write_block(catalog, PrefixDiskCatalog::kNoRecord, 10);
        write_block(catalog, first, 11);
        write_block(catalog, first, 12);
        // Snapshot the journal mid-session: the destructor's checkpoint replaces it.
        std::error_code error;
        std::filesystem::copy_file(journal, mid_journal,
                                   std::filesystem::copy_options::overwrite_existing, error);
        check(!error, "journal copied mid-session");
    }
    // Restore the journal alone and tear its last record: the destructor's checkpoint (which
    // holds all three blocks) is removed, so only the journal replays.
    {
        std::error_code error;
        std::filesystem::remove(directory / "catalog.ckpt", error);
        std::filesystem::copy_file(mid_journal, journal,
                                   std::filesystem::copy_options::overwrite_existing, error);
        check(!error, "journal restored");
        const auto size = std::filesystem::file_size(journal);
        std::filesystem::resize_file(journal, size - 1, error);
        check(!error, "journal truncated");
    }
    {
        PrefixDiskCatalog catalog(options_of(directory), geometry(), fingerprint);
        check(catalog.open(), "reopen after truncation");
        check(catalog.replayed_blocks().size() == 2,
              "the torn record is dropped, earlier ones stand");
    }
}

// A foreign fingerprint opens as foreign; wipe() re-enables the directory.
void foreign_fingerprint(const std::filesystem::path& root) {
    const auto directory = root / "foreign";
    {
        PrefixDiskCatalog catalog(options_of(directory), geometry(), "build-A");
        check(catalog.open(), "first build opens");
        write_block(catalog, PrefixDiskCatalog::kNoRecord, 20);
        catalog.checkpoint();
    }
    {
        PrefixDiskCatalog catalog(options_of(directory), geometry(), "build-B");
        check(!catalog.open(), "a foreign fingerprint reports itself");
        check(catalog.replayed_blocks().empty(), "nothing replays from a foreign store");
        catalog.wipe();
        check(catalog.begin_block_write() != PrefixDiskCatalog::kNoRecord,
              "the wiped catalog accepts writes");
    }
}

// Reserved-but-unfinished records refuse reads, are reclaimable, and never replay.
void abandoned_records(const std::filesystem::path& root) {
    const auto directory = root / "abandoned";
    PrefixDiskCatalog::Options options = options_of(directory);
    options.journal_flush_records      = 1000; // keep everything in the journal
    const std::string fingerprint      = "fingerprint-abandoned";
    {
        PrefixDiskCatalog catalog(options, geometry(), fingerprint);
        check(catalog.open(), "open");
        const std::uint32_t finished = write_block(catalog, PrefixDiskCatalog::kNoRecord, 30);
        (void)finished;
        const std::uint32_t reserved = catalog.begin_block_write();
        check(reserved != PrefixDiskCatalog::kNoRecord, "a provisional record reserves");
        check(catalog.reclaim_abandoned() == 1, "the provisional record is reclaimed");
        check(catalog.stats().blocks == 1, "only the finished block remains");
    }
}

// Removing a snapshot drops its image and tail payloads with it.
void snapshot_removal(const std::filesystem::path& root) {
    const auto directory = root / "remove";
    const std::string fingerprint = "fingerprint-remove";
    std::uint32_t snap            = 0;
    {
        PrefixDiskCatalog catalog(options_of(directory), geometry(), fingerprint);
        check(catalog.open(), "open");
        const std::uint32_t anchor = write_block(catalog, PrefixDiskCatalog::kNoRecord, 40);
        snap                       = catalog.begin_snapshot_write();
        check(catalog.write_image_segment(snap, 0, payload(kSegmentBytes, 41)), "segment");
        check(catalog.write_image_segment(snap, 1, payload(kSegmentBytes, 42)), "segment");
        PrefixDiskCatalog::SnapIdentityRec identity;
        identity.anchor   = anchor;
        identity.frontier = kBlockTokens;
        catalog.commit_snapshot(snap, identity);
        catalog.remove(snap);
        bool threw = false;
        try {
            std::vector<std::byte> segment(kSegmentBytes);
            catalog.read_image_segment(snap, 0, segment);
        } catch (const std::exception&) { threw = true; }
        check(threw, "removed image payloads refuse reads");
        check(!catalog.has_tail(snap), "removed snapshot has no tail");
    }
    {
        PrefixDiskCatalog catalog(options_of(directory), geometry(), fingerprint);
        check(catalog.open(), "reopen");
        check(catalog.replayed_snapshots().empty(), "a removed snapshot never replays");
        check(catalog.replayed_blocks().size() == 1, "the anchor block remains");
    }
}

// A full block store refuses reservations (backpressure, not eviction).
void capacity_refusal(const std::filesystem::path& root) {
    const auto directory = root / "capacity";
    PrefixDiskCatalog::Options options = options_of(directory);
    options.block_capacity_bytes       = 3 * kBlockBytes;
    const std::string fingerprint      = "fingerprint-capacity";
    PrefixDiskCatalog catalog(options, geometry(), fingerprint);
    check(catalog.open(), "open");
    std::uint32_t placed = 0;
    for (std::uint32_t seed = 50; seed < 60; ++seed) {
        if (catalog.begin_block_write() == PrefixDiskCatalog::kNoRecord) { break; }
        ++placed;
    }
    check(placed <= 3, "reservations stop at capacity");
}

// A payload whose journal record never landed, in a session where a sibling's abandon persisted
// the store index eagerly: the crash loses next_id, so the reopen must recover the allocator above
// the surviving payload key and reclaim the bytes no record reaches. Otherwise the next write
// reuses the id, the store answers its reserved upsert with the stale payload, and the orphan's
// slot leaks forever.
void unjournaled_payload_alias(const std::filesystem::path& root) {
    const auto directory = root / "alias";
    PrefixDiskCatalog::Options options = options_of(directory);
    options.journal_flush_records      = 1000; // nothing auto-checkpoints before the crash
    const std::string fingerprint      = "fingerprint-alias";
    const std::uint32_t slots =
        static_cast<std::uint32_t>(options.block_capacity_bytes / kBlockBytes);
    std::uint32_t orphan = PrefixDiskCatalog::kNoRecord;
    {
        PrefixDiskCatalog& catalog = crash_session(options, geometry(), fingerprint);
        orphan = catalog.begin_block_write();
        check(orphan != PrefixDiskCatalog::kNoRecord, "the unjournaled payload reserves an id");
        check(catalog.write_block(orphan, payload(kBlockBytes, 60)), "the orphan payload lands");
        const std::uint32_t sibling = catalog.begin_block_write();
        check(sibling != PrefixDiskCatalog::kNoRecord, "the sibling reserves an id");
        check(catalog.write_block(sibling, payload(kBlockBytes, 61)), "the sibling payload lands");
        catalog.abandon(sibling);
        check(std::filesystem::exists(directory / "blocks.bin.idx"),
              "the abandon persisted the store index");
        check(catalog.stats().block_slots_free == slots - 1U,
              "the orphan still holds its slot before the crash");
    }
    {
        PrefixDiskCatalog catalog(options, geometry(), fingerprint);
        check(catalog.open(), "the reopen succeeds");
        check(catalog.replayed_blocks().empty(), "an unjournaled payload replays as nothing");
        check(catalog.stats().next_id > orphan, "the allocator recovers above the orphan id");
        check(catalog.stats().block_slots_free == slots,
              "recovery reclaims the unreachable payload's slot");
        const std::uint32_t fresh = catalog.begin_block_write();
        check(fresh != PrefixDiskCatalog::kNoRecord && fresh > orphan,
              "a fresh record never reuses a recovered id");
        const auto bytes = payload(kBlockBytes, 62);
        check(catalog.write_block(fresh, bytes), "the fresh payload lands");
        std::vector<std::byte> read_back(kBlockBytes);
        catalog.read_block(fresh, read_back);
        check(std::memcmp(read_back.data(), bytes.data(), kBlockBytes) == 0,
              "a fresh record never reads the crashed write's bytes");
        catalog.finish_block(fresh, block_identity(PrefixDiskCatalog::kNoRecord, 62));
    }
}

// Every payload key kind an interrupted session can leave — a block, an image segment and a
// snapshot tail — consumes its id and holds its slot until recovery reclaims what no record
// reaches.
void orphan_keys_recover_allocator(const std::filesystem::path& root) {
    const auto directory = root / "orphan-keys";
    PrefixDiskCatalog::Options options = options_of(directory);
    const std::string fingerprint      = "fingerprint-orphan-keys";
    const std::uint32_t block_slots =
        static_cast<std::uint32_t>(options.block_capacity_bytes / kBlockBytes);
    const std::uint32_t segment_slots =
        static_cast<std::uint32_t>(options.image_capacity_bytes / kSegmentBytes);
    std::uint32_t snap = PrefixDiskCatalog::kNoRecord;
    {
        PrefixDiskCatalog& catalog = crash_session(options, geometry(), fingerprint);
        const std::uint32_t block = catalog.begin_block_write();
        check(catalog.write_block(block, payload(kBlockBytes, 90)), "the block payload lands");
        snap = catalog.begin_snapshot_write();
        check(snap != PrefixDiskCatalog::kNoRecord, "the snapshot reservation succeeds");
        check(catalog.write_image_segment(snap, 0, payload(kSegmentBytes, 91)),
              "the image segment lands");
        check(catalog.write_tail(snap, payload(kBlockBytes, 92)), "the tail lands");
        // Nothing is finished or committed, so the crash leaves no metadata: the payload keys are
        // the only trace of these ids, and only the tail's own bytes record that a tail exists.
    }
    {
        PrefixDiskCatalog catalog(options, geometry(), fingerprint);
        check(catalog.open(), "the reopen succeeds");
        check(catalog.replayed_blocks().empty() && catalog.replayed_snapshots().empty(),
              "an interrupted session replays nothing");
        check(catalog.stats().next_id > snap, "the allocator recovers above every payload key");
        check(catalog.stats().block_slots_free == block_slots &&
                  catalog.stats().segment_slots_free == segment_slots,
              "every unreachable payload returns its slot");
        const std::uint32_t fresh = catalog.begin_block_write();
        check(fresh != PrefixDiskCatalog::kNoRecord && fresh > snap,
              "a fresh record never reuses a recovered id");
        const auto bytes = payload(kBlockBytes, 93);
        check(catalog.write_block(fresh, bytes), "the fresh payload lands");
        std::vector<std::byte> read_back(kBlockBytes);
        catalog.read_block(fresh, read_back);
        check(std::memcmp(read_back.data(), bytes.data(), kBlockBytes) == 0,
              "the fresh record reads its own bytes");
        catalog.finish_block(fresh, block_identity(PrefixDiskCatalog::kNoRecord, 93));
    }
}

// A checkpoint's records are not journal records: after a reopen the next record still starts a
// journal with its own header, so an abrupt stop after that write replays it.
void journal_after_checkpoint_reopen(const std::filesystem::path& root) {
    const auto directory = root / "journal-reopen";
    PrefixDiskCatalog::Options options = options_of(directory);
    options.journal_flush_records      = 1000;
    const std::string fingerprint      = "fingerprint-journal-reopen";
    std::uint32_t first  = PrefixDiskCatalog::kNoRecord;
    std::uint32_t second = PrefixDiskCatalog::kNoRecord;
    {
        PrefixDiskCatalog catalog(options, geometry(), fingerprint);
        check(catalog.open(), "open");
        first = write_block(catalog, PrefixDiskCatalog::kNoRecord, 70);
        catalog.checkpoint();
    }
    check(!std::filesystem::exists(directory / "catalog.journal"),
          "the checkpoint left no journal behind");
    {
        PrefixDiskCatalog& catalog = crash_session(options, geometry(), fingerprint);
        check(catalog.replayed_blocks().size() == 1, "the checkpoint's own records replay");
        check(catalog.stats().journal_records == 0,
              "checkpoint records never count as journal records");
        second = write_block(catalog, first, 71);
        check(catalog.stats().journal_records == 1, "the journal holds the new record");
        std::vector<std::byte> read_back(kBlockBytes);
        catalog.read_block(second, read_back);
        check(std::memcmp(read_back.data(), payload(kBlockBytes, 71).data(), kBlockBytes) == 0,
              "the post-checkpoint payload lands under its own key");
    }
    {
        PrefixDiskCatalog catalog(options, geometry(), fingerprint);
        check(catalog.open(), "the reopen after an abrupt stop succeeds");
        check(catalog.replayed_blocks().size() == 2,
              "the record written after the reopen replays");
        check(catalog.replayed_blocks().back().id == second,
              "the replayed record is the one written after the reopen");
    }
}

// A torn journal tail must not push later records behind it: the reopen cuts the tail off, so the
// next append continues the durable prefix and both records replay after the next abrupt stop.
void journal_append_after_torn_tail(const std::filesystem::path& root) {
    const auto directory = root / "journal-torn-append";
    const auto journal   = directory / "catalog.journal";
    PrefixDiskCatalog::Options options = options_of(directory);
    options.journal_flush_records      = 1000;
    const std::string fingerprint      = "fingerprint-torn-append";
    std::uint32_t second = PrefixDiskCatalog::kNoRecord;
    std::uint32_t third  = PrefixDiskCatalog::kNoRecord;
    {
        PrefixDiskCatalog& catalog = crash_session(options, geometry(), fingerprint);
        const std::uint32_t first = write_block(catalog, PrefixDiskCatalog::kNoRecord, 72);
        second                    = write_block(catalog, first, 73);
    }
    {
        std::error_code error;
        const auto size = std::filesystem::file_size(journal, error);
        check(!error && size > 1, "the journal holds both records");
        std::filesystem::resize_file(journal, size - 1, error);
        check(!error, "the journal's last record is torn");
    }
    {
        PrefixDiskCatalog& catalog = crash_session(options, geometry(), fingerprint);
        check(catalog.replayed_blocks().size() == 1, "the torn record is dropped");
        check(catalog.stats().next_id > second,
              "the dropped record's payload still consumes its id");
        third = write_block(catalog, PrefixDiskCatalog::kNoRecord, 74);
        check(third > second, "the appended record takes an id of its own");
        std::vector<std::byte> read_back(kBlockBytes);
        catalog.read_block(third, read_back);
        check(std::memcmp(read_back.data(), payload(kBlockBytes, 74).data(), kBlockBytes) == 0,
              "the appended payload lands under its own key");
    }
    {
        PrefixDiskCatalog catalog(options, geometry(), fingerprint);
        check(catalog.open(), "the reopen succeeds");
        check(catalog.replayed_blocks().size() == 2,
              "the record appended after the torn tail replays");
        check(catalog.replayed_blocks().back().id == third,
              "the replayed record is the one appended after the torn tail");
    }
}

// A committed closure's tail has no persisted flag: its payload is its record, so the reopen
// restores it — cleanly or after an abrupt stop — and removing the closure returns its slot.
void tailed_snapshot_restart(const std::filesystem::path& root) {
    for (const bool crash : {false, true}) {
        const auto directory = root / (crash ? "tail-crash" : "tail-clean");
        PrefixDiskCatalog::Options options = options_of(directory);
        options.journal_flush_records      = 1000;
        const std::string fingerprint      = "fingerprint-tail";
        std::uint32_t snap                 = PrefixDiskCatalog::kNoRecord;
        if (crash) {
            PrefixDiskCatalog& catalog = crash_session(options, geometry(), fingerprint);
            snap                       = write_tailed_snapshot(catalog, 80);
        } else {
            PrefixDiskCatalog catalog(options, geometry(), fingerprint);
            check(catalog.open(), "open");
            snap = write_tailed_snapshot(catalog, 80);
        }
        {
            PrefixDiskCatalog catalog(options, geometry(), fingerprint);
            check(catalog.open(), "the reopen succeeds");
            check(catalog.replayed_snapshots().size() == 1, "the committed closure replays");
            check(catalog.has_tail(snap), "the committed tail survives the restart");
            std::vector<std::byte> tail(kBlockBytes);
            catalog.read_tail(snap, tail);
            check(std::memcmp(tail.data(), payload(kBlockBytes, 83).data(), kBlockBytes) == 0,
                  "the tail reads back after the restart");
            const std::uint32_t before = catalog.stats().block_slots_free;
            catalog.remove(snap);
            check(!catalog.has_tail(snap), "the removed closure keeps no tail");
            check(catalog.stats().block_slots_free == before + 1U,
                  "removing the closure returns its tail slot");
            bool threw = false;
            try {
                std::vector<std::byte> removed_tail(kBlockBytes);
                catalog.read_tail(snap, removed_tail);
            } catch (const std::exception&) { threw = true; }
            check(threw, "a removed tail refuses a read");
        }
    }
}

// A crash between the checkpoint's rename and the removal of the journal it covers. The stale
// journal holds records the checkpoint already reflects, so replaying it would apply them a second
// time — re-adding a record the checkpoint dropped (a removal, whose payload is evicted). The
// checkpoint's covered sequence makes the replay skip them; a record it does not cover still
// applies.
void checkpoint_journal_window(const std::filesystem::path& root) {
    const auto directory = root / "ckpt-window";
    const auto journal   = directory / "catalog.journal";
    const auto stale     = directory / "journal.stale";
    PrefixDiskCatalog::Options options = options_of(directory);
    options.journal_flush_records      = 1000; // every record stays in the journal
    const std::string fingerprint      = "fingerprint-ckpt-window";
    std::uint32_t kept    = PrefixDiskCatalog::kNoRecord;
    std::uint32_t removed = PrefixDiskCatalog::kNoRecord;
    {
        PrefixDiskCatalog catalog(options, geometry(), fingerprint);
        check(catalog.open(), "open");
        kept    = write_block(catalog, PrefixDiskCatalog::kNoRecord, 100);
        removed = write_block(catalog, kept, 101);
        catalog.remove(removed);
        std::error_code error;
        std::filesystem::copy_file(journal, stale,
                                   std::filesystem::copy_options::overwrite_existing, error);
        check(!error, "the journal of the covered records is kept");
        catalog.checkpoint();
    }
    auto restore_stale_journal = [&] {
        std::error_code error;
        std::filesystem::copy_file(stale, journal,
                                   std::filesystem::copy_options::overwrite_existing, error);
        check(!error, "the covered journal is restored where an interrupted removal leaves it");
    };
    auto covered_journal_replays_nothing = [&] {
        PrefixDiskCatalog catalog(options, geometry(), fingerprint);
        check(catalog.open(), "the reopen succeeds");
        check(catalog.replayed_blocks().size() == 1 &&
                  catalog.replayed_blocks().front().id == kept,
              "a covered journal replays nothing beyond the checkpoint");
        check(catalog.stats().next_id > removed, "covered records do not rewind the allocator");
        const auto identity = block_identity(kept, 101);
        check(!catalog.find_block(kept, identity.lookup_hash,
                                  std::span<const TokenId>(identity.tokens.data(),
                                                           identity.tokens.size()),
                                  identity.extra)
                   .has_value(),
              "the removed record does not come back");
    };
    restore_stale_journal();
    covered_journal_replays_nothing();
    // A killed append leaves the journal's last record torn instead of whole: the covered records
    // before it must still be skipped rather than replayed over the checkpoint.
    restore_stale_journal();
    {
        std::error_code error;
        const auto size = std::filesystem::file_size(journal, error);
        check(!error && size > 1, "the covered journal holds every record");
        std::filesystem::resize_file(journal, size - 1, error);
        check(!error, "the covered journal's last record is torn");
    }
    covered_journal_replays_nothing();
}

} // namespace

int main() {
    try {
        const std::filesystem::path root = fresh_directory();
        lifecycle_and_restart(root);
        torn_journal_tail(root);
        foreign_fingerprint(root);
        abandoned_records(root);
        snapshot_removal(root);
        capacity_refusal(root);
        unjournaled_payload_alias(root);
        orphan_keys_recover_allocator(root);
        journal_after_checkpoint_reopen(root);
        journal_append_after_torn_tail(root);
        tailed_snapshot_restart(root);
        checkpoint_journal_window(root);
        std::filesystem::remove_all(root);
    } catch (const std::exception& error) {
        std::cerr << "prefix disk catalog test: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) { return 1; }
    std::cout << "prefix disk catalog: lifecycle, restart, torn tail, foreign, abandoned, "
                 "removal, capacity, crash recovery, allocator recovery and covered journal ok\n";
    return 0;
}
