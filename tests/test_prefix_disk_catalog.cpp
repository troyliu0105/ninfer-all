#include "runtime/prefix_cache/disk_catalog.h"

#include <cstring>
#include <filesystem>
#include <iostream>
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
        std::filesystem::remove_all(root);
    } catch (const std::exception& error) {
        std::cerr << "prefix disk catalog test: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) { return 1; }
    std::cout << "prefix disk catalog: lifecycle, restart, torn tail, foreign, abandoned, "
                 "removal and capacity ok\n";
    return 0;
}
