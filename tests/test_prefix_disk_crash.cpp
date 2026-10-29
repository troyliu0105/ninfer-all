// Abrupt-process crash and reopen coverage for the hybrid prefix cache's disk tier catalog
// (docs/maintainer/hybrid-prefix-cache-disk-tier.md §6).
//
// Every killed session here is a real forked child process that stops without running any
// destructor — SIGKILL from the parent, or the child's own _exit() — so no shutdown checkpoint and
// no index flush can run: the reopen sees exactly what an abrupt process death leaves on disk. The
// child's statically owned session stays alive until process termination; _exit() and SIGKILL
// skip static destruction as well as stack unwinding.
// Parent and child synchronize through pipes (a phase message with record ids, a command byte),
// never through sleeps, and the parent reaps every child even when an assertion fails.
//
// Interruption windows no barrier can time from outside — inside the checkpoint's
// write-rename-unpublish sequence, between a removal's journal append and its payload eviction —
// are assembled from real production bytes captured at real production barriers: journal bytes a
// killed session appended, a checkpoint a live session wrote, store files copied while the record
// was still alive. No hand-made record formats; nothing pretends a before/after checkpoint barrier
// proves an interruption inside the checkpoint. Expected bytes and ids come from deterministic
// oracles recomputed in the parent, and every comparison is byte-exact.
//
// Standalone: compiles with disk_catalog.cpp, disk_kv_store.cpp, block_hash.cpp and
// prefix_index.cpp (pure C++; no CUDA, no model inference). The crash paths need fork(), SIGKILL
// and poll(); a host without them reports the limit and exits 77.

#include "runtime/prefix_cache/disk_catalog.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::runtime::prefix_cache;

#if defined(__linux__)
#define NINFER_CRASH_FORK_TESTS 1
#else
#define NINFER_CRASH_FORK_TESTS 0
#endif

#if NINFER_CRASH_FORK_TESTS

#include <algorithm>
#include <array>
#include <cerrno>
#include <functional>
#include <stdexcept>
#include <utility>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

void check(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

template <typename Fn>
bool throws_exception(Fn&& fn) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

// A scratch directory that removes itself, even when an assertion fails mid-scenario.
class TempDir {
public:
    TempDir() {
        path_ = std::filesystem::temp_directory_path() /
                ("ninfer_prefix_disk_crash_" + std::to_string(std::random_device{}()));
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    TempDir(const TempDir&)            = delete;
    TempDir& operator=(const TempDir&) = delete;
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

constexpr std::uint64_t kBlockBytes   = 4096;
constexpr std::uint64_t kSegmentBytes = 8192;
constexpr std::uint32_t kSegments     = 2;

PrefixDiskCatalog::Options options_of(const std::filesystem::path& directory) {
    PrefixDiskCatalog::Options options;
    options.directory = directory;
    options.block_capacity_bytes = 16 * kBlockBytes;
    options.image_capacity_bytes = 16 * kSegmentBytes;
    // Killed sessions must never auto-checkpoint before the parent's SIGKILL lands.
    options.journal_flush_records = 1000;
    return options;
}

PrefixDiskCatalog::Geometry geometry() {
    return PrefixDiskCatalog::Geometry{
        .block_payload_bytes = kBlockBytes,
        .image_segment_bytes = kSegmentBytes,
        .image_segments      = kSegments,
    };
}

std::uint32_t block_capacity() {
    return static_cast<std::uint32_t>(16 * kBlockBytes / kBlockBytes);
}

std::uint32_t segment_capacity() {
    return static_cast<std::uint32_t>(16 * kSegmentBytes / kSegmentBytes);
}

// The byte oracle: child and parent derive the same bytes from a seed alone.
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

// A snapshot closure's identity and payload seeds, shared by whichever process writes them and the
// parent that verifies them: segments take seed_base + 1 + segment, the tail takes
// seed_base + 1 + kSegments, the tail tokens derive from seed_base + 4.
struct ClosureSpec {
    std::uint32_t anchor    = PrefixDiskCatalog::kNoRecord;
    std::uint32_t frontier  = 0;
    std::uint32_t tail_len  = 0;
    std::uint32_t seed_base = 0;
    std::uint32_t hits      = 0;
    bool has_tail           = false;
};

PrefixDiskCatalog::SnapIdentityRec closure_identity(const ClosureSpec& spec) {
    PrefixDiskCatalog::SnapIdentityRec identity;
    identity.anchor   = spec.anchor;
    identity.frontier = spec.frontier;
    identity.tail_len = spec.tail_len;
    identity.kind     = SnapshotKind::Boundary;
    identity.hits     = spec.hits;
    if (spec.tail_len != 0) {
        const auto tokens = block_tokens(spec.seed_base + 4U);
        std::copy_n(tokens.begin(), spec.tail_len, identity.tail.begin());
    }
    return identity;
}

// Child-side writes report refusal through the return value; the child turns false into a failure
// phase instead of asserting (its checks would die with the process).
bool put_block(PrefixDiskCatalog& catalog, std::uint32_t parent, std::uint32_t seed,
               std::uint32_t& out) {
    out = catalog.begin_block_write();
    if (out == PrefixDiskCatalog::kNoRecord) { return false; }
    if (!catalog.write_block(out, payload(kBlockBytes, seed))) { return false; }
    catalog.finish_block(out, block_identity(parent, seed));
    return true;
}

bool land_block_payload(PrefixDiskCatalog& catalog, std::uint32_t seed, std::uint32_t& out) {
    out = catalog.begin_block_write();
    return out != PrefixDiskCatalog::kNoRecord &&
           catalog.write_block(out, payload(kBlockBytes, seed));
}

// Lands every payload of a closure without committing it: an interrupted commit window.
bool land_closure_payloads(PrefixDiskCatalog& catalog, const ClosureSpec& spec,
                           std::uint32_t& out) {
    out = catalog.begin_snapshot_write(spec.has_tail);
    if (out == PrefixDiskCatalog::kNoRecord) { return false; }
    for (std::uint32_t segment = 0; segment < kSegments; ++segment) {
        if (!catalog.write_image_segment(out, segment,
                                         payload(kSegmentBytes, spec.seed_base + 1U + segment))) {
            return false;
        }
    }
    if (spec.has_tail &&
        !catalog.write_tail(out, payload(kBlockBytes, spec.seed_base + 1U + kSegments))) {
        return false;
    }
    return true;
}

bool put_closure(PrefixDiskCatalog& catalog, const ClosureSpec& spec, std::uint32_t& out) {
    if (!land_closure_payloads(catalog, spec, out)) { return false; }
    catalog.commit_snapshot(out, closure_identity(spec));
    return true;
}

// Parent-side byte oracles.
void expect_block_bytes(const PrefixDiskCatalog& catalog, std::uint32_t ref, std::uint32_t seed,
                        const char* what) {
    std::vector<std::byte> read_back(kBlockBytes);
    catalog.read_block(ref, read_back);
    const auto expected = payload(kBlockBytes, seed);
    check(std::memcmp(read_back.data(), expected.data(), kBlockBytes) == 0, what);
}

void expect_segment_bytes(const PrefixDiskCatalog& catalog, std::uint32_t ref,
                          std::uint32_t segment, std::uint32_t seed, const char* what) {
    std::vector<std::byte> read_back(kSegmentBytes);
    catalog.read_image_segment(ref, segment, read_back);
    const auto expected = payload(kSegmentBytes, seed);
    check(std::memcmp(read_back.data(), expected.data(), kSegmentBytes) == 0, what);
}

void expect_tail_bytes(const PrefixDiskCatalog& catalog, std::uint32_t ref, std::uint32_t seed,
                       const char* what) {
    std::vector<std::byte> read_back(kBlockBytes);
    catalog.read_tail(ref, read_back);
    const auto expected = payload(kBlockBytes, seed);
    check(std::memcmp(read_back.data(), expected.data(), kBlockBytes) == 0, what);
}

void expect_closure_bytes(const PrefixDiskCatalog& catalog, const ClosureSpec& spec,
                          std::uint32_t snap) {
    for (std::uint32_t segment = 0; segment < kSegments; ++segment) {
        expect_segment_bytes(catalog, snap, segment, spec.seed_base + 1U + segment,
                             "image segment bytes match the oracle");
    }
    if (spec.has_tail) {
        expect_tail_bytes(catalog, snap, spec.seed_base + 1U + kSegments,
                          "tail bytes match the oracle");
    }
}

void copy_file_checked(const std::filesystem::path& from, const std::filesystem::path& to,
                       const char* what) {
    std::error_code error;
    std::filesystem::copy_file(from, to, std::filesystem::copy_options::overwrite_existing, error);
    check(!error, what);
}

// ---- subprocess scaffolding -------------------------------------------------------------------------

constexpr std::uint8_t kPhaseWorkDone = 1;
constexpr std::uint8_t kPhaseOpened   = 2;
constexpr std::uint8_t kPhaseRemoved  = 3;
constexpr std::uint8_t kPhaseFailed   = 0xFF;

// One pipe message: phase byte plus up to four record ids, a single atomic pipe write.
struct ChildMessage {
    std::array<std::uint32_t, 4> ids{};
    std::uint8_t phase = 0;
};

bool write_all_fd(int fd, const void* data, std::size_t bytes) {
    const auto* out  = static_cast<const unsigned char*>(data);
    std::size_t done = 0;
    while (done < bytes) {
        const ssize_t written = ::write(fd, out + done, bytes - done);
        if (written <= 0) {
            if (written < 0 && errno == EINTR) { continue; }
            return false;
        }
        done += static_cast<std::size_t>(written);
    }
    return true;
}

bool read_full_fd(int fd, void* data, std::size_t bytes) {
    auto* out        = static_cast<unsigned char*>(data);
    std::size_t done = 0;
    while (done < bytes) {
        pollfd waiting{fd, POLLIN, 0};
        if (::poll(&waiting, 1, 60000) <= 0) { return false; } // bounded wait, never a sleep
        const ssize_t received = ::read(fd, out + done, bytes - done);
        if (received <= 0) {
            if (received < 0 && errno == EINTR) { continue; }
            return false;
        }
        done += static_cast<std::size_t>(received);
    }
    return true;
}

struct ChildSide {
    int out_fd = -1; // child -> parent phases
    int cmd_fd = -1; // parent -> child commands

    void notify(std::uint8_t phase, std::uint32_t a = 0, std::uint32_t b = 0, std::uint32_t c = 0,
                std::uint32_t d = 0) const {
        ChildMessage message;
        message.ids   = {a, b, c, d};
        message.phase = phase;
        (void)write_all_fd(out_fd, &message, sizeof(message));
    }
    void fail() const { notify(kPhaseFailed); }
    void wait_command() const {
        char command = 0;
        while (::read(cmd_fd, &command, 1) < 0 && errno == EINTR) {}
    }
};

struct ChildHandle {
    pid_t pid          = -1;
    int msg_fd         = -1;
    int cmd_fd         = -1;
    bool reaped        = false;
    int exit_status    = 0;
    std::uint8_t phase = 0;
    std::array<std::uint32_t, 4> ids{};

    ChildHandle()                              = default;
    ChildHandle(const ChildHandle&)            = delete;
    ChildHandle& operator=(const ChildHandle&) = delete;
    ChildHandle(ChildHandle&& other) noexcept
        : pid(std::exchange(other.pid, -1)), msg_fd(std::exchange(other.msg_fd, -1)),
          cmd_fd(std::exchange(other.cmd_fd, -1)), reaped(other.reaped),
          exit_status(other.exit_status), phase(other.phase), ids(other.ids) {}
    ~ChildHandle() {
        if (!reaped && pid > 0) { // reap even when an assertion failed mid-scenario
            ::kill(pid, SIGKILL);
            while (::waitpid(pid, &exit_status, 0) < 0 && errno == EINTR) {}
        }
        if (msg_fd >= 0) { ::close(msg_fd); }
        if (cmd_fd >= 0) { ::close(cmd_fd); }
    }

    // Bounded wait for the child's next phase; false on timeout, EOF or a failure phase.
    bool wait_for_phase() {
        ChildMessage message;
        if (!read_full_fd(msg_fd, &message, sizeof(message))) {
            check(false, "the child's phase message arrived in time");
            return false;
        }
        phase = message.phase;
        ids   = message.ids;
        if (phase == kPhaseFailed) {
            check(false, "the child reported a failure");
            return false;
        }
        return true;
    }

    bool command(char value) const { return write_all_fd(cmd_fd, &value, 1); }

    // SIGKILL: the child dies parked on its command pipe, so no destructor runs.
    void finish_by_kill() {
        if (reaped) { return; }
        ::kill(pid, SIGKILL);
        reap();
        check(WIFSIGNALED(exit_status) && WTERMSIG(exit_status) == SIGKILL,
              "the child died by SIGKILL");
    }

    // _exit(): also skips every destructor; exercises the child-initiated abrupt stop.
    void finish_by_exit() {
        if (reaped) { return; }
        (void)command('e');
        reap();
        check(WIFEXITED(exit_status) && WEXITSTATUS(exit_status) == 0,
              "the child left via _exit without destructors");
    }

private:
    void reap() {
        while (::waitpid(pid, &exit_status, 0) < 0 && errno == EINTR) {}
        reaped = true;
    }
};

// A child-owned session stays alive through the parent's barrier. The child always terminates
// with SIGKILL or _exit(), before static destruction; ordinary C++ ownership needs no leak.
PrefixDiskCatalog& crashed_session(const PrefixDiskCatalog::Options& options,
                                   const PrefixDiskCatalog::Geometry& geometry,
                                   const std::string& fingerprint) {
    static std::vector<std::unique_ptr<PrefixDiskCatalog>> sessions;
    sessions.push_back(std::make_unique<PrefixDiskCatalog>(options, geometry, fingerprint));
    PrefixDiskCatalog& catalog = *sessions.back();
    if (!catalog.open()) { throw std::runtime_error("the crashed session failed to open"); }
    return catalog;
}

// Runs `body` in a forked child. The child never returns from here: after its work it parks on a
// command read until the parent kills it or tells it to _exit. A dying parent leaves the child
// blocked until its pipe EOF, so no scenario can orphan one.
ChildHandle spawn(const std::function<void(const ChildSide&)>& body) {
    int up[2]   = {-1, -1};
    int down[2] = {-1, -1};
    if (::pipe(up) != 0) { throw std::runtime_error("pipe"); }
    if (::pipe(down) != 0) {
        ::close(up[0]);
        ::close(up[1]);
        throw std::runtime_error("pipe");
    }
    std::fflush(nullptr);
    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(up[0]);
        ::close(up[1]);
        ::close(down[0]);
        ::close(down[1]);
        throw std::runtime_error("fork");
    }
    if (pid == 0) {
        ::close(up[0]);
        ::close(down[1]);
        const ChildSide side{up[1], down[0]};
        try {
            body(side);
        } catch (...) {
            side.fail();
        }
        side.wait_command();
        ::close(side.out_fd);
        ::close(side.cmd_fd);
        _exit(0); // no destructors: no shutdown checkpoint, no index flush
    }
    ::close(up[1]);
    ::close(down[0]);
    ChildHandle handle;
    handle.pid    = pid;
    handle.msg_fd = up[0];
    handle.cmd_fd = down[1];
    return handle;
}

std::filesystem::path journal_of(const std::filesystem::path& directory) {
    return directory / "catalog.journal";
}

std::filesystem::path checkpoint_of(const std::filesystem::path& directory) {
    return directory / "catalog.ckpt";
}

std::filesystem::path checkpoint_tmp_of(const std::filesystem::path& directory) {
    return directory / "catalog.ckpt.tmp";
}

// ---- scenarios --------------------------------------------------------------------------------------

// A payload lands in the block store but its record is never journaled: the killed session leaves
// only bytes behind. The reopen must publish nothing, reclaim the slot, move the allocator above
// the orphan id, and never let a fresh record read the orphan's bytes.
void unpublished_payload_never_aliases(const std::filesystem::path& root) {
    const auto directory        = root / "unpublished";
    const std::string fingerprint = "crash-unpublished";
    const auto options          = options_of(directory);
    std::uint32_t orphan        = 0;
    {
        ChildHandle child = spawn([&](const ChildSide& side) {
            auto& catalog = crashed_session(options, geometry(), fingerprint);
            std::uint32_t ref = 0;
            if (!land_block_payload(catalog, 491, ref)) { side.fail(); return; }
            side.notify(kPhaseWorkDone, ref);
        });
        if (!child.wait_for_phase()) { child.finish_by_kill(); return; }
        orphan = child.ids[0];
        child.finish_by_kill();
    }
    check(std::filesystem::exists(directory / "blocks.bin"),
          "the killed session left payload bytes in the store");
    check(!std::filesystem::exists(journal_of(directory)) &&
              !std::filesystem::exists(checkpoint_of(directory)),
          "an unpublished payload journaled and checkpointed nothing");

    PrefixDiskCatalog reopened(options, geometry(), fingerprint);
    check(reopened.open(), "the reopen after the kill succeeds");
    check(reopened.replayed_blocks().empty(), "an unjournaled payload replays as nothing");
    check(reopened.stats().block_slots_free == block_capacity(),
          "the orphan's slot returns to the pool");
    check(reopened.stats().next_id > orphan, "the allocator recovers above the orphan id");
    const std::uint32_t fresh = reopened.begin_block_write();
    check(fresh != PrefixDiskCatalog::kNoRecord && fresh > orphan,
          "a fresh record takes an id of its own");
    const auto bytes = payload(kBlockBytes, 492);
    check(reopened.write_block(fresh, bytes), "the fresh payload lands");
    reopened.finish_block(fresh, block_identity(PrefixDiskCatalog::kNoRecord, 492));
    std::vector<std::byte> read_back(kBlockBytes);
    reopened.read_block(fresh, read_back);
    check(std::memcmp(read_back.data(), bytes.data(), kBlockBytes) == 0,
          "the fresh record reads its own bytes");
    check(std::memcmp(read_back.data(), payload(kBlockBytes, 491).data(), kBlockBytes) != 0,
          "the fresh record never reads the crashed write's bytes");
}

// The same crash with the orphan's key already listed in the persisted store index: a sibling's
// abandon flushed the index eagerly, so the reopen must reclaim the orphan from the live-key scan
// while every journaled record and its bytes survive.
void orphan_payload_with_persisted_index(const std::filesystem::path& root) {
    const auto directory          = root / "orphan-idx";
    const std::string fingerprint = "crash-orphan-idx";
    const auto options            = options_of(directory);
    std::uint32_t orphan          = 0;
    std::uint32_t finished        = 0;
    {
        ChildHandle child = spawn([&](const ChildSide& side) {
            auto& catalog = crashed_session(options, geometry(), fingerprint);
            std::uint32_t ref = 0;
            if (!land_block_payload(catalog, 501, ref)) { side.fail(); return; }
            const std::uint32_t crashed = ref;
            if (!put_block(catalog, PrefixDiskCatalog::kNoRecord, 502, ref)) { side.fail(); return; }
            const std::uint32_t journaled = ref;
            if (!land_block_payload(catalog, 503, ref)) { side.fail(); return; }
            catalog.abandon(ref); // persists the store index with the first orphan still there
            side.notify(kPhaseWorkDone, crashed, journaled, ref);
        });
        if (!child.wait_for_phase()) { child.finish_by_kill(); return; }
        orphan   = child.ids[0];
        finished = child.ids[1];
        child.finish_by_kill();
    }
    PrefixDiskCatalog reopened(options, geometry(), fingerprint);
    check(reopened.open(), "the reopen succeeds");
    check(reopened.replayed_blocks().size() == 1 &&
              reopened.replayed_blocks().front().id == finished,
          "only the journaled record replays");
    expect_block_bytes(reopened, finished, 502, "the journaled record's bytes survive");
    check(reopened.stats().block_slots_free == block_capacity() - 1U,
          "the orphans and the abandoned slot are reclaimed, the live record keeps its slot");
    // The cleanly abandoned id survives nowhere (its eviction is eager), so only the ids whose
    // payload keys survived — the orphan's and the live record's — must stay consumed.
    check(reopened.stats().next_id > finished,
          "the allocator recovers above every surviving payload key");
    const std::uint32_t fresh = reopened.begin_block_write();
    check(fresh != PrefixDiskCatalog::kNoRecord && fresh > orphan && fresh > finished,
          "no id whose payload key survived is ever handed out again");
    const auto bytes = payload(kBlockBytes, 504);
    check(reopened.write_block(fresh, bytes), "the fresh payload lands");
    reopened.finish_block(fresh, block_identity(PrefixDiskCatalog::kNoRecord, 504));
    expect_block_bytes(reopened, fresh, 504, "the fresh record reads its own bytes");
}

// Finished blocks plus a closure whose every payload landed but whose commit never did: the
// restart must find the blocks, never the snapshot, and hand back the closure's full capacity —
// which a fresh committed closure then really uses.
void uncommitted_closure_never_publishes(const std::filesystem::path& root) {
    const auto directory          = root / "uncommitted";
    const std::string fingerprint = "crash-uncommitted";
    const auto options            = options_of(directory);
    std::uint32_t anchor          = 0;
    std::uint32_t snap            = 0;
    {
        ChildHandle child = spawn([&](const ChildSide& side) {
            auto& catalog = crashed_session(options, geometry(), fingerprint);
            std::uint32_t ref = 0;
            if (!put_block(catalog, PrefixDiskCatalog::kNoRecord, 511, ref)) { side.fail(); return; }
            ClosureSpec spec;
            spec.anchor    = ref;
            spec.frontier  = kBlockTokens + 5U;
            spec.tail_len  = 5;
            spec.seed_base = 512;
            spec.hits      = 3;
            spec.has_tail  = true;
            std::uint32_t closure = 0;
            if (!land_closure_payloads(catalog, spec, closure)) { side.fail(); return; }
            side.notify(kPhaseWorkDone, ref, closure);
        });
        if (!child.wait_for_phase()) { child.finish_by_kill(); return; }
        anchor = child.ids[0];
        snap   = child.ids[1];
        child.finish_by_kill();
    }
    PrefixDiskCatalog reopened(options, geometry(), fingerprint);
    check(reopened.open(), "the reopen succeeds");
    check(reopened.replayed_blocks().size() == 1 && reopened.replayed_blocks().front().id == anchor,
          "the finished block survives the crash");
    check(reopened.replayed_snapshots().empty() && reopened.stats().snapshots == 0 &&
              reopened.stats().committed == 0,
          "the uncommitted closure publishes nothing on restart");
    check(reopened.stats().block_slots_free == block_capacity() - 1U &&
              reopened.stats().segment_slots_free == segment_capacity(),
          "the closure's image and tail slots are all reclaimable");
    check(!reopened.has_tail(snap), "the crashed closure keeps no tail");
    check(throws_exception([&] {
              std::vector<std::byte> segment(kSegmentBytes);
              reopened.read_image_segment(snap, 0, segment);
          }),
          "the crashed closure's images refuse reads");
    check(throws_exception([&] {
              std::vector<std::byte> tail(kBlockBytes);
              reopened.read_tail(snap, tail);
          }),
          "the crashed closure's tail refuses reads");

    ClosureSpec spec;
    spec.anchor    = anchor;
    spec.frontier  = kBlockTokens + 5U;
    spec.tail_len  = 5;
    spec.seed_base = 515;
    spec.hits      = 4;
    spec.has_tail  = true;
    std::uint32_t fresh = 0;
    check(put_closure(reopened, spec, fresh) && fresh > snap,
          "the reclaimed capacity carries a fresh complete closure");
    check(reopened.stats().committed == 1, "the fresh closure commits after the restart");
    expect_block_bytes(reopened, anchor, 511, "the surviving block's bytes are untouched");
    expect_closure_bytes(reopened, spec, fresh);
    reopened.checkpoint(); // clean shutdown path; the fresh closure must be durable
    PrefixDiskCatalog verified(options, geometry(), fingerprint);
    check(verified.open(), "the verification reopen succeeds");
    check(verified.replayed_blocks().size() == 1 &&
              verified.replayed_snapshots().size() == 1 &&
              verified.replayed_snapshots().front().id == fresh,
          "the post-restart closure is durable beside its anchor");
    expect_closure_bytes(verified, spec, fresh);
}

// A fully committed closure, killed before any destructor: blocks, image segments and the tail
// must come back byte-exact under stable ids, find_block must resolve, and the absence of a
// checkpoint proves the kill really skipped the shutdown path.
void committed_closure_exact_after_kill(const std::filesystem::path& root) {
    const auto directory          = root / "committed";
    const std::string fingerprint = "crash-committed";
    const auto options            = options_of(directory);
    std::uint32_t block_a         = 0;
    std::uint32_t block_b         = 0;
    std::uint32_t snap            = 0;
    ClosureSpec spec;
    {
        ChildHandle child = spawn([&](const ChildSide& side) {
            auto& catalog = crashed_session(options, geometry(), fingerprint);
            std::uint32_t ref = 0;
            if (!put_block(catalog, PrefixDiskCatalog::kNoRecord, 521, ref)) { side.fail(); return; }
            const std::uint32_t first = ref;
            if (!put_block(catalog, first, 522, ref)) { side.fail(); return; }
            ClosureSpec closing;
            closing.anchor    = ref;
            closing.frontier  = 2 * kBlockTokens + 5U;
            closing.tail_len  = 5;
            closing.seed_base = 523;
            closing.hits      = 52;
            closing.has_tail  = true;
            std::uint32_t closure = 0;
            if (!put_closure(catalog, closing, closure)) { side.fail(); return; }
            side.notify(kPhaseWorkDone, first, ref, closure);
        });
        if (!child.wait_for_phase()) { child.finish_by_kill(); return; }
        block_a = child.ids[0];
        block_b = child.ids[1];
        snap    = child.ids[2];
        child.finish_by_exit(); // the child's own abrupt stop, destructor-free
        spec.anchor    = block_b;
        spec.frontier  = 2 * kBlockTokens + 5U;
        spec.tail_len  = 5;
        spec.seed_base = 523;
        spec.hits      = 52;
        spec.has_tail  = true;
    }
    check(std::filesystem::exists(journal_of(directory)) &&
              !std::filesystem::exists(checkpoint_of(directory)) &&
              !std::filesystem::exists(checkpoint_tmp_of(directory)),
          "the abrupt stop left the journal and no shutdown checkpoint");

    PrefixDiskCatalog reopened(options, geometry(), fingerprint);
    check(reopened.open(), "the reopen after the kill succeeds");
    const auto& blocks    = reopened.replayed_blocks();
    const auto& snapshots = reopened.replayed_snapshots();
    check(blocks.size() == 2 && blocks.front().id == block_a && blocks.back().id == block_b,
          "both blocks replay under their original ids");
    check(blocks.front().identity.parent == PrefixDiskCatalog::kNoRecord &&
              blocks.back().identity.parent == block_a,
          "the parent links survive the kill");
    check(snapshots.size() == 1 && snapshots.front().id == snap,
          "the committed closure replays under its original id");
    const auto identity  = closure_identity(spec);
    const auto& replayed = snapshots.front().identity;
    check(replayed.anchor == spec.anchor && replayed.frontier == spec.frontier &&
              replayed.tail_len == spec.tail_len && replayed.kind == identity.kind &&
              replayed.hits == spec.hits,
          "the snapshot identity survives the kill field for field");
    check(std::equal(replayed.tail.begin(), replayed.tail.begin() + spec.tail_len,
                     identity.tail.begin()),
          "the snapshot tail tokens survive the kill");
    expect_block_bytes(reopened, block_a, 521, "the first block reads back byte for byte");
    expect_block_bytes(reopened, block_b, 522, "the second block reads back byte for byte");
    expect_closure_bytes(reopened, spec, snap);
    check(reopened.has_tail(snap), "the committed tail is recorded after the restart");
    check(reopened.stats().blocks == 2 && reopened.stats().committed == 1,
          "the replayed tables see exactly the committed state");

    const auto child_identity = block_identity(block_a, 522);
    const auto found          = reopened.find_block(
        block_a, child_identity.lookup_hash,
        std::span<const TokenId>(child_identity.tokens.data(), child_identity.tokens.size()),
        child_identity.extra);
    check(found.has_value() && *found == block_b, "find_block resolves the child after the kill");
}

// A removal journaled by a session that is then itself killed: the tombstone must survive the
// kill, the removed closure must never resurrect, and its capacity must be reusable.
void tombstone_survives_killed_remover(const std::filesystem::path& root) {
    const auto directory          = root / "tombstone";
    const std::string fingerprint = "crash-tombstone";
    const auto options            = options_of(directory);
    std::uint32_t anchor          = 0;
    std::uint32_t snap            = 0;
    {
        ChildHandle child = spawn([&](const ChildSide& side) {
            auto& catalog = crashed_session(options, geometry(), fingerprint);
            std::uint32_t ref = 0;
            if (!put_block(catalog, PrefixDiskCatalog::kNoRecord, 531, ref)) { side.fail(); return; }
            ClosureSpec spec;
            spec.anchor    = ref;
            spec.frontier  = kBlockTokens;
            spec.seed_base = 532;
            spec.hits      = 1;
            spec.has_tail  = false;
            std::uint32_t closure = 0;
            if (!put_closure(catalog, spec, closure)) { side.fail(); return; }
            side.notify(kPhaseWorkDone, ref, closure);
        });
        if (!child.wait_for_phase()) { child.finish_by_kill(); return; }
        anchor = child.ids[0];
        snap   = child.ids[1];
        child.finish_by_kill();
    }
    {
        ChildHandle remover = spawn([&](const ChildSide& side) {
            auto& catalog = crashed_session(options, geometry(), fingerprint);
            catalog.remove(snap); // journals the Remove record
            side.notify(kPhaseRemoved);
        });
        if (!remover.wait_for_phase()) { remover.finish_by_kill(); return; }
        remover.finish_by_kill(); // the tombstone session dies without a shutdown checkpoint
    }
    check(std::filesystem::exists(journal_of(directory)) &&
              !std::filesystem::exists(checkpoint_of(directory)),
          "the tombstone sits in the journal and no shutdown checkpoint ran");

    PrefixDiskCatalog reopened(options, geometry(), fingerprint);
    check(reopened.open(), "the reopen after the tombstone kill succeeds");
    check(reopened.replayed_snapshots().empty(), "the removed closure never resurrects");
    check(reopened.replayed_blocks().size() == 1 && reopened.replayed_blocks().front().id == anchor,
          "the anchor block survives the removal");
    check(reopened.stats().segment_slots_free == segment_capacity(),
          "the removed closure's image slots are free again");
    check(!reopened.has_tail(snap) &&
              throws_exception([&] {
                  std::vector<std::byte> segment(kSegmentBytes);
                  reopened.read_image_segment(snap, 0, segment);
              }),
          "the removed closure's payloads refuse reads");
    expect_block_bytes(reopened, anchor, 531, "the anchor's bytes are untouched by the removal");

    ClosureSpec spec;
    spec.anchor    = anchor;
    spec.frontier  = kBlockTokens;
    spec.seed_base = 535;
    spec.hits      = 2;
    spec.has_tail  = false;
    std::uint32_t fresh = 0;
    check(put_closure(reopened, spec, fresh) && fresh > snap,
          "the tombstoned capacity carries a fresh closure");
    expect_closure_bytes(reopened, spec, fresh);
}

// A journal-open failure makes the first removal throw without landing its record: no slot may be
// released for reuse, the retry must complete the removal, and a replay must not resurrect it.
void failed_removal_retries_then_stays_dead(const std::filesystem::path& root) {
    const auto directory          = root / "removal-retry";
    const std::string fingerprint = "crash-removal-retry";
    const auto options            = options_of(directory);
    std::uint32_t anchor          = 0;
    std::uint32_t snap            = 0;
    {
        ChildHandle child = spawn([&](const ChildSide& side) {
            auto& catalog = crashed_session(options, geometry(), fingerprint);
            std::uint32_t ref = 0;
            if (!put_block(catalog, PrefixDiskCatalog::kNoRecord, 541, ref)) { side.fail(); return; }
            ClosureSpec spec;
            spec.anchor    = ref;
            spec.frontier  = kBlockTokens;
            spec.seed_base = 542;
            spec.hits      = 3;
            spec.has_tail  = false;
            std::uint32_t closure = 0;
            if (!put_closure(catalog, spec, closure)) { side.fail(); return; }
            side.notify(kPhaseWorkDone, ref, closure);
        });
        if (!child.wait_for_phase()) { child.finish_by_kill(); return; }
        anchor = child.ids[0];
        snap   = child.ids[1];
        child.finish_by_kill();
    }
    {
        PrefixDiskCatalog catalog(options, geometry(), fingerprint);
        check(catalog.open(), "the removal session opens");
        catalog.checkpoint(); // clears the journal so a directory can impersonate it
        const auto before = catalog.stats();
        std::error_code error;
        std::filesystem::create_directory(journal_of(directory), error);
        check(!error, "the journal-open failure is injected");
        check(throws_exception([&] { catalog.remove(snap); }),
              "the failed journal publication is observable");
        check(catalog.stats().block_slots_free == before.block_slots_free &&
                  catalog.stats().segment_slots_free == before.segment_slots_free,
              "the failed removal releases no slot for reuse");
        check(std::filesystem::remove(journal_of(directory), error) && !error,
              "the injected failure is removed");
        catalog.remove(snap);
        const auto after = catalog.stats();
        check(after.segment_slots_free == before.segment_slots_free + kSegments &&
                  after.block_slots_free == before.block_slots_free,
              "the retry completes the removal and refunds every slot");
        catalog.remove(snap);
        check(catalog.stats().segment_slots_free == after.segment_slots_free,
              "the completed removal is idempotent");
    }
    PrefixDiskCatalog reopened(options, geometry(), fingerprint);
    check(reopened.open(), "the reopen succeeds");
    check(reopened.replayed_snapshots().empty(),
          "the replay does not resurrect the removed closure");
    check(reopened.replayed_blocks().size() == 1 && reopened.replayed_blocks().front().id == anchor,
          "the anchor block survives");
    expect_block_bytes(reopened, anchor, 541, "the anchor's bytes survive the removal");
}

// The crash window between a removal's journal append and its payload eviction: the Remove record
// landed but the store still answers to the payload's key. Assembled from real production bytes —
// the store files captured while the record was still alive, the journal captured after the
// removal — the reopen must apply the tombstone, reclaim the surviving bytes and never reuse the
// id.
void remove_journal_landed_but_payload_alive(const std::filesystem::path& root) {
    const auto directory          = root / "remove-window";
    const std::string fingerprint = "crash-remove-window";
    const auto options            = options_of(directory);
    std::uint32_t anchor          = 0;
    {
        ChildHandle child = spawn([&](const ChildSide& side) {
            auto& catalog = crashed_session(options, geometry(), fingerprint);
            std::uint32_t ref = 0;
            if (!put_block(catalog, PrefixDiskCatalog::kNoRecord, 551, ref)) { side.fail(); return; }
            side.notify(kPhaseWorkDone, ref);
        });
        if (!child.wait_for_phase()) { child.finish_by_kill(); return; }
        anchor = child.ids[0];
        child.finish_by_kill();
    }
    const auto blocks_file  = directory / "blocks.bin";
    const auto blocks_idx   = directory / "blocks.bin.idx";
    const auto pre_bin      = root / "remove-window-pre.bin";
    const auto pre_idx      = root / "remove-window-pre.idx";
    const auto post_journal = root / "remove-window-post.journal";
    {
        ChildHandle remover = spawn([&](const ChildSide& side) {
            auto& catalog = crashed_session(options, geometry(), fingerprint);
            side.notify(kPhaseOpened, anchor);
            side.wait_command(); // the parent captures the store while the record still lives
            catalog.remove(anchor);
            side.notify(kPhaseRemoved);
        });
        if (!remover.wait_for_phase()) { remover.finish_by_kill(); return; }
        copy_file_checked(blocks_file, pre_bin, "the pre-removal store bytes are captured");
        const bool have_index = std::filesystem::exists(blocks_idx);
        if (have_index) {
            copy_file_checked(blocks_idx, pre_idx, "the pre-removal store index is captured");
        }
        check(remover.command('g'), "the remover is told to proceed");
        if (!remover.wait_for_phase()) { remover.finish_by_kill(); return; }
        copy_file_checked(journal_of(directory), post_journal,
                          "the journal with Block and Remove records is captured");
        remover.finish_by_kill();
    }
    // Install the interruption state: the removal's journal landed, the eviction did not.
    std::error_code error;
    copy_file_checked(pre_bin, blocks_file, "the payload bytes are restored");
    if (std::filesystem::exists(pre_idx)) {
        copy_file_checked(pre_idx, blocks_idx, "the pre-removal store index is restored");
    }
    copy_file_checked(post_journal, journal_of(directory), "the post-removal journal is installed");
    std::filesystem::remove(checkpoint_of(directory), error);
    std::filesystem::remove(checkpoint_tmp_of(directory), error);

    PrefixDiskCatalog reopened(options, geometry(), fingerprint);
    check(reopened.open(), "the reopen succeeds");
    check(reopened.replayed_blocks().empty(),
          "the journaled Remove erases the record even though its bytes survived");
    check(reopened.stats().block_slots_free == block_capacity() &&
              reopened.stats().segment_slots_free == segment_capacity(),
          "the surviving payload bytes are reclaimed with their slot");
    check(reopened.stats().next_id > anchor, "the removed id is never handed out again");
    const std::uint32_t fresh = reopened.begin_block_write();
    check(fresh != PrefixDiskCatalog::kNoRecord && fresh > anchor, "a fresh id is allocated");
    const auto bytes = payload(kBlockBytes, 552);
    check(reopened.write_block(fresh, bytes), "the fresh payload lands in the reclaimed slot");
    reopened.finish_block(fresh, block_identity(PrefixDiskCatalog::kNoRecord, 552));
    expect_block_bytes(reopened, fresh, 552, "the fresh record reads its own bytes");
    check(std::memcmp(payload(kBlockBytes, 551).data(), bytes.data(), kBlockBytes) != 0,
          "the fresh record's bytes differ from the reclaimed record's");
}

// The checkpoint interruption window: the checkpoint replaced the old one but the journal it
// covers was not removed yet. A kill cannot be timed inside that rename-to-unpublish window, so
// the exact on-disk state is assembled from real production bytes: the journal a killed session
// appended (block, block, remove) and the checkpoint a live session wrote over it. The covered
// journal must replay as covered — no resurrection of the removed record — and a record appended
// behind the covered prefix, by a session that dies mid-journal, must survive a further kill.
void checkpoint_covered_journal_window(const std::filesystem::path& root) {
    const auto directory          = root / "ckpt-window";
    const std::string fingerprint = "crash-ckpt-window";
    const auto options            = options_of(directory);
    const auto stale_journal      = root / "ckpt-window-stale.journal";
    std::uint32_t kept            = 0;
    std::uint32_t dropped         = 0;
    {
        ChildHandle child = spawn([&](const ChildSide& side) {
            auto& catalog = crashed_session(options, geometry(), fingerprint);
            std::uint32_t ref = 0;
            if (!put_block(catalog, PrefixDiskCatalog::kNoRecord, 561, ref)) { side.fail(); return; }
            const std::uint32_t first = ref;
            if (!put_block(catalog, first, 562, ref)) { side.fail(); return; }
            const std::uint32_t second = ref;
            catalog.remove(second);
            side.notify(kPhaseWorkDone, first, second);
        });
        if (!child.wait_for_phase()) { child.finish_by_kill(); return; }
        kept    = child.ids[0];
        dropped = child.ids[1];
        child.finish_by_kill();
    }
    copy_file_checked(journal_of(directory), stale_journal,
                      "the journal the checkpoint will cover is captured");
    {
        PrefixDiskCatalog catalog(options, geometry(), fingerprint);
        check(catalog.open(), "the checkpoint session opens");
        check(catalog.replayed_blocks().size() == 1 &&
                  catalog.replayed_blocks().front().id == kept,
              "the session sees the kept block and the removal");
        catalog.checkpoint(); // replaces the checkpoint; publish drops the journal
    }
    check(std::filesystem::exists(checkpoint_of(directory)) &&
              !std::filesystem::exists(journal_of(directory)),
          "the checkpoint stands and the journal it covered is gone");
    copy_file_checked(stale_journal, journal_of(directory),
                      "the covered journal is restored where the interruption leaves it");

    PrefixDiskCatalog reopened(options, geometry(), fingerprint);
    check(reopened.open(), "the reopen over the covered journal succeeds");
    check(reopened.replayed_blocks().size() == 1 &&
              reopened.replayed_blocks().front().id == kept,
          "the covered journal replays nothing beyond the checkpoint");
    const auto dropped_identity = block_identity(kept, 562);
    check(!reopened.find_block(
                     kept, dropped_identity.lookup_hash,
                     std::span<const TokenId>(dropped_identity.tokens.data(),
                                              dropped_identity.tokens.size()),
                     dropped_identity.extra)
                  .has_value(),
          "the removed record does not come back from the covered journal");
    check(reopened.stats().next_id > dropped, "covered records do not rewind the allocator");
    check(reopened.stats().journal_records == 3,
          "the covered journal's three records are counted, not applied");

    std::uint32_t appended = 0;
    {
        ChildHandle child = spawn([&](const ChildSide& side) {
            auto& catalog = crashed_session(options, geometry(), fingerprint);
            if (catalog.stats().journal_records != 3) { side.fail(); return; }
            std::uint32_t ref = 0;
            if (!put_block(catalog, kept, 563, ref)) { side.fail(); return; }
            side.notify(kPhaseWorkDone, ref);
        });
        if (!child.wait_for_phase()) { child.finish_by_kill(); return; }
        appended = child.ids[0];
        child.finish_by_kill();
    }
    PrefixDiskCatalog verified(options, geometry(), fingerprint);
    check(verified.open(), "the final reopen succeeds");
    check(verified.replayed_blocks().size() == 2 &&
              verified.replayed_blocks().front().id == kept &&
              verified.replayed_blocks().back().id == appended,
          "the record appended behind the covered prefix replays");
    check(verified.stats().journal_records == 4,
          "the journal holds the three covered records plus the append");
    check(!verified.find_block(
                     kept, dropped_identity.lookup_hash,
                     std::span<const TokenId>(dropped_identity.tokens.data(),
                                              dropped_identity.tokens.size()),
                     dropped_identity.extra)
                  .has_value(),
          "the removed record stays dead under the covering checkpoint");
    expect_block_bytes(verified, appended, 563, "the appended block reads back byte for byte");
}

// A journal whose last record is torn and which no checkpoint covers: the reopen inside a fresh
// session drops the torn record, reclaims its orphaned payload, and an append behind the cut
// continues the durable prefix — the appended record survives the session's own kill while the
// torn one stays gone.
void torn_uncovered_tail_recovery(const std::filesystem::path& root) {
    const auto directory          = root / "torn-tail";
    const std::string fingerprint = "crash-torn-tail";
    const auto options            = options_of(directory);
    std::uint32_t first           = 0;
    std::uint32_t second          = 0;
    std::uint32_t torn            = 0;
    {
        ChildHandle child = spawn([&](const ChildSide& side) {
            auto& catalog = crashed_session(options, geometry(), fingerprint);
            std::uint32_t ref = 0;
            if (!put_block(catalog, PrefixDiskCatalog::kNoRecord, 571, ref)) { side.fail(); return; }
            const std::uint32_t a = ref;
            if (!put_block(catalog, a, 572, ref)) { side.fail(); return; }
            const std::uint32_t b = ref;
            if (!put_block(catalog, b, 573, ref)) { side.fail(); return; }
            side.notify(kPhaseWorkDone, a, b, ref);
        });
        if (!child.wait_for_phase()) { child.finish_by_kill(); return; }
        first  = child.ids[0];
        second = child.ids[1];
        torn   = child.ids[2];
        child.finish_by_kill();
    }
    {
        std::error_code error;
        const auto size = std::filesystem::file_size(journal_of(directory), error);
        check(!error && size > 1, "the killed session left a complete journal");
        std::filesystem::resize_file(journal_of(directory), size - 1, error);
        check(!error, "the journal's last record is torn mid-record");
    }
    std::uint32_t appended = 0;
    std::uint32_t replayed = 0;
    std::uint32_t next_id  = 0;
    std::uint32_t free_at  = 0;
    {
        ChildHandle child = spawn([&](const ChildSide& side) {
            auto& catalog = crashed_session(options, geometry(), fingerprint);
            replayed = static_cast<std::uint32_t>(catalog.replayed_blocks().size());
            next_id  = catalog.stats().next_id;
            free_at  = catalog.stats().block_slots_free;
            const auto torn_identity = block_identity(second, 573);
            if (catalog.find_block(
                                second, torn_identity.lookup_hash,
                                std::span<const TokenId>(torn_identity.tokens.data(),
                                                         torn_identity.tokens.size()),
                                torn_identity.extra)
                    .has_value()) {
                side.fail(); // the torn record must already be unfindable in this session
                return;
            }
            std::uint32_t ref = 0;
            if (!put_block(catalog, second, 574, ref)) { side.fail(); return; }
            side.notify(kPhaseWorkDone, ref, replayed, next_id, free_at);
        });
        if (!child.wait_for_phase()) { child.finish_by_kill(); return; }
        appended = child.ids[0];
        check(child.ids[1] == 2, "the torn record is dropped and everything before it stands");
        check(child.ids[2] > torn, "the torn record's payload still consumes its id");
        check(child.ids[3] == block_capacity() - 2U,
              "the torn record's orphaned slot is reclaimed, the replayed ones keep theirs");
        child.finish_by_kill();
    }
    PrefixDiskCatalog verified(options, geometry(), fingerprint);
    check(verified.open(), "the final reopen succeeds");
    check(verified.replayed_blocks().size() == 3 && verified.replayed_blocks()[0].id == first &&
              verified.replayed_blocks()[1].id == second &&
              verified.replayed_blocks()[2].id == appended,
          "the append behind the cut tail replays with its ancestors");
    const auto appended_identity = block_identity(second, 574);
    const auto found             = verified.find_block(
        second, appended_identity.lookup_hash,
        std::span<const TokenId>(appended_identity.tokens.data(), appended_identity.tokens.size()),
        appended_identity.extra);
    check(found.has_value() && *found == appended, "the appended block resolves under its parent");
    const auto still_torn = block_identity(second, 573);
    check(!verified.find_block(
                     second, still_torn.lookup_hash,
                     std::span<const TokenId>(still_torn.tokens.data(), still_torn.tokens.size()),
                     still_torn.extra)
                  .has_value(),
          "the torn record stays gone");
    check(verified.stats().block_slots_free == block_capacity() - 3U,
          "only the three live blocks hold slots");
    expect_block_bytes(verified, appended, 574, "the appended block reads back byte for byte");
}

// The checkpoint's rename interruption: the temporary checkpoint was fully written but the rename
// never happened, so no checkpoint stands, the journal still holds every record, and a leftover
// catalog.ckpt.tmp sits next to them. Built from real captured files, this state must replay the
// journal alone and leave the leftover untouched.
void checkpoint_tmp_leftover_ignored(const std::filesystem::path& root) {
    const auto directory          = root / "ckpt-tmp";
    const std::string fingerprint = "crash-ckpt-tmp";
    const auto options            = options_of(directory);
    const auto full_journal       = root / "ckpt-tmp-full.journal";
    std::uint32_t last            = 0;
    {
        ChildHandle child = spawn([&](const ChildSide& side) {
            auto& catalog = crashed_session(options, geometry(), fingerprint);
            std::uint32_t parent = PrefixDiskCatalog::kNoRecord;
            std::uint32_t ref    = 0;
            for (std::uint32_t seed = 581; seed < 585; ++seed) {
                if (!put_block(catalog, parent, seed, ref)) { side.fail(); return; }
                parent = ref;
            }
            side.notify(kPhaseWorkDone, ref);
        });
        if (!child.wait_for_phase()) { child.finish_by_kill(); return; }
        last = child.ids[0];
        child.finish_by_kill();
    }
    copy_file_checked(journal_of(directory), full_journal,
                      "the killed session's full journal is captured");
    {
        // The checkpoint whose rename an interruption would leave undone.
        ChildHandle checkpointer = spawn([&](const ChildSide& side) {
            auto& catalog = crashed_session(options, geometry(), fingerprint);
            catalog.checkpoint();
            side.notify(kPhaseWorkDone);
        });
        if (!checkpointer.wait_for_phase()) { checkpointer.finish_by_kill(); return; }
        checkpointer.finish_by_kill();
    }
    check(std::filesystem::exists(checkpoint_of(directory)) &&
              !std::filesystem::exists(journal_of(directory)),
          "the checkpoint session left its checkpoint and no journal");
    std::error_code error;
    copy_file_checked(checkpoint_of(directory), checkpoint_tmp_of(directory),
                      "the checkpoint bytes become the interrupted rename's leftover");
    std::filesystem::remove(checkpoint_of(directory), error);
    check(!error, "the standing checkpoint is gone, as before the rename");
    copy_file_checked(full_journal, journal_of(directory),
                      "the full journal stands, as before the rename");

    PrefixDiskCatalog reopened(options, geometry(), fingerprint);
    check(reopened.open(), "the reopen over the leftover tmp succeeds");
    check(reopened.replayed_blocks().size() == 4 && reopened.replayed_blocks().back().id == last,
          "the journal alone reconstructs the whole state; the tmp is never read");
    check(reopened.stats().next_id > last, "the allocator stands above every replayed record");
    check(std::filesystem::exists(checkpoint_tmp_of(directory)),
          "open leaves the leftover tmp untouched");
    check(!std::filesystem::exists(checkpoint_of(directory)),
          "open does not fabricate a checkpoint");
    expect_block_bytes(reopened, last, 584, "the last replayed block reads back byte for byte");
    const auto& chain = reopened.replayed_blocks();
    check(chain[1].identity.parent == chain[0].id && chain[2].identity.parent == chain[1].id &&
              chain[3].identity.parent == chain[2].id,
          "the replayed chain keeps its parent links");
}

} // namespace

#endif // NINFER_CRASH_FORK_TESTS

int main() {
#if NINFER_CRASH_FORK_TESTS
    try {
        const TempDir root;
        unpublished_payload_never_aliases(root.path());
        orphan_payload_with_persisted_index(root.path());
        uncommitted_closure_never_publishes(root.path());
        committed_closure_exact_after_kill(root.path());
        tombstone_survives_killed_remover(root.path());
        failed_removal_retries_then_stays_dead(root.path());
        remove_journal_landed_but_payload_alive(root.path());
        checkpoint_covered_journal_window(root.path());
        torn_uncovered_tail_recovery(root.path());
        checkpoint_tmp_leftover_ignored(root.path());
    } catch (const std::exception& error) {
        std::cerr << "prefix disk crash test: " << error.what() << '\n';
        return 1;
    }
    std::cout << "prefix disk crash: SIGKILL/_exit sessions, unpublished payload, uncommitted "
                 "closure, exact committed bytes, tombstone, removal windows, covered journal, "
                 "torn tail and checkpoint tmp ok\n";
    return 0;
#else
    std::cout << "prefix disk crash: these tests fork a child and SIGKILL it, which needs Linux; "
                 "skipping\n";
    return 77;
#endif
}
