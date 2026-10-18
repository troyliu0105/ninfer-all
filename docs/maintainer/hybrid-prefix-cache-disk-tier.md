# Hybrid prefix cache: disk tier

**Status: implemented.** This document defines the disk-tier extension to the
[Hybrid prefix cache](hybrid-prefix-cache-spec.md). Hybrid mode accepts `--disk-kv-path` with
`--disk-kv-gib` (default 64 GiB) and opens its own catalog under the artifact-scoped directory;
the Legacy-only `--disk-kv-restore` and `--disk-kv-directstorage` switches remain Legacy-only.
Host-only unit tests cover the index residency rules, the disk-committed tail fill, the Host-copy
reclamation a committed closure triggers, the source-ranking cost model and the catalog; the
demotion and preparation lifecycle is exercised by the real-engine scenarios
(`disk-demotion`, `disk-restore`, `disk-restart`, `disk-tail-restore`, `disk-small-capacity`,
`disk-long-chain`, plus the `disk-persist-exclusive` option check), and the performance contract
(§9) holds only within what those runs measured.

The extension retains cold prefixes after the pinned Host cache comes under pressure, serves
requests from Device, Host or Disk copies, and reconstructs the radix tree from committed disk
records after restart. It keeps one logical tree and isolates storage and asynchronous jobs from
model execution so that subsequent upstream rebases have a bounded integration surface.

The Hybrid specification remains the authority for implemented behavior. This document owns the
disk-specific contracts, including the command-line behavior in §8. The
[Engine architecture](engine-architecture.md) and
[paged KV contracts](paged-kv-cache.md) continue to govern request publication and physical pages.

## 1. Scope and decisions

The target is the existing single-device Qwen3.5 Hybrid path, including its supported KV formats,
Vision identity and speculative backends. Disk is a cold prefix-cache tier, not active-request
preemption, expert residency, a second model execution path or a shared cache across Engines.
All model arithmetic stays on the GPU.

The required behavior is:

- Start bounded asynchronous Host-to-Disk demotion before the Host cache exhausts its space.
- Match exact prefixes once, then source each required object from Device, Host or Disk in that
  order. Select a resume point by total predicted remaining time, not by frontier alone.
- Rebuild Disk-resident nodes and complete snapshots after restart without loading all payloads
  into RAM.
- Keep kernels, CUDA Graph keys, prefill decomposition, protocol cache hints and the request
  scheduler unchanged. Confine integration to cache residency, background preparation and final
  materialization.
- Reject simultaneous use of the disk tier and `--prefix-cache-file`.

A Qwen3.5 resume point needs a complete recurrent StateImage. KV alone cannot restore GDN state.
A reusable snapshot therefore requires its full ancestor path, its optional partial-block tail
and its StateImage, including the selected backend's state. Backend page commitment rules,
including MTP's frontier lag, remain unchanged.

### Architecture choice

Correctness, bounded interference and one source of residency truth take precedence over the
smallest initial diff. Rebase cost favors explicit modules and narrow integration points.

| Choice | Integration cost | Ongoing cost | Decision |
|---|---|---|---|
| One radix tree with per-tier copies | Extend residency, eviction and restoration in the existing index | One identity, dependency and pin model; disk mechanics stay in separate modules | Selected |
| Separate hot and cold trees | Preserve more of the two-tier index initially | Merge candidates, deduplicate publications and coordinate migration between two logical indexes | Rejected |

The selected design requires changes to existing two-tier assumptions. It does not introduce a
cache base class, storage-plugin framework or a copy of the Legacy owner/catalog planner.

## 2. Ownership and integration boundaries

| Owner | Responsibility |
|---|---|
| Core storage | Physical records, bounded I/O, payload checksums and read claims |
| `runtime/prefix_cache` | Exact tree identity, snapshot dependencies, per-tier residency, victim selection and model-independent persistent metadata |
| Qwen3.5 `HybridPrefixCache` | Bind opaque records to KV bundles and StateImage parts; own staging, copy events and asynchronous jobs |
| Program admission | Quote ready sources, validate resources, bind prepared data and activate a lane |
| Engine | FIFO, deadlines, cancellation, execution order and publication |

The Engine worker remains the only mutator of the tree and residency state. I/O workers receive
immutable job descriptions and return completion records. They do not release logical pages,
edit nodes or publish snapshots. The Engine must receive a completion notification that re-arms
head preparation/admission, including when no GPU unit is running; polling must not busy-spin or
rely solely on another request reaching a boundary.

Existing code offers these reuse boundaries:

- [`DiskKVStore`](../../src/core/disk_kv_store.h) is CUDA-independent and supplies checksummed
  records and read claims. Its automatic page LRU is unsuitable for live Hybrid dependencies;
  Hybrid needs caller-controlled allocation/reclamation that fails when no reserved space exists.
  Legacy behavior remains separate.
- [`DiskKVBridge`](../../src/core/disk_kv_bridge.h) supplies bounded queued writes and completion
  tickets, but its reads execute on the calling thread and its capacity split is fixed by family.
  Reuse physical mechanisms where appropriate, not this policy as the Hybrid tier policy.
- [`HybridPrefixCache`](../../src/models/qwen3_5/program/prefix/hybrid_cache.h) already owns Host
  slabs and Device transfers. Preserve its existing Host write-through and layer-pipelined
  restore paths.
- [`PrefixCacheIndex`](../../src/runtime/prefix_cache/prefix_index.h) already owns tokens, extra
  keys, references and eviction. Extend this index rather than recreating those facts elsewhere.

New implementation files should separate persistent metadata from model binding. For example,
`runtime/prefix_cache/disk_catalog.*` owns metadata encoding/replay, while
`models/qwen3_5/program/prefix/hybrid_disk.*` owns demotion and preparation jobs.

## 3. Residency, identity and capacity

### Copies and validity

Each node keeps an opaque Disk record reference in addition to its Device and Host references.
Snapshot metadata identifies its disk image parts and optional tail. Each location tracks
Absent, Filling or Resident independently; Disk is a location, not a fourth transfer state.

A Filling destination is not an available copy. A resident Disk record has completed its write
and identity/checksum publication; that does not imply power-loss durability. Reads validate the
record before any consumer uses its bytes.

The index distinguishes:

- Dropping one tier's copy while another valid copy remains.
- Removing a logical snapshot and updating its dependencies.
- Losing the last copy of a required object and invalidating affected resume points.

A node with only a Disk copy remains in the tree. A snapshot can resume through mixed residency:
its ancestors, tail and image need not occupy the same tier. Losing a corrupt Disk copy preserves
any intact Device/Host copies. Hashes remain lookup accelerators; exact tokens and extra keys,
including Vision identity, decide equality.

Centralize these predicates and transitions instead of scattering disk-enabled branches through
admission, capture and release. Existing backed/unbacked Device LRU membership must recognize a
valid Disk backup. Supersession and Host pressure must distinguish a tier drop from deliberate
logical deletion; neither may silently delete a recoverable cold entry as a side effect.

### Dependency accounting

Global snapshot reachability and a tier's reclaimable bytes are different facts. A block may be
live because a snapshot needs it but still yield its Host slab because Disk holds a copy.
Maintain enough per-tier dependency information to compute that marginal reclaimable space.
Do not keep a Host slab merely because a disk-backed descendant exists.

A committed disk snapshot is a stronger condition than mixed-tier runtime validity: its entire
path, image and tail must exist on Disk. Shared ancestors occupy one disk payload record.
Incomplete disk closures may exist while jobs run, but cannot advertise restart reuse.

### Budgets

Resolve and report these bounds at startup:

- Cache-owned Device pages and snapshot slots, respecting active execution entitlements.
- Host slabs available for cache storage after reserving disk staging.
- Staging bytes, queued/in-flight bytes and maximum Host bytes leased to offload jobs.
- Disk payload and metadata capacity, including journal/checkpoint and temporary-work headroom.
- In-memory node, snapshot and disk-directory capacity.

Account pinned staging within the advertised Host budget; do not add an unreported second RAM
cache. A conservative staging layout holds one full StateImage and a bounded number of KV copy
batches. Its size depends on model geometry, not prefix length. The remaining Host pool must
still satisfy its minimum geometry. Reject an insufficient configuration before serving.

The implementation reserves that staging out of `--host-cache-mib` *before* it sizes the slab
pool: [`plan_hybrid_disk_staging`](../../src/models/qwen3_5/program/prefix/hybrid_host_layout.cpp)
takes one StateImage in its physical slab segments, one block payload for the snapshot-tail slot,
and up to `kHybridDiskMaxStagingBlocks` (256) block payloads — where the block batch gets only
what the budget leaves after the slab pool's own minimum geometry, so a Host budget that is legal
without the disk tier stays legal with it. The tier pins the result once with `PinnedHostBuffer`
(no `make_unique` side allocation), `staging_bytes()` reports the same rounded bytes, the
`HostKvPin` startup phase's total is the advertised budget (slab pool plus staging), and a budget
that cannot hold a single block payload beside the slab minimum is rejected at startup with the
MiB it needs — the plan's own `min_budget_bytes` (image segments, tail payload, slab minimum and
one block payload), never a re-derived approximation. Disk capacity is not part of the Host
budget: the catalog's byte capacity stays real bytes.

The current node bound derives from Device pages plus Host slabs in
[`create_hybrid_prefix_cache`](../../src/models/qwen3_5/program/prefix/hybrid_program.cpp).
Disk-only entries invalidate that bound. Derive representable metadata limits from the new
capacity plan and report their RAM cost. Reaching a limit must reject new retention, not overflow
an arena or fail an otherwise executable request. Hot victim selection must not scan the entire
cold catalog as the disk tier grows.

## 4. Host pressure and asynchronous demotion

### Watermarks and victims

Use high/low occupancy watermarks with hysteresis. Start demotion above the high watermark and
stop selecting more work below the low watermark. Resolve the gap from snapshot/copy geometry
and the allowed in-flight bytes so that new Host writes have headroom. Percentages alone do not
protect a small pool that must receive a large StateImage. Qualify the actual thresholds; this
design does not claim a measured optimum.

Reclaim in this order:

1. Discard dead data that no snapshot or pending publication can use.
2. Release cold Host copies that already have valid Disk backups.
3. Schedule demotion of cold, useful data without a Disk copy.

Use existing supersession and GDSF concepts to choose cold candidates, while separating the
value of keeping a Host copy from the value of keeping a Disk copy. Host retention avoids a slow
restore; Disk retention avoids recomputation. The current Host score is not automatically the
right score for both tiers.

A committed closure is also the trigger for reclaiming what it duplicated: the tier releases the
snapshot's Host image and tail slabs through `release_snapshot_host_copy` and the anchor chain's
Host copies through `release_redundant_host_copies`, which keeps a chain node's copy while any
live snapshot anchored at or below it still lacks a committed disk closure. A full record
directory recycles instead of blocking: the tier runs the index's bounded disk victim selection
(`evict_disk_entries`, lowest-value committed closures first, each including the exclusive anchor
blocks it kept) plus `reclaim_abandoned` for reserved records no write will finish, and only then
treats the cycle as backpressure. A closure whose anchor or awaited record the index removed in
the meantime never commits, so a restart cannot resurrect a closure with a dangling anchor.

A closure that cannot commit returns everything it reserved in the same completion pass. The
completion step drops each awaited block record as its write lands, and a closure that will never
commit — a failed block write, a catalog commit failure, a chain record the index removed, or a
snapshot that left the index — closes its snapshot's disk fill through the index
(`abort_open_snapshot_fill`), whose release callback hands the image record back to the journaled
removal path. No snapshot is left Filling and no record stays reserved for a closure that cannot
publish; a failed closure keeps its Host copy, so the snapshot stays a valid volatile resume point.
A demotion candidate must also carry a complete Host copy to be written at all: the closure copies
the snapshot's image grid plus, when a tail exists, its tail slab, and a candidate whose slabs are
not exactly that set (an aborted Host fill, or a tail the Host tier never allocated) is skipped
rather than published as a half-closure.

### Write and publication protocol

A demotion plan targets useful snapshot closures but writes independent block bundles and image
parts in bounded batches:

1. Reserve disk space and copy the required immutable identity metadata into the job plan.
2. Lease only the Host payloads the batch reads. Require completed D2H writes before submitting
   those payloads to disk workers.
3. Write missing records, reusing committed ancestors rather than copying the whole prefix for
   each snapshot.
4. On completion, the worker publishes the new Disk copies and releases the batch's Host leases.
   It may drop those Host copies when the remaining runtime dependencies allow it.
5. Publish a restartable snapshot record only after its complete Disk closure exists. Protect
   all records required by that commit until publication completes.

Blocks can reach Disk before the associated snapshot closes. This permits a prefix larger than
the Host cache to accumulate on Disk as Host write-through advances. Bound provisional records
and pending metadata references; collect abandoned records when their publication can no longer
complete. Payloads without a committed snapshot never create restart hits by themselves.

The implementation must distinguish offload source leases from execution pins. A source lease
prevents reuse of the Host bytes under an accepted write, including after cancellation, without
pinning unrelated Device ancestors for the duration of disk I/O.

### Backpressure

A full queue stops new demotion submissions. It does not make Host allocation wait for SSD
completion. If Host pressure outpaces the permitted write rate, discard lower-value cache data
or skip new retention according to policy, while preserving in-flight sources. Active requests
keep priority over retention.

Sustained incoming spill bytes above the allowed disk write rate cannot coexist with bounded
memory and lossless retention. This design chooses bounded memory and best-effort retention.
Normal Host-to-Disk demotion reads existing Host bytes; it adds no model pass or snapshot tap.

## 5. Lookup, cost and cold preparation

### Lookup and source selection

Walk the radix tree once and compare exact identities as today. For each required object, prefer
Device, then Host, then Disk. A complete hot hit performs no filesystem probe, synchronous disk
checksum or wait for a metadata checkpoint. The worker-owned in-memory catalog supplies Disk
references without a lock held across storage operations.

This source order is not a rule to stop at the first shallow hot snapshot. Compare feasible
resume points, including root, by their remaining work:

```text
remaining_time(s) = remaining_preparation_time(s)
                  + prefill_time(frontier(s), remaining_prompt_tokens)
                  + activation_and_first_token_work(s)
```

Account for the disk read queue, outstanding preparation, source bytes, checksum work and H2D.
Use measured end-to-end restore coefficients with margin, not SSD sequential-read specifications
or the Host H2D coefficient. A useful Disk choice must beat the best other source by enough to
cover prediction uncertainty. Predictions rank choices; actual pages, slots and validity decide
whether they can execute.

### Prepare outside the global context transaction

Disk reads and cold preparation run while the FIFO head remains queued. They do not occupy the
Engine's global context transaction or an active lane. Preserve FIFO; do not add backfill or run a
second model prefill in a race against the read.

This boundary is required by the current Engine. In
[`engine_core.h`](../../src/runtime/engine/engine_core.h), `snapshot_cancellations`,
`settle_terminal_requests` and `cancel_active_requests` defer topology changes while a context
transaction exists. A long-lived disk materialization would delay other requests' cancellation,
completion and resource release even if their decode kernels continued to run.

Cold preparation proceeds as follows:

1. Lease the selected Disk records and establish a bounded, cancellable preparation job for the
   head. Disk read leases protect storage records and metadata, not the Device LRU path.
2. Read and verify a bounded batch into pinned staging. Coalesce suitable file extents and keep
   read concurrency bounded. Give demanded reads priority over background demotion writes.

   The batch is the staging pool, never the whole cold path: `stage_blocks` queues what the pool
   can hold now and the next cadence point queues more, so a chain longer than the pool (which is
   the normal case for a long conversation) advances a batch per cycle instead of requesting more
   slots than exist and making no progress. Requests are deduplicated against both the tier's
   queued/running/landed reads and the bytes the Program already holds for a landing batch. A
   staging slot recycles only after the copy that consumed it lands — the node is device-Resident
   again, or no batch reads it — and a duplicate release is ignored, so two nodes can never be
   handed the same slot.
3. When bytes are ready and spare cache capacity exists, allocate cache-owned Device
   destinations, submit H2D on the restore stream and mark those destinations Filling.
   No Device destination stays pinned while waiting for an unsubmitted disk read.
4. Publish copies after their CUDA events complete. Completed prefetched entries remain ordinary
   cache and may yield to active growth. Do not hold a whole cold path unevictable throughout a
   slow preparation.
5. Once the chosen path, image and tail are ready on Device or Host, request a fresh admission
   quote. The short existing transaction revalidates the source, reserves the execution
   entitlement and binds/activates the lane. It waits on no unfinished disk read.

The complete prefix need not fit in Host. Batches pass through staging to Device; optional Host
promotion follows normal retention policy rather than becoming a prerequisite for admission.
StateImage staging may hold a whole image so existing image copy primitives remain usable.
Device snapshot slots are acquired only for ready image copies, not while waiting on SSD reads.

Preparation is opportunistic, not a guarantee that a request can acquire its whole Device path
beside active lanes. If capacity is unavailable, leave the head queued, retry when resources
change, or choose another feasible source. No cold reservation may borrow active entitlements.
After any eviction, stale quote or changed source, re-evaluate the ready subset instead of
assuming prefetch reserved a future admission.

A remaining-time budget limits waiting. If the disk choice loses its advantage or the request's
deadline/cancellation intervenes, abandon that choice before admission and select another source
when appropriate. Accepted I/O and CUDA work retain their buffers until completion; abandoning
a request does not permit early buffer reuse.

### Transfer isolation

Keep disk preparation jobs separate from existing GPU transfer draining. Current `hybrid_quote`
and `hybrid_make_room` can call `HybridPrefixCache::drain()` in
[`hybrid_program.cpp`](../../src/models/qwen3_5/program/prefix/hybrid_program.cpp). Neither
`drain()`, `settle_prefetch()` nor an admission retry may start waiting for disk workers.

Only a batch whose disk bytes are ready may join the bounded CUDA copy lifecycle. Recycle its
staging only after the corresponding event. The compute stream receives no wait on an event
whose production depends on a future disk read. Preserve the existing Host-only layer-pipelined
restore; activation does not require introducing layer-pipelined disk consumption.

## 6. Persistent records and restart

### Format and identity

Store original KV and StateImage bytes without requantization. The persistent catalog needs:

| Record | Contents |
|---|---|
| Store header | Format version, store identity, execution fingerprint, payload geometry |
| Block | Stable object ID, parent ID, exact 64 tokens and extra key, payload reference |
| Snapshot | Anchor ID, exact frontier and tail tokens, kind, image parts and optional tail reference |
| Commit | Snapshot publication and the record generations it depends on |
| Removal | Retired snapshot or copy references and the generations they invalidate |

Runtime `NodeRef` generations, Device page IDs and Host slab indices are not persistent IDs.
Allocate stable record identities that do not alias a reused slot; recover allocator state
across restart, including records orphaned by an interrupted commit. A digest selects metadata
candidates but does not replace exact identity comparison. CRC detects payload damage, not token
identity collisions.

Use a versioned, checksummed append journal for catalog updates and a bounded periodic checkpoint
for restart cost. Batch updates; do not rewrite the entire catalog per KV block. Metadata I/O and
compaction run outside the Engine worker. Checkpoint replacement must preserve a recoverable
checkpoint/journal pair across interruption, including the journal position the checkpoint
covers. Reserve metadata/temporary space inside the configured store budget.

The fingerprint covers the artifact identity, KV/backend layouts, prefill/execution signature,
RoPE settings and implementation identity. Start with the existing Hybrid persistence's
conservative invalidation policy: a binary rebuild may invalidate reuse. Supporting reuse across
binary changes requires a separate demonstrated layout/semantics contract; this design does not
promise it. Use a distinct Hybrid store namespace/format, not the Legacy disk files. Report an
incompatible writable store and require a compatible location rather than overwriting foreign
records or interpreting their bytes under the new format.

### Commit, recovery and durability

Write payloads before their snapshot commit. A runtime Disk copy becomes readable only after
its record is complete; a restartable snapshot becomes visible only after its whole closure and
metadata commit complete. Removal records precede reuse of their physical storage. Generations
and checksums still validate references if an interrupted write leaves stale metadata.

On startup:

1. Verify the header and fingerprint before interpreting stored objects.
2. Replay the last valid checkpoint and complete journal records, ignoring an incomplete tail.
3. Validate referenced record identities/generations and dependency closure. Rebuild parents
   before children, then attach complete Disk snapshots.
4. Recompute runtime handles, reachability, per-tier accounting and eviction structures. Preserve
   bounded useful history such as hit counts, not pointers or old in-flight states.
5. Retain only paths supporting committed snapshots. Reclaim abandoned records in bounded work.
   Admit requests once the restored catalog is coherent.

Read headers/index metadata at startup; verify full payload checksums on demand rather than
preloading the disk cache into Host. A damaged or missing payload later invalidates that copy
and affected disk closures before reuse. Other intact copies and shallower snapshots remain
eligible. A corrupt catalog must not publish a partial tree with unchecked ancestry.

Normal shutdown stops new submissions, settles accepted jobs, commits complete closures and
flushes catalog updates. An interrupted shutdown may lose pending cache work. There is no
power-loss durability promise: missing, torn or inconsistent records become misses, never
unchecked model inputs. The existing `DiskKVStore::flush_index` is not a durability barrier.

Only committed Disk closures survive restart. Host-only hot content is not automatically exported
at shutdown; this is not a replacement full-Host snapshot under another name.

## 7. Disk pressure and failure handling

The index chooses reclamation; the physical store cannot evict an arbitrary referenced page.
Reserve space before accepting writes and fail a new cache insertion when no safe space exists.
A Hybrid-specific no-automatic-eviction storage operation may extend the Core primitive without
changing Legacy LRU behavior.

Choose low-value, unpinned retention to release, update the affected disk closure records, and
then reclaim payloads that are no longer required. Mixed-tier runtime snapshots also count:
a Host image can still depend on a Disk-only ancestor. Removing a disk snapshot does not by
itself make every record on its path reclaimable. Shared ancestors, active dependencies and
in-flight read/write claims protect the records they still need.

| Failure | Required result |
|---|---|
| Queue full, slow writes or budget exhaustion | Stop submissions, skip retention or reclaim lower-value cache; keep serving active work |
| Missing record, checksum failure or read error | Invalidate the bad copy, release completed job resources and reselect a valid source before activation |
| Cancelled preparation | Detach request interest; retain accepted I/O buffers and copy destinations until safe completion |
| Stale completion after deletion/reuse | Reject it by runtime handle and record generation; release only resources owned by that job. A closure whose block write failed or whose anchor record was evicted never commits; the index's removal callbacks reach the catalog, which journals the removal so a restart never resurrects it. A failed closure's own image record is reclaimed only after that write returned, so a queued write can never recreate a deleted payload. |
| CUDA transfer failure | Preserve existing Engine-wide CUDA failure semantics; do not disguise it as a disk miss |
| Invalid path, incompatible store or insufficient startup geometry | Fail startup with the incompatible prerequisite identified |

Disk-disabled operation must preserve the existing two-tier behavior. Failure isolation must not
suppress invariant violations or convert arbitrary Program exceptions into cache misses.

## 8. Configuration and rebase surface

The Hybrid configuration reuses the existing spellings `--disk-kv-path` and `--disk-kv-gib` for
an opt-in path and positive bounded capacity (default 64 GiB). Hybrid opens and rebuilds its own
catalog whenever this tier is enabled. The Legacy `--disk-kv-restore` and
`--disk-kv-directstorage` switches remain Legacy-only; Hybrid adds no DirectStorage or
direct-to-GPU backend and rejects switches it ignores.

Validate the same combinations at the CLI and Engine boundary:

- Disk requires enabled Hybrid reuse and a nonzero Host budget large enough for staging and the
  remaining Host cache geometry.
- Disk and `hybrid.persistent_file` (`--prefix-cache-file`) are mutually exclusive. Reject both;
  neither takes precedence and neither silently exports/imports the other format.
- Disk path and capacity must resolve together. Existing Hybrid single-device restrictions and
  rejection of unrelated Legacy policies remain in force.

Watermarks, worker limits and batch sizes should be resolved from resource geometry and
qualification, with effective values in the startup ledger. Add user controls only for a
measured deployment requirement, not an unqualified knob for every internal limit.

The implementation patch is concentrated in:

- `runtime/prefix_cache`: per-tier copy predicates, retention/accounting and persistent catalog.
- `models/qwen3_5/program/prefix`: disk jobs, staging and prepared-source publication.
- `hybrid_program.cpp` and its quote/transaction data: ready-source validation and final binding.
- The existing head-prefetch/progress boundary: completion wakeup and bounded preparation before
  admission, without a second scheduler.
- Option normalization/validation, explicit CMake source lists and the affected tests/docs.

No loader weight changes, model mathematics or new family hierarchy belong to this feature. The
two-tier Hybrid reference remains authoritative where `--disk-kv-path` is absent.

## 9. Performance contract

The goal is to reduce recomputation after Host eviction while protecting hot requests and active
decode. Separate streams and CPU workers provide overlap, not isolation from shared bandwidth.
No absolute TTFT or zero-overhead claim follows from this design.

### TTFT

Protect Device/Host hits from disk work, catalog locks and checkpoints. Count queue wait and
unhidden preparation separately so overlap does not get counted twice. Select Disk only when
its remaining preparation plus prefill beats the best alternative with a safety margin.
Cold-start and OS-page-cache-warm reads need distinct coefficients and results.

FIFO-head prefetch may hide preparation under existing queue wait. It must not reserve a whole
cold path against active growth, synchronously settle an unfinished disk job, or wait forever
because a predicted cache hit exists.

### Prefill

Preserve tap density, chunk boundaries and arithmetic. Offload batches existing Host bytes;
it does not add exact taps, forward passes, conversion or compression. Bound worker-side
completion adoption and avoid catalog-wide scans at each commit/admission.

Report actually computed prefill tokens per second separately from effective input throughput
that includes reused tokens. Include time lost to pressure and CPU bookkeeping, not just GPU
kernel duration.

### Decode

Preserve the current prefill/decode service schedule. Give demanded reads precedence over
background writes inside the cache I/O subsystem without changing request FIFO or product QoS.
Use bounded queues and H2D batches; qualify byte/time limits while other lanes decode. Idle
execution can use a larger transfer budget than active decode.

H2D competes for PCIe and HBM bandwidth even on copy engines. Host-to-Disk writes compete for CPU,
RAM and storage bandwidth. Include expert host/disk streaming on the same devices in qualification
when that placement is used. Protect decode tail latency by slowing preparation or discarding
retention, not by extending GPU wait chains into SSD latency.

### Initial acceptance targets

These qualification targets are not measured results or previously accepted hardware-independent
SLAs. Fix the workload and thresholds before collecting implementation results; disclose and
justify any later change.

| Workload | Baseline | Initial target |
|---|---|---|
| Device/Host hits | Current Hybrid at matched effective hot capacity | TTFT p50 and p95 regress by no more than 2% |
| Root/cold prefill with background offload | Same model, executed tokens, chunk and concurrency without disk jobs | Actual prefill throughput regresses by no more than 2% |
| Decode with background offload | Same active sequences/backend without offload | Accepted-token throughput regresses by no more than 2%; streaming-gap p99 by no more than 5% |
| Decode with demanded cold preparation | Matched decode workload without preparation | Meet the same decode targets while reporting the cold request's TTFT and worst stalls separately |
| Disk-restored requests | Current Hybrid recomputation at the same Host budget | Demonstrate TTFT benefit across the selected workload distribution, including tails and exceptions |

Run both fixed-total-memory and matched-effective-hot-capacity comparisons. The first exposes
staging's reduction of hot retention; the second isolates mechanism overhead. Report cold fills'
eviction of other hot prefixes rather than attributing all hit-rate changes to workload alone.

A comparison records model/artifact, GPU and link, SSD/filesystem, toolchain, KV/backend,
concurrency, prompt/output lengths, shared-prefix distribution, cache state, budgets and offered
load. Include repeated comparable runs, sample counts, p50/p95/p99 where applicable, worst
changes, and every failed cell. Do not hide pauses behind average throughput or dismiss a small
regression as noise without evidence.

## 10. Verification and delivery

Implement in dependency order: persistent record/catalog correctness, bounded Host demotion,
pre-admission cold preparation, then public Engine qualification. These are integration steps,
not separately shippable incomplete substitutes for the four required behaviors.

### CPU behavior

Extend existing behavioral coverage for:

- Exact token/extra-key identity under forced lookup collisions; stable IDs across restart.
- Disk-only and mixed-tier validity, shared ancestors, nonaligned tails and last-copy loss.
- Tier-local reclaimable bytes, bounded metadata, and no automatic eviction of referenced records.
- Provisional blocks, complete snapshot commit, removal, interrupted checkpoints/journals,
  missing payloads and checksum failures.
- The demotion commit state machine: a closure commits only after every awaited block record and
  its image landed; a failed block write, a failed catalog commit or a record the index removed
  closes the fill and reclaims the image record, and a failed closure keeps its Host copy.
- Host-copy reclamation at commit: `release_redundant_host_copies` frees a chain node's slab only
  once no live snapshot below it lacks a committed closure, and never a pinned node's.
- The disk-committed tail as a device-fill source after device pressure shed its Device copy, and
  the refusal of a tail with no Host slab and no committed closure.
- Demotion candidate selection (unpinned, Host-resident, no committed closure, lowest GDSF first)
  and the source-ranking cost model, including a disabled or slow disk rate.
- The staging budget split: `plan_hybrid_disk_staging` caps the block batch, never shrinks the
  slab pool below its own minimum, and reports the exact rejection figure.
- Queue saturation, stale completions, cancellation and source lifetime through accepted I/O.
- CLI and Engine option validation, including persistence-file mutual exclusion.

The existing `ninfer_prefix_cache_index_test`, `ninfer_disk_kv_store_test` and
`ninfer_disk_kv_bridge_test` cover reusable components, not this new composition. Native I/O
queue tests may skip in restricted environments; report that gap rather than treating a skip as
asynchronous-read qualification.

### Real execution and numerical semantics

Exercise the public Engine with a real artifact and force placement so a supposed Disk hit cannot
silently come from Device, Host or the old persistence file. Verify a prefix larger than the Host
pool, restart reconstruction, multiple shared branches, pressure during active growth and
cancellation while disk reads or CUDA copies remain outstanding. Include an idle Engine so job
completion itself must wake admission.

Compare restored KV and StateImage bytes exactly with their captured representation. Compare
Disk and hot restores of the same snapshot under the same continuation path. Root-vs-resumed
model checks retain the Hybrid specification's chunk-decomposition-aware numerical criteria;
byte-exact storage does not imply identical root-prefill arithmetic. Cover the promised KV and
speculative backend combinations, including MTP commitment lag and DFlash local state, and
preserve Vision's exact identity and forbidden resume spans.

### Performance and resource evidence

Measure hot hits, root prefill, offload-only, restore-only and simultaneous read/write pressure.
Include one and multiple lanes, long prefixes with short suffixes, sequential and scattered
records, storage-cold and OS-page-cache-warm reads, disk saturation and the relevant expert
placements. Track source-specific bytes, preparation/queue time, actual prefill work, accepted
decode tokens, streaming gaps, staging/leased Host peaks and cache drops. Extend existing logs
and benchmarks rather than creating a separate observability project.

CPU tests establish metadata, ownership and I/O behavior. GPU runs establish restore correctness,
TTFT and interference. Until those runs satisfy the selected acceptance conditions, report this
feature as unqualified and keep it opt-in. No GPU performance result is recorded by this design.
