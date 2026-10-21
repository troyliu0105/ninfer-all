"""Split a Qwen3.5/3.8 MTP (nextn) block out of a GGUF into standalone weights.

A Qwen3.5/3.8 GGUF that carries multi-token prediction appends one more transformer block after
the model's layers (`blk.<num_hidden_layers>.*`, the `nextn.*` tensors included) and counts it in
`qwen35.block_count`; PrismML's ternary releases and llama.cpp's `-mtp` exports both do. NInfer's
recipes take MTP from a companion source under the checkpoint's `mtp.*` names, so this tool writes
that source and a text-only GGUF of the same weights:

    python3 -m tools.convert.split_mtp --gguf MODEL.gguf \
        --text MODEL.text.gguf --mtp MODEL.mtp.safetensors

The text GGUF is the source's own header with the MTP block's directory entries removed and the
kept tensors re-laid byte for byte: every other tensor keeps its bytes and its stored type, and
only `qwen35.block_count` changes while `qwen35.nextn_predict_layers` goes. The safetensors holds
every MTP tensor as FP32 under the names the recipe reads (`mtp.fc.weight`,
`mtp.pre_fc_norm_embedding.weight`, `mtp.layers.0.*` ...), decoded from whatever block type stored
it. A ternary release quantizes its MTP matrices to two bits, so a drafter artifact built from
this source is a Q8 re-quantization of those values, exactly as one built from a separately
trained MTP head is.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
from typing import Any

import numpy as np
from safetensors.torch import save_file
import torch

from .sources.gguf import (
    BLOCK_FORMATS,
    GGUF_MAGIC,
    GGUF_VERSION,
    TERNARY_TYPES,
    TYPE_BF16,
    TYPE_F16,
    TYPE_F32,
    GGUFFile,
    align_up,
    encode_kv_entry,
)
from .sources.gguf_source import qwen35_mtp_name_map

DIRECT_TYPES = frozenset((TYPE_F32, TYPE_F16, TYPE_BF16))
MTP_LAYERS_KEY = "qwen35.nextn_predict_layers"
BLOCK_COUNT_KEY = "qwen35.block_count"


def mtp_block(gguf: GGUFFile, block: int | None) -> int:
    """The MTP block's index, defaulting to the last block of a file that predicts one layer."""

    if block is None:
        if gguf.kv.get(MTP_LAYERS_KEY) != 1:
            raise ValueError(
                f"{gguf.path}: {MTP_LAYERS_KEY} is {gguf.kv.get(MTP_LAYERS_KEY)!r}; "
                "name the MTP block with --block"
            )
        count = gguf.kv.get(BLOCK_COUNT_KEY)
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"{gguf.path}: {BLOCK_COUNT_KEY} is {count!r}")
        block = count - 1
    if not any(name.startswith(f"blk.{block}.") for name in gguf.tensors):
        raise ValueError(f"{gguf.path}: no tensors under blk.{block}.")
    return block


def mtp_names(block: int) -> dict[str, str]:
    """The `mtp.*` source name of every tensor of one MTP block, keyed by its GGUF name."""

    names, _, _ = qwen35_mtp_name_map(block, hf_prefix="mtp.")
    return {gguf_name: source for source, gguf_name in names.items()}


def tensor_values(gguf: GGUFFile, name: str) -> np.ndarray:
    """One tensor's values as FP32, whatever stored type holds it."""

    info = gguf.info(name)
    if info.type_id in TERNARY_TYPES:
        blocks = gguf.read_ternary(name)
        return (
            blocks.values.astype(np.float32) * blocks.scales.astype(np.float32)[:, :, None]
        ).reshape(info.shape)
    if info.type_id in DIRECT_TYPES:
        return gguf.read_direct(name)
    if info.type_id in BLOCK_FORMATS:
        return gguf.read_dequantized(name)
    raise ValueError(f"{name}: {info.type_name} is not a readable tensor type")


def write_text_gguf(
    gguf: GGUFFile,
    path: str | Path,
    *,
    dropped: frozenset[str],
    updates: dict[str, Any],
    remove: tuple[str, ...] = (),
) -> None:
    """Write the source's header without some tensors and keys, and its data section re-laid.

    Every kept entry is copied verbatim from the source's header, so its array element types, its
    string tables and its tensor byte ranges survive exactly; only the two counts, the updated
    values and each kept tensor's data offset are rewritten. Kept tensors keep the source's data
    order and the alignment the source declared.
    """

    unknown = sorted(set(updates) - set(gguf.kv))
    if unknown:
        raise ValueError(f"updates name keys the source does not hold: {unknown}")
    absent = sorted(set(remove) - set(gguf.kv))
    if absent:
        raise ValueError(f"removals name keys the source does not hold: {absent}")
    with gguf.path.open("rb") as handle:
        original = handle.read(gguf.header_bytes)
    if original[:4] != GGUF_MAGIC:
        raise ValueError(f"{gguf.path}: not a GGUF file")
    kept = sorted(
        (name for name in gguf.tensors if name not in dropped),
        key=lambda name: gguf.tensors[name].offset,
    )
    header = bytearray(
        GGUF_MAGIC
        + struct.pack("<IQQ", GGUF_VERSION, len(kept), len(gguf.kv) - len(remove))
    )
    for name in sorted(gguf.kv_spans, key=lambda key: gguf.kv_spans[key][0]):
        if name in remove:
            continue
        if name in updates:
            header += encode_kv_entry(name, updates[name])
        else:
            begin, end = gguf.kv_spans[name]
            header += original[begin:end]
    # Each directory entry ends with its tensor's data offset, which the re-laid section changes.
    offset_fields: dict[str, int] = {}
    for name in sorted(gguf.tensor_spans, key=lambda key: gguf.tensor_spans[key][0]):
        if name in dropped:
            continue
        begin, end, offset_at = gguf.tensor_spans[name]
        offset_fields[name] = len(header) + offset_at - begin
        header += original[begin:end]
    offsets: dict[str, int] = {}
    cursor = 0
    for name in kept:
        offsets[name] = cursor
        cursor = align_up(cursor + gguf.info(name).nbytes, gguf.alignment)
    for name, at in offset_fields.items():
        header[at : at + 8] = struct.pack("<Q", offsets[name])
    data_offset = align_up(len(header), gguf.alignment)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as handle:
        handle.write(header)
        handle.write(b"\x00" * (data_offset - len(header)))
        for name in kept:
            raw = memoryview(gguf.tensor_bytes(name))
            handle.write(raw)
            handle.write(
                b"\x00"
                * (align_up(offsets[name] + len(raw), gguf.alignment) - offsets[name] - len(raw))
            )


def split(
    source: str | Path,
    *,
    text: str | Path,
    mtp: str | Path,
    block: int | None = None,
) -> dict:
    """Write the text-only GGUF and the standalone MTP weights of one source file."""

    with GGUFFile(source) as gguf:
        block = mtp_block(gguf, block)
        names = mtp_names(block)
        block_tensors = {
            name for name in gguf.tensors if name.startswith(f"blk.{block}.")
        }
        unmapped = sorted(block_tensors - set(names))
        if unmapped:
            raise ValueError(
                f"{gguf.path}: blk.{block} holds tensors this splitter does not map: {unmapped}"
            )
        missing = sorted(set(names) - set(gguf.tensors))
        if missing:
            raise ValueError(f"{gguf.path}: the MTP block is missing {missing}")
        rotated = set(gguf.kv.get("prism.hadamard.weight_names", ())) | set(
            gguf.kv.get("prism.hadamard.inverse_weight_names", ())
        )
        shared = sorted(set(names) & rotated)
        if shared:
            raise ValueError(
                f"{gguf.path}: the MTP block is Hadamard-rotated ({shared[:3]}); its rotations "
                "need the source's sign auxiliaries, which no mtp source carries"
            )
        tensors = {
            source_name: tensor_values(gguf, name) for name, source_name in names.items()
        }
        write_text_gguf(
            gguf,
            text,
            dropped=frozenset(names),
            updates={BLOCK_COUNT_KEY: block},
            remove=(MTP_LAYERS_KEY,),
        )
        Path(mtp).parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {name: torch.tensor(values) for name, values in tensors.items()},
            str(mtp),
            metadata={
                "format": "ninfer-mtp-subset-v1",
                "source": str(gguf.path),
                "source_bytes": str(gguf.file_bytes),
                "source_block": str(block),
                "tensors": str(len(tensors)),
            },
        )
        return {
            "block": block,
            "mtp_tensors": len(tensors),
            "text_tensors": len(gguf.tensors) - len(names),
            "mtp_bytes": sum(int(values.nbytes) for values in tensors.values()),
        }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True, help="the source GGUF to split")
    parser.add_argument("--text", type=Path, required=True, help="the text-only GGUF to write")
    parser.add_argument(
        "--mtp", type=Path, required=True, help="the standalone MTP weights to write"
    )
    parser.add_argument(
        "--block",
        type=int,
        help="the MTP block's index (default: the last block of a file predicting one layer)",
    )
    args = parser.parse_args(argv)
    report = split(args.gguf, text=args.text, mtp=args.mtp, block=args.block)
    print(
        f"wrote {args.text} ({report['text_tensors']} tensors) and {args.mtp} "
        f"(blk.{report['block']}, {report['mtp_tensors']} tensors, "
        f"{report['mtp_bytes'] / 2**20:.1f} MiB of FP32)",
        flush=True,
    )


if __name__ == "__main__":
    main()
