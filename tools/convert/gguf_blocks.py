"""A Qwen3.5/3.8 dense GGUF kept in its own ggml block formats (GSQ-RCO mixed-precision releases).

Every text projection, the token table, the output head and the MTP head of an ``-mtp`` build are
stored as the exporter's own blocks, one ggml type per tensor, so no weight is re-quantised: a row
of the artifact is byte for byte a row of the GGUF. llama.cpp's Qwen3.5 exporter conventions are
undone where they touch rows or small tensors: value heads return to the grouped order,
zero-centred norms drop their stored ``1 + w`` and ``ssm_a`` becomes ``A_log`` again.

``ssm_out`` is the exception. Its tiled value heads are its input columns, which a row copy cannot
move without re-quantising, so its Use carries an ``input_columns`` auxiliary: column ``c`` of the
stored matrix multiplies element ``input_columns[c]`` of the grouped activation.

The GDN controls ``ssm_alpha`` and ``ssm_beta`` are BF16 operands of the Engine. Exporters store
them as BF16 words or as Q8_0 blocks (Underdog-Saluki and similar releases), whose values are
rounded to BF16: the one place a stored value changes.

Vision comes from the release's ``mmproj`` GGUF (``--source vision=mmproj.gguf``) with the official
Vision formats, DFlash2 from ``--source dflash2`` and the frontend resources from ``--model``, which
needs only the base checkpoint's config and tokenizer files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from tools.artifact.formats import GGUF_FORMATS_BY_TYPE

from .methods import AuxiliaryValue, cast_direct, grouped_absmax, import_encoded
from .official_recipes import Q4, Q5, Q6, Q8
from .sources.gguf import GGUFFile
from .sources.logical import EncodedRows, LogicalSource, array_source
from .ternary import (
    ATTENTION_HEAD_DIM,
    GDN_HEAD_DIM,
    GDN_KEY_HEADS,
    GDN_TAPS,
    GDN_VALUE_HEADS,
    HIDDEN,
    INTERMEDIATE,
    LAYERS,
    VOCABULARY,
    RowMap,
    _direct,
    _flat,
    _norm,
    rows,
)

MTP_BLOCK = LAYERS
# The storage the GDN controls may have (gdn_control_source).
GDN_CONTROL_TYPES = ("BF16", "Q8_0")


@dataclass(frozen=True, slots=True)
class Geometry:
    """One dense Qwen3.5/3.8 GGUF geometry: every constant the importer paths share.

    ``gdn_value_heads // gdn_key_heads`` decides the exporter's value-head tiling: one value
    head per key head needs no regrouping, a ratio above 1 stores the heads tiled and the
    importer returns them to the grouped order.
    """

    hidden: int
    intermediate: int
    vocabulary: int
    layers: int
    attention_heads: int
    attention_head_dim: int
    attention_kv_heads: int
    gdn_key_heads: int
    gdn_value_heads: int
    gdn_head_dim: int
    gdn_taps: int
    full_attention_interval: int
    # A tied text tower stores no separate output head; the artifact shares the embedding.
    tied: bool

    @property
    def attention_query_rows(self) -> int:
        return self.attention_heads * self.attention_head_dim

    @property
    def attention_kv_rows(self) -> int:
        return self.attention_kv_heads * self.attention_head_dim

    @property
    def gdn_key_dim(self) -> int:
        return self.gdn_key_heads * self.gdn_head_dim

    @property
    def gdn_value_dim(self) -> int:
        return self.gdn_value_heads * self.gdn_head_dim

    @property
    def gdn_channels(self) -> int:
        return 2 * self.gdn_key_dim + self.gdn_value_dim

    @property
    def per_key(self) -> int:
        return self.gdn_value_heads // self.gdn_key_heads

    @property
    def mtp_block(self) -> int:
        return self.layers

    def full_attention(self, layer: int) -> bool:
        return layer % self.full_attention_interval == self.full_attention_interval - 1

    def _tiled_to_grouped(self) -> np.ndarray:
        """For grouped value head ``h = k * per_key + r``, the exporter's tiled position
        ``r * gdn_key_heads + k``."""

        heads = np.arange(self.gdn_value_heads)
        return (heads % self.per_key) * self.gdn_key_heads + heads // self.per_key

    def _untile(self, array: np.ndarray, head_dim: int) -> np.ndarray:
        """Reorder axis 0 from the exporter's tiled value-head order back to grouped."""

        if self.per_key == 1:
            return array
        if array.shape[0] != self.gdn_value_heads * head_dim:
            raise ValueError(
                f"axis 0 has {array.shape[0]} entries, "
                f"expected {self.gdn_value_heads * head_dim}"
            )
        view = array.reshape(self.gdn_value_heads, head_dim, *array.shape[1:])
        return np.ascontiguousarray(view[self._tiled_to_grouped()].reshape(array.shape))

    def _untiled_rows(self, base: int) -> RowMap:
        """Grouped value-head rows of a tiled family starting at source row ``base``."""

        if self.per_key == 1:
            return rows(base)
        permutation, dim = self._tiled_to_grouped(), self.gdn_head_dim

        def select(begin: int, end: int) -> np.ndarray:
            r = np.arange(begin, end, dtype=np.int64)
            return base + permutation[r // dim] * dim + r % dim

        return select

    def _attention_rows(self, gate: bool) -> RowMap:
        """The query (or output gate) rows of the head-interleaved ``attn_q``."""

        d = self.attention_head_dim

        def select(begin: int, end: int) -> np.ndarray:
            r = np.arange(begin, end, dtype=np.int64)
            return (r // d) * 2 * d + (d if gate else 0) + r % d

        return select

    def expected_header(self) -> dict[str, object]:
        return {
            "general.architecture": "qwen35",
            "qwen35.embedding_length": self.hidden,
            "qwen35.feed_forward_length": self.intermediate,
            "qwen35.attention.head_count": self.attention_heads,
            "qwen35.attention.head_count_kv": self.attention_kv_heads,
            "qwen35.attention.key_length": self.attention_head_dim,
            "qwen35.attention.value_length": self.attention_head_dim,
            "qwen35.ssm.conv_kernel": self.gdn_taps,
            "qwen35.ssm.state_size": self.gdn_head_dim,
            "qwen35.ssm.group_count": self.gdn_key_heads,
            "qwen35.ssm.time_step_rank": self.gdn_value_heads,
            "qwen35.ssm.inner_size": self.gdn_value_dim,
            "qwen35.full_attention_interval": self.full_attention_interval,
        }

    def _attention_tensors(self, prefix: str) -> dict[str, tuple[tuple[int, ...], str]]:
        return {
            prefix + "attn_q.weight": ((2 * self.attention_query_rows, self.hidden), "blocks"),
            prefix + "attn_k.weight": ((self.attention_kv_rows, self.hidden), "blocks"),
            prefix + "attn_v.weight": ((self.attention_kv_rows, self.hidden), "blocks"),
            prefix + "attn_output.weight": (
                (self.hidden, self.attention_query_rows),
                "blocks",
            ),
            prefix + "attn_q_norm.weight": ((self.attention_head_dim,), "F32"),
            prefix + "attn_k_norm.weight": ((self.attention_head_dim,), "F32"),
        }

    def _layer_tensors(self, prefix: str) -> dict[str, tuple[tuple[int, ...], str]]:
        return {
            prefix + "attn_norm.weight": ((self.hidden,), "F32"),
            prefix + "post_attention_norm.weight": ((self.hidden,), "F32"),
            prefix + "ffn_gate.weight": ((self.intermediate, self.hidden), "blocks"),
            prefix + "ffn_up.weight": ((self.intermediate, self.hidden), "blocks"),
            prefix + "ffn_down.weight": ((self.hidden, self.intermediate), "blocks"),
        }

    def expected_tensors(
        self, mtp: bool
    ) -> dict[str, tuple[tuple[int, ...], str | tuple[str, ...]]]:
        """Row-major shape and kind of every tensor: "blocks" (any stored ggml block type) or
        the ggml type names it may have."""

        out = {
            "token_embd.weight": ((self.vocabulary, self.hidden), "blocks"),
            "output_norm.weight": ((self.hidden,), "F32"),
        }
        if not self.tied:
            out["output.weight"] = ((self.vocabulary, self.hidden), "blocks")
        for layer in range(self.layers):
            p = f"blk.{layer}."
            out |= self._layer_tensors(p)
            if self.full_attention(layer):
                out |= self._attention_tensors(p)
                continue
            out |= {
                p + "attn_qkv.weight": ((self.gdn_channels, self.hidden), "blocks"),
                p + "attn_gate.weight": ((self.gdn_value_dim, self.hidden), "blocks"),
                p + "ssm_out.weight": ((self.hidden, self.gdn_value_dim), "blocks"),
                p
                + "ssm_alpha.weight": (
                    (self.gdn_value_heads, self.hidden),
                    GDN_CONTROL_TYPES,
                ),
                p
                + "ssm_beta.weight": (
                    (self.gdn_value_heads, self.hidden),
                    GDN_CONTROL_TYPES,
                ),
                p + "ssm_a": ((self.gdn_value_heads,), "F32"),
                p + "ssm_dt.bias": ((self.gdn_value_heads,), "F32"),
                p + "ssm_conv1d.weight": ((self.gdn_channels, self.gdn_taps), "F32"),
                p + "ssm_norm.weight": ((self.gdn_head_dim,), "F32"),
            }
        if mtp:
            p = f"blk.{self.mtp_block}."
            out |= self._layer_tensors(p) | self._attention_tensors(p)
            out |= {
                p + "nextn.eh_proj.weight": ((self.hidden, 2 * self.hidden), "blocks"),
                p + "nextn.enorm.weight": ((self.hidden,), "F32"),
                p + "nextn.hnorm.weight": ((self.hidden,), "F32"),
                p + "nextn.shared_head_norm.weight": ((self.hidden,), "F32"),
            }
        return out

    def has_mtp(self, gguf: GGUFFile) -> bool:
        return f"blk.{self.mtp_block}.nextn.eh_proj.weight" in gguf.tensors

    def validate(self, gguf: GGUFFile) -> None:
        """Refuse any GGUF that is not this geometry's dense Qwen3.5/3.8 text tower of
        stored block types."""

        for key, expected in self.expected_header().items():
            if gguf.kv.get(key) != expected:
                raise ValueError(
                    f"{gguf.path}: {key} = {gguf.kv.get(key)!r}, expected {expected!r}"
                )
        mtp = self.has_mtp(gguf)
        blocks = gguf.kv.get("qwen35.block_count")
        if blocks != self.layers + (1 if mtp else 0):
            raise ValueError(
                f"{gguf.path}: {blocks} blocks, expected {self.layers} (+1 for MTP)"
            )
        expected = self.expected_tensors(mtp)
        if set(gguf.tensors) != set(expected):
            missing = sorted(set(expected) - set(gguf.tensors))[:5]
            extra = sorted(set(gguf.tensors) - set(expected))[:5]
            raise ValueError(
                f"{gguf.path}: tensor set mismatch (missing {missing}, extra {extra})"
            )
        for name, (shape, kind) in expected.items():
            info = gguf.tensors[name]
            if kind == "blocks":
                accepted = info.type_id in GGUF_FORMATS_BY_TYPE
            else:
                accepted = info.type_name in ((kind,) if isinstance(kind, str) else kind)
            if info.shape != shape or not accepted:
                raise ValueError(f"{gguf.path}: {name} is {info.type_name} {info.shape}")
        end = max(info.offset + info.nbytes for info in gguf.tensors.values())
        if gguf.data_bytes_available < end:
            raise ValueError(f"{gguf.path}: the data section is truncated")

    def gdn_control_source(self, gguf: GGUFFile, tensor: str) -> LogicalSource:
        """A GDN A/B control as its BF16 operand, its value heads in the grouped order: BF16
        words as stored, Q8_0 values rounded to BF16."""

        if gguf.info(tensor).type_id == GGML_Q8_0:
            values = self._untile(_dequantize(gguf, tensor, 0, self.gdn_value_heads).numpy(), 1)
            return array_source(
                torch.from_numpy(np.ascontiguousarray(values)).to(torch.bfloat16), tensor
            )
        words = self._untile(gguf.read_bf16_words(tensor), 1)
        return array_source(
            torch.from_numpy(np.ascontiguousarray(words.view(np.int16))).view(torch.bfloat16),
            tensor,
        )

    def tiled_input_columns(self) -> np.ndarray:
        """For each stored ``ssm_out`` column (tiled value heads), its grouped activation
        element."""

        columns = np.arange(self.gdn_value_dim, dtype=np.int64)
        tiled_head, lane = columns // self.gdn_head_dim, columns % self.gdn_head_dim
        grouped_head = (
            tiled_head % self.gdn_key_heads
        ) * self.per_key + tiled_head // self.gdn_key_heads
        return (grouped_head * self.gdn_head_dim + lane).astype(np.int32)


# The Qwen3.8-27B Dense release the original recipe imports. The module-level entry points stay
# bound to this geometry so existing callers see unchanged behavior.
GEOMETRY_27B = Geometry(
    hidden=HIDDEN,
    intermediate=INTERMEDIATE,
    vocabulary=VOCABULARY,
    layers=LAYERS,
    attention_heads=24,
    attention_head_dim=ATTENTION_HEAD_DIM,
    attention_kv_heads=4,
    gdn_key_heads=GDN_KEY_HEADS,
    gdn_value_heads=GDN_VALUE_HEADS,
    gdn_head_dim=GDN_HEAD_DIM,
    gdn_taps=GDN_TAPS,
    full_attention_interval=4,
    tied=False,
)

# The unsloth Qwen3.5-2B GGUF release: 8 attention heads (2 KV), GDN 16 key + 16 value heads
# (one value head per key head, so the exporter's tiling is the identity), a tied output head
# and no MTP tensors.
GEOMETRY_2B = Geometry(
    hidden=2048,
    intermediate=6144,
    vocabulary=248320,
    layers=24,
    attention_heads=8,
    attention_head_dim=256,
    attention_kv_heads=2,
    gdn_key_heads=16,
    gdn_value_heads=16,
    gdn_head_dim=128,
    gdn_taps=4,
    full_attention_interval=4,
    tied=True,
)

EXPECTED_HEADER = GEOMETRY_27B.expected_header()


def expected_tensors(mtp: bool) -> dict[str, tuple[tuple[int, ...], str | tuple[str, ...]]]:
    """Row-major shape and kind of every 27B tensor (the module's original geometry)."""

    return GEOMETRY_27B.expected_tensors(mtp)


def has_mtp(gguf: GGUFFile) -> bool:
    return GEOMETRY_27B.has_mtp(gguf)


def validate(gguf: GGUFFile) -> None:
    """Refuse any GGUF that is not a dense Qwen3.5/3.8-27B text tower of stored block types."""

    GEOMETRY_27B.validate(gguf)


def gdn_control_source(gguf: GGUFFile, tensor: str) -> LogicalSource:
    return GEOMETRY_27B.gdn_control_source(gguf, tensor)


def tiled_input_columns() -> np.ndarray:
    """For each stored 27B ``ssm_out`` column, its grouped activation element."""

    return GEOMETRY_27B.tiled_input_columns()


def block_format(gguf: GGUFFile, tensor: str) -> str:
    return GGUF_FORMATS_BY_TYPE[gguf.info(tensor).type_id].name


GGML_Q8_0 = 8
GGML_Q2_0 = 42


def dequantize_q2_0(blocks: np.ndarray) -> np.ndarray:
    """ggml Q2_0 rows: 64 values per 18-byte block, a binary16 d then 16 code bytes; value j is
    d * (c - 1) with c the bit pair 2 (j % 4) of byte j / 4. gguf-py predates the type."""

    rows = blocks.shape[0]
    b = np.ascontiguousarray(blocks).reshape(rows, -1, 18)
    d = b[..., :2].copy().view("<f2").astype(np.float32)
    q = b[..., 2:]
    codes = np.stack([(q >> (2 * i)) & 3 for i in range(4)], axis=-1).reshape(rows, b.shape[1], 64)
    return ((codes.astype(np.float32) - 1.0) * d).reshape(rows, -1)


def dequantize_q8_0(blocks: np.ndarray) -> np.ndarray:
    """ggml Q8_0 rows: 32 values per 34-byte block, a binary16 d then 32 signed code bytes q;
    value j is d * q[j], exact in FP32."""

    rows = blocks.shape[0]
    b = np.ascontiguousarray(blocks).reshape(rows, -1, 34)
    d = b[..., :2].copy().view("<f2").astype(np.float32)
    q = b[..., 2:].copy().view(np.int8).astype(np.float32)
    return (q * d).reshape(rows, -1)


def quantize_q8_0(values: np.ndarray) -> np.ndarray:
    """ggml quantize_row_q8_0_ref over rows of whole blocks, in FP32: per 32 values a binary16
    d = amax / 127 and codes round(x / d), halves away from zero. Returns [rows, row bytes]."""

    rows = values.shape[0]
    blocks = np.ascontiguousarray(values, dtype=np.float32).reshape(rows, -1, 32)
    d = np.abs(blocks).max(axis=-1) / np.float32(127.0)
    inverse = np.divide(np.float32(1.0), d, out=np.zeros_like(d), where=d != 0)
    scaled = blocks * inverse[..., None]
    codes = (np.sign(scaled) * np.floor(np.abs(scaled) + np.float32(0.5))).astype(np.int8)
    out = np.empty((rows, blocks.shape[1], 34), dtype=np.uint8)
    out[..., :2] = d.astype("<f2")[..., None].view(np.uint8)
    out[..., 2:] = codes.view(np.uint8)
    return out.reshape(rows, -1)


def _dequantize(gguf: GGUFFile, tensor: str, first: int, last: int) -> torch.Tensor:
    blocks = gguf.read_blocks(tensor, first, last)
    local = {GGML_Q2_0: dequantize_q2_0, GGML_Q8_0: dequantize_q8_0}.get(
        gguf.info(tensor).type_id
    )
    if local is not None:
        values = local(blocks)
        return torch.from_numpy(np.ascontiguousarray(values)).reshape(last - first, -1)
    try:
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
    except ImportError as error:
        raise ValueError(
            "reading the values of a GGUF block matrix needs the `gguf` package; "
            "storing its blocks does not"
        ) from error
    kind = GGMLQuantizationType(gguf.info(tensor).type_id)
    values = dequantize(np.ascontiguousarray(blocks), kind)
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)).reshape(
        last - first, -1
    )


def block_source(
    gguf: GGUFFile, tensor: str, shape: tuple[int, int], select: RowMap
) -> LogicalSource:
    """Rows of a stored block matrix as encoded ggml rows, and their values on request."""

    format = block_format(gguf, tensor)
    empty = torch.empty(0, dtype=torch.float16)

    def encoded(begin: int, end: int) -> EncodedRows:
        index = select(begin, end)
        low, high = int(index.min()), int(index.max()) + 1
        blocks = gguf.read_blocks(tensor, low, high)
        return EncodedRows(
            format, torch.from_numpy(np.ascontiguousarray(blocks[index - low])), empty
        )

    def values(first: int, last: int) -> torch.Tensor:
        index = select(first, last)
        low, high = int(index.min()), int(index.max()) + 1
        return _dequantize(gguf, tensor, low, high)[torch.from_numpy(index - low)]

    return LogicalSource(
        shape, f"{tensor}[{format}]{list(shape)}", _flat(values, shape[1]), encoded
    )


def q2_native_source(
    gguf: GGUFFile, tensor: str, shape: tuple[int, int], select: RowMap
) -> LogicalSource:
    """Import GSQ Q2_0 as native offset codes and the exact signed binary16 scale words.

    The row map selects rows within a tensor or flattened expert bank. The physical K padding
    holds logical integer zero (stored code 1); wholly padding groups have positive-zero scales.
    No floating-point dequantization or quantizer participates in the encoded import.
    """
    if gguf.info(tensor).type_id != GGML_Q2_0:
        raise ValueError(f"{tensor}: native Q2 import requires stored Q2_0 blocks")
    if len(shape) != 2 or shape[1] != gguf.info(tensor).shape[-1] or shape[1] % 64:
        raise ValueError(f"{tensor}: native Q2 import requires complete 64-value blocks")
    raw_source = block_source(gguf, tensor, shape, select)
    k = shape[1]
    groups, padded_groups = k // 64, ((k + 127) // 128) * 2

    def encoded(begin: int, end: int) -> EncodedRows:
        index = select(begin, end)
        low, high = int(index.min()), int(index.max()) + 1
        blocks = gguf.read_blocks(tensor, low, high)[index - low].reshape(end - begin, groups, 18)
        scale_words = np.zeros((end - begin, padded_groups), dtype="<u2")
        scale_words[:, :groups] = np.ascontiguousarray(blocks[..., :2]).view("<u2").squeeze(-1)
        scales = scale_words.view("<f2")
        if not np.isfinite(scales).all():
            raise ValueError(f"{tensor}: Q2 scales must be finite binary16 words")
        codes = np.full((end - begin, padded_groups, 16), 0x55, dtype=np.uint8)
        codes[:, :groups] = blocks[..., 2:]
        return EncodedRows("q2_g64_fp16", torch.from_numpy(codes.reshape(end - begin, -1)),
                           torch.from_numpy(scales), packed=True)

    return LogicalSource(shape, f"{tensor}[q2_g64_fp16]{list(shape)}",
                         raw_source.read_values, encoded)


def q8_native_source(
    gguf: GGUFFile, tensor: str, shape: tuple[int, int], select: RowMap
) -> LogicalSource:
    """Preserve compatible Q8_0 integer codes and binary16 scales as native Q8 planes.

    Native Q8 excludes -128 and negative-zero scales, and requires zero codes for zero-scale
    groups. Refuse an incompatible source instead of silently changing its representation.
    """
    if gguf.info(tensor).type_id != 8:
        raise ValueError(f"{tensor}: native Q8 import requires stored Q8_0 blocks")
    if len(shape) != 2 or shape[1] != gguf.info(tensor).shape[-1] or shape[1] % 32:
        raise ValueError(f"{tensor}: native Q8 import requires complete 32-value blocks")
    raw_source = block_source(gguf, tensor, shape, select)
    groups, padded_groups = shape[1] // 32, ((shape[1] + 127) // 128) * 4

    def encoded(begin: int, end: int) -> EncodedRows:
        index = select(begin, end)
        low, high = int(index.min()), int(index.max()) + 1
        blocks = gguf.read_blocks(tensor, low, high)[index - low].reshape(end - begin, groups, 34)
        raw_words = np.ascontiguousarray(blocks[..., :2]).view("<u2").squeeze(-1)
        raw_scales = raw_words.view("<f2")
        raw_codes = blocks[..., 2:]
        if (not np.isfinite(raw_scales).all() or (raw_words == 0x8000).any() or
                (raw_codes == 0x80).any() or
                ((raw_scales == 0)[..., None] & (raw_codes != 0)).any()):
            raise ValueError(f"{tensor}: Q8_0 source is outside the native Q8 code/scale contract")
        scales = np.zeros((end - begin, padded_groups), dtype="<f2")
        scales[:, :groups] = raw_scales
        codes = np.zeros((end - begin, padded_groups, 32), dtype=np.uint8)
        codes[:, :groups] = raw_codes
        return EncodedRows("q8_g32_fp16", torch.from_numpy(codes.reshape(end - begin, -1)),
                           torch.from_numpy(scales), packed=True)

    return LogicalSource(shape, f"{tensor}[q8_g32_fp16]{list(shape)}",
                         raw_source.read_values, encoded)


def _attention(
    geometry: Geometry, gguf: GGUFFile, g: str, a: str
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    q_shape, kv_shape = (geometry.attention_query_rows, geometry.hidden), (
        geometry.attention_kv_rows,
        geometry.hidden,
    )
    encoded = {
        a + "query": block_source(gguf, g + "attn_q.weight", q_shape, geometry._attention_rows(False)),
        a + "gate": block_source(gguf, g + "attn_q.weight", q_shape, geometry._attention_rows(True)),
        a + "key": block_source(gguf, g + "attn_k.weight", kv_shape, rows()),
        a + "value": block_source(gguf, g + "attn_v.weight", kv_shape, rows()),
        a + "output": block_source(
            gguf, g + "attn_output.weight", (geometry.hidden, geometry.attention_query_rows), rows()
        ),
    }
    direct = {
        a + "query_norm": _norm(gguf, g + "attn_q_norm.weight", True),
        a + "key_norm": _norm(gguf, g + "attn_k_norm.weight", True),
    }
    return encoded, direct


def _layer(
    geometry: Geometry, gguf: GGUFFile, g: str, p: str
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    encoded = {
        p + "mlp/gate": block_source(
            gguf, g + "ffn_gate.weight", (geometry.intermediate, geometry.hidden), rows()
        ),
        p + "mlp/up": block_source(
            gguf, g + "ffn_up.weight", (geometry.intermediate, geometry.hidden), rows()
        ),
        p + "mlp/down": block_source(
            gguf, g + "ffn_down.weight", (geometry.hidden, geometry.intermediate), rows()
        ),
    }
    direct = {
        p + "input_norm": _norm(gguf, g + "attn_norm.weight", True),
        p + "post_attention_norm": _norm(gguf, g + "post_attention_norm.weight", True),
    }
    return encoded, direct


def _text_sources(
    geometry: Geometry, gguf: GGUFFile, *, mtp: bool
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    """Encoded block sources and direct sources of every GGUF-backed text (and MTP) parameter
    of one geometry."""

    encoded = {
        "text/token_embedding": block_source(
            gguf, "token_embd.weight", (geometry.vocabulary, geometry.hidden), rows()
        ),
    }
    if not geometry.tied:
        encoded["text/output_head"] = block_source(
            gguf, "output.weight", (geometry.vocabulary, geometry.hidden), rows()
        )
    direct = {"text/final_norm": _norm(gguf, "output_norm.weight", True)}
    for layer in range(geometry.layers):
        g, p = f"blk.{layer}.", f"text/layers/{layer}/"
        e, d = _layer(geometry, gguf, g, p)
        encoded |= e
        direct |= d
        if geometry.full_attention(layer):
            e, d = _attention(geometry, gguf, g, p + "attention/")
            encoded |= e
            direct |= d
            continue
        n = p + "gdn/"
        qkv = g + "attn_qkv.weight"
        encoded[n + "query"] = block_source(
            gguf, qkv, (geometry.gdn_key_dim, geometry.hidden), rows()
        )
        encoded[n + "key"] = block_source(
            gguf, qkv, (geometry.gdn_key_dim, geometry.hidden), rows(geometry.gdn_key_dim)
        )
        encoded[n + "value"] = block_source(
            gguf, qkv, (geometry.gdn_value_dim, geometry.hidden),
            geometry._untiled_rows(2 * geometry.gdn_key_dim),
        )
        encoded[n + "z"] = block_source(
            gguf, g + "attn_gate.weight", (geometry.gdn_value_dim, geometry.hidden),
            geometry._untiled_rows(0),
        )
        # Tiled input columns: the Use gathers the grouped activation (see tiled_input_columns).
        encoded[n + "output"] = block_source(
            gguf, g + "ssm_out.weight", (geometry.hidden, geometry.gdn_value_dim), rows()
        )
        for role, tensor in (
            ("a_projection", "ssm_alpha.weight"),
            ("b_projection", "ssm_beta.weight"),
        ):
            direct[n + role] = geometry.gdn_control_source(gguf, g + tensor)
        ssm_a = geometry._untile(gguf.read_direct(g + "ssm_a"), 1).astype(np.float64)
        if not np.all(ssm_a < 0):
            raise ValueError(f"{g}ssm_a must be strictly negative (-exp(A_log))")
        direct[n + "a_log"] = _direct(
            np.log(-ssm_a).astype(np.float32), torch.float32, g + "ssm_a"
        )
        direct[n + "dt_bias"] = _direct(
            geometry._untile(gguf.read_direct(g + "ssm_dt.bias"), 1),
            torch.float32,
            g + "ssm_dt.bias",
        )
        taps = gguf.read_direct(g + "ssm_conv1d.weight")
        channels = np.concatenate(
            [
                taps[: 2 * geometry.gdn_key_dim],
                geometry._untile(taps[2 * geometry.gdn_key_dim :], geometry.gdn_head_dim),
            ]
        )
        direct[n + "convolution"] = _direct(
            channels.T, torch.bfloat16, g + "ssm_conv1d.weight"
        )
        direct[n + "norm"] = _norm(gguf, g + "ssm_norm.weight", False)
    if mtp:
        g, p = f"blk.{geometry.mtp_block}.", "mtp/layers/0/"
        e, d = _layer(geometry, gguf, g, p)
        encoded |= e
        direct |= d
        e, d = _attention(geometry, gguf, g, p + "attention/")
        encoded |= e
        direct |= d
        encoded["mtp/input_projection"] = block_source(
            gguf, g + "nextn.eh_proj.weight", (geometry.hidden, 2 * geometry.hidden), rows()
        )
        for role, tensor in (
            ("embedding_norm", "nextn.enorm.weight"),
            ("hidden_norm", "nextn.hnorm.weight"),
            ("final_norm", "nextn.shared_head_norm.weight"),
        ):
            direct["mtp/" + role] = _norm(gguf, g + tensor, True)
    return encoded, direct


def text_sources(
    gguf: GGUFFile, *, mtp: bool
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    """Encoded block sources and direct sources of every GGUF-backed 27B text (and MTP)
    parameter."""

    return _text_sources(GEOMETRY_27B, gguf, mtp=mtp)


def _vision_format(name: str) -> str:
    if name == "vision/patch_embedding":
        return Q6
    if name.startswith("vision/merger/"):
        return Q8
    if name.endswith(("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")):
        return Q4
    return Q5


def vision_source(model, sources):
    """The `vision` source as HF names: an mmproj GGUF through the model's vision name map.

    Imported lazily so a conversion that does not read a GGUF needs no gguf-py.
    """

    from .sources.gguf_source import GGUFSource, HFAliasSource, qwen35_vision_name_map

    mmproj = sources["vision"]
    if not isinstance(mmproj, GGUFSource):
        return mmproj
    config = model.components["vision"]["config"]
    names, transforms, shapes, composites = qwen35_vision_name_map(
        config["depth"],
        hidden_size=config["hidden_size"],
        in_channels=config.get("in_channels", 3),
        temporal_patch_size=config["temporal_patch_size"],
        patch_size=config["patch_size"],
    )
    return HFAliasSource(
        mmproj, names, transforms=transforms, shapes=shapes, composites=composites
    )


def _companions(model, recipe, sources) -> None:
    """Vision from the mmproj GGUF in the official formats; DFlash2 as the official recipe."""

    vision = vision_source(model, sources) if "vision" in model.components else None
    for name, parameter in model.parameters.items():
        if name.startswith("vision/"):
            source = model.source(name, vision)
            if parameter.projection:
                recipe.assign(
                    name, format=_vision_format(name), method=grouped_absmax, source=source
                )
            else:
                recipe.assign(
                    name,
                    format=parameter.direct_format,
                    method=cast_direct,
                    source=source,
                )
        elif name.startswith(("dflash/", "dflash2/")) and parameter.projection:
            if name.endswith(
                (
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            recipe.assign(name, format=Q8, method=grouped_absmax)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _group_same_format(recipe, names: list[str], formats: dict[str, str]) -> None:
    if len({formats[name] for name in names}) == 1:
        recipe.group(names)


def _gguf_recipe(geometry: Geometry, model, recipe, sources) -> None:
    """Assign every GGUF-backed parameter of one dense geometry to its stored blocks."""

    gguf = sources["gguf"]
    if not isinstance(gguf, GGUFFile):
        raise ValueError("--source gguf must name the model's .gguf file")
    geometry.validate(gguf)
    mtp = "mtp" in model.components
    if mtp and not geometry.has_mtp(gguf):
        raise ValueError(
            f"{gguf.path}: the MTP component needs an -mtp GGUF "
            f"(blk.{geometry.mtp_block}.nextn.*)"
        )
    _companions(model, recipe, sources)
    encoded, direct = _text_sources(geometry, gguf, mtp=mtp)
    formats = {}
    for name, source in encoded.items():
        format = source.read_encoded(0, 1).format
        formats[name] = format
        recipe.assign(
            name, format=format, method=import_encoded, source=source, activation_policy="AllowA8"
        )
    for name, source in direct.items():
        recipe.assign(
            name,
            format=model.parameters[name].direct_format,
            method=cast_direct,
            source=source,
        )
    if geometry.tied:
        # The GGUF stores no output head; the artifact shares the token embedding's tensor.
        recipe.share("text/output_head", "text/token_embedding")
    prefixes = [f"text/layers/{layer}/" for layer in range(geometry.layers)]
    if mtp:
        prefixes.append("mtp/layers/0/")
    for p in prefixes:
        if p.startswith("text/") and not geometry.full_attention(int(p.split("/")[2])):
            n = p + "gdn/"
            qkv = [n + "query", n + "key", n + "value"]
            recipe.group(qkv + [n + "z"] if formats[n + "z"] == formats[n + "query"] else qkv)
            recipe.separate([n + "a_projection", n + "b_projection"])
        else:
            a = p + "attention/"
            _group_same_format(
                recipe, [a + "query", a + "key", a + "gate", a + "value"], formats
            )
        _group_same_format(recipe, [p + "mlp/gate", p + "mlp/up"], formats)
    columns = AuxiliaryValue(
        "int32",
        (geometry.gdn_value_dim,),
        geometry.tiled_input_columns().astype("<i4").tobytes(),
    )
    for layer in range(geometry.layers):
        if geometry.full_attention(layer):
            continue
        name = f"text/layers/{layer}/gdn/output"
        for input_name in model.parameters[name].inputs:
            recipe.use(name, input_name, auxiliaries={"input_columns": columns})


def qwen3_8_27b_gguf(model, recipe, sources):
    """A Qwen3.8-27B GGUF in its own block formats from `--source gguf=MODEL.gguf`."""

    config = model.config
    if (
        "num_experts" in config
        or config["hidden_size"] != HIDDEN
        or config["num_hidden_layers"] != LAYERS
    ):
        raise ValueError("the GGUF block recipe requires the Qwen3.8-27B Dense geometry")
    _gguf_recipe(GEOMETRY_27B, model, recipe, sources)


def qwen3_5_2b_gguf(model, recipe, sources):
    """The unsloth Qwen3.5-2B GGUF in its own block formats from `--source gguf=MODEL.gguf`."""

    config = model.config
    if (
        "num_experts" in config
        or config["hidden_size"] != GEOMETRY_2B.hidden
        or config["num_hidden_layers"] != GEOMETRY_2B.layers
    ):
        raise ValueError("this recipe requires the Qwen3.5-2B geometry")
    if not config["tie_word_embeddings"]:
        raise ValueError("the Qwen3.5-2B GGUF release has a tied output head")
    if "mtp" in model.components:
        raise ValueError("the Qwen3.5-2B GGUF release has no MTP head")
    _gguf_recipe(GEOMETRY_2B, model, recipe, sources)


RECIPES = {
    "qwen3_8_27b_gguf": qwen3_8_27b_gguf,
    "qwen3_5_2b_gguf": qwen3_5_2b_gguf,
}


__all__ = [
    "EXPECTED_HEADER",
    "GEOMETRY_2B",
    "GEOMETRY_27B",
    "Geometry",
    "RECIPES",
    "block_format",
    "block_source",
    "expected_tensors",
    "gdn_control_source",
    "has_mtp",
    "vision_source",
    "qwen3_5_2b_gguf",
    "qwen3_8_27b_gguf",
    "q2_native_source",
    "q8_native_source",
    "text_sources",
    "tiled_input_columns",
    "validate",
]
