"""The generic dense GGUF geometry machinery and the Qwen3.5-2B recipe facts.

A fully scaled-down Geometry instance exercises every geometry-generic path (tied head,
identity value-head tiling, conv split, A_log transform) on a ~100 KB fixture. The real 2B
constants stay pinned by pure-data assertions and an optional check of the fetched release
header.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tools.convert import gguf_blocks
from tools.convert.sources.gguf import GGUFFile, TYPE_F32, write_gguf

# The unsloth Qwen3.5-2B UD-IQ2_XXS release, fetched as a 16 MiB header prefix during
# development. Absent on machines that never downloaded it; that check skips.
REAL_PREFIX = Path("/tmp/q35-2b-iq2xxs.part")

TYPE_Q8_0 = 8
TYPE_IQ2_XXS = 16

# A miniature dense geometry: every family present, one value head per key head (the 2B
# tiling property), a tied head, four blocks with one full-attention layer.
TINY = gguf_blocks.Geometry(
    hidden=256,
    intermediate=512,
    vocabulary=320,
    layers=4,
    attention_heads=2,
    attention_head_dim=128,
    attention_kv_heads=1,
    gdn_key_heads=2,
    gdn_value_heads=2,
    gdn_head_dim=128,
    gdn_taps=4,
    full_attention_interval=4,
    tied=True,
)


def _q8_0(values: np.ndarray) -> bytes:
    return gguf_blocks.quantize_q8_0(values.astype(np.float32)).tobytes()


def _zeros_iq2_xxs(rows: int, columns: int) -> bytes:
    """IQ2_XXS blocks are opaque stored bytes to the importer; zero payload is valid."""

    return bytes(rows * (columns // 256 * 66) if columns >= 256 else rows * columns * 2 // 3)


def _iq2_xxs_bytes(rows: int, columns: int) -> bytes:
    assert columns % 256 == 0, "fixture keeps whole IQ2_XXS blocks"
    return bytes(rows * columns // 256 * 66)


def _f32(count: int, value: float) -> bytes:
    return np.full(count, value, dtype="<f4").tobytes()


def _write_tiny_gguf(path: Path, g: gguf_blocks.Geometry) -> None:
    kv = g.expected_header() | {"qwen35.block_count": g.layers}
    tensors: list[tuple[str, list[int], int, bytes]] = [
        ("token_embd.weight", [g.vocabulary, g.hidden], TYPE_IQ2_XXS,
         _iq2_xxs_bytes(g.vocabulary, g.hidden)),
        ("output_norm.weight", [g.hidden], TYPE_F32, _f32(g.hidden, 1.0)),
    ]
    for layer in range(g.layers):
        p = f"blk.{layer}."
        tensors += [
            (p + "attn_norm.weight", [g.hidden], TYPE_F32, _f32(g.hidden, 1.0)),
            (p + "post_attention_norm.weight", [g.hidden], TYPE_F32, _f32(g.hidden, 1.0)),
            (p + "ffn_gate.weight", [g.intermediate, g.hidden], TYPE_IQ2_XXS,
             _iq2_xxs_bytes(g.intermediate, g.hidden)),
            (p + "ffn_up.weight", [g.intermediate, g.hidden], TYPE_IQ2_XXS,
             _iq2_xxs_bytes(g.intermediate, g.hidden)),
            (p + "ffn_down.weight", [g.hidden, g.intermediate], TYPE_Q8_0,
             _q8_0(np.zeros((g.hidden, g.intermediate)))),
        ]
        if g.full_attention(layer):
            tensors += [
                (p + "attn_q.weight", [2 * g.attention_query_rows, g.hidden], TYPE_IQ2_XXS,
                 _iq2_xxs_bytes(2 * g.attention_query_rows, g.hidden)),
                (p + "attn_k.weight", [g.attention_kv_rows, g.hidden], TYPE_IQ2_XXS,
                 _iq2_xxs_bytes(g.attention_kv_rows, g.hidden)),
                (p + "attn_v.weight", [g.attention_kv_rows, g.hidden], TYPE_Q8_0,
                 _q8_0(np.zeros((g.attention_kv_rows, g.hidden)))),
                (p + "attn_output.weight", [g.hidden, g.attention_query_rows], TYPE_IQ2_XXS,
                 _iq2_xxs_bytes(g.hidden, g.attention_query_rows)),
                (p + "attn_q_norm.weight", [g.attention_head_dim], TYPE_F32,
                 _f32(g.attention_head_dim, 1.0)),
                (p + "attn_k_norm.weight", [g.attention_head_dim], TYPE_F32,
                 _f32(g.attention_head_dim, 1.0)),
            ]
        else:
            tensors += [
                (p + "attn_qkv.weight", [g.gdn_channels, g.hidden], TYPE_IQ2_XXS,
                 _iq2_xxs_bytes(g.gdn_channels, g.hidden)),
                (p + "attn_gate.weight", [g.gdn_value_dim, g.hidden], TYPE_IQ2_XXS,
                 _iq2_xxs_bytes(g.gdn_value_dim, g.hidden)),
                (p + "ssm_out.weight", [g.hidden, g.gdn_value_dim], TYPE_Q8_0,
                 _q8_0(np.zeros((g.hidden, g.gdn_value_dim)))),
                (p + "ssm_alpha.weight", [g.gdn_value_heads, g.hidden], TYPE_Q8_0,
                 _q8_0(np.zeros((g.gdn_value_heads, g.hidden)))),
                (p + "ssm_beta.weight", [g.gdn_value_heads, g.hidden], TYPE_Q8_0,
                 _q8_0(np.zeros((g.gdn_value_heads, g.hidden)))),
                (p + "ssm_a", [g.gdn_value_heads], TYPE_F32, _f32(g.gdn_value_heads, -2.0)),
                (p + "ssm_dt.bias", [g.gdn_value_heads], TYPE_F32, _f32(g.gdn_value_heads, 0.1)),
                (p + "ssm_conv1d.weight", [g.gdn_channels, g.gdn_taps], TYPE_F32,
                 _f32(g.gdn_channels * g.gdn_taps, 0.01)),
                (p + "ssm_norm.weight", [g.gdn_head_dim], TYPE_F32, _f32(g.gdn_head_dim, 1.0)),
            ]
    write_gguf(path, kv, tensors)


@pytest.fixture(scope="session")
def tiny_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.getbasetemp() / "tiny.gguf"
    _write_tiny_gguf(path, TINY)
    return path


def test_2b_constants_match_the_release():
    g = gguf_blocks.GEOMETRY_2B
    assert g.expected_header() == {
        "general.architecture": "qwen35",
        "qwen35.embedding_length": 2048,
        "qwen35.feed_forward_length": 6144,
        "qwen35.attention.head_count": 8,
        "qwen35.attention.head_count_kv": 2,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
        "qwen35.ssm.conv_kernel": 4,
        "qwen35.ssm.state_size": 128,
        "qwen35.ssm.group_count": 16,
        "qwen35.ssm.time_step_rank": 16,
        "qwen35.ssm.inner_size": 2048,
        "qwen35.full_attention_interval": 4,
    }
    tensors = g.expected_tensors(mtp=False)
    assert tensors["token_embd.weight"] == ((248320, 2048), "blocks")
    assert "output.weight" not in tensors
    assert tensors["blk.0.attn_qkv.weight"] == ((6144, 2048), "blocks")
    assert tensors["blk.3.attn_q.weight"] == ((4096, 2048), "blocks")
    assert tensors["blk.3.attn_k.weight"] == ((512, 2048), "blocks")
    assert sum(g.full_attention(layer) for layer in range(g.layers)) == 6
    # One value head per key head: the exporter's tiling is the identity.
    assert (g.tiled_input_columns() == np.arange(2048)).all()


def test_tiny_geometry_validate_accepts_and_27b_refuses(tiny_path):
    with GGUFFile(tiny_path) as gguf:
        assert not TINY.has_mtp(gguf)
        TINY.validate(gguf)
        with pytest.raises(ValueError):
            gguf_blocks.validate(gguf)


def test_tiny_text_sources_cover_every_family_and_omit_the_tied_head(tiny_path):
    with GGUFFile(tiny_path) as gguf:
        encoded, direct = gguf_blocks._text_sources(TINY, gguf, mtp=False)
    assert "text/output_head" not in encoded
    assert "text/output_head" not in direct
    assert encoded["text/token_embedding"].shape == (TINY.vocabulary, TINY.hidden)
    assert encoded["text/layers/3/attention/query"].shape == (
        TINY.attention_query_rows,
        TINY.hidden,
    )
    assert encoded["text/layers/3/attention/gate"].shape == (
        TINY.attention_query_rows,
        TINY.hidden,
    )
    assert encoded["text/layers/0/gdn/query"].shape == (TINY.gdn_key_dim, TINY.hidden)
    assert encoded["text/layers/0/gdn/value"].shape == (TINY.gdn_value_dim, TINY.hidden)
    assert encoded["text/layers/0/gdn/z"].shape == (TINY.gdn_value_dim, TINY.hidden)
    assert direct["text/layers/0/gdn/a_log"].shape == (TINY.gdn_value_heads,)
    assert direct["text/layers/0/gdn/dt_bias"].shape == (TINY.gdn_value_heads,)
    assert direct["text/layers/0/gdn/convolution"].shape == (
        TINY.gdn_taps,
        TINY.gdn_channels,
    )
    # ssm_a = -2 becomes A_log = ln 2.
    assert torch.allclose(direct["text/layers/0/gdn/a_log"].values(),
                          torch.full((TINY.gdn_value_heads,), float(np.log(2.0))))


def test_tiny_gdn_controls_keep_the_stored_order_for_equal_head_counts(tiny_path):
    with GGUFFile(tiny_path) as gguf:
        source = TINY.gdn_control_source(gguf, "blk.0.ssm_alpha.weight")
        values = source.rows(0, TINY.gdn_value_heads)
    assert values.shape == (TINY.gdn_value_heads, TINY.hidden)
    assert values.dtype == torch.bfloat16
    assert not values.any()


def test_2b_recipe_registered():
    assert gguf_blocks.RECIPES["qwen3_5_2b_gguf"] is gguf_blocks.qwen3_5_2b_gguf


@pytest.mark.skipif(not REAL_PREFIX.exists(), reason="the real 2B GGUF prefix is absent")
def test_2b_real_release_header_facts():
    with GGUFFile(REAL_PREFIX) as gguf:
        assert gguf.kv["general.name"] == "Qwen3.5-2B"
        assert gguf.kv["qwen35.block_count"] == 24
        assert gguf.tensors["token_embd.weight"].type_id == 13  # Q5_K
        assert "output.weight" not in gguf.tensors
        assert not any("nextn" in name for name in gguf.tensors)
        present = {info.type_id for info in gguf.tensors.values()}
        assert present == {0, 8, 10, 12, 13, 16, 18, 21, 22}
        from tools.artifact.formats import GGUF_FORMATS_BY_TYPE

        assert present - {0} <= set(GGUF_FORMATS_BY_TYPE)
        expected = gguf_blocks.GEOMETRY_2B.expected_tensors(mtp=False)
        assert set(gguf.tensors) == set(expected)
        assert all(
            gguf.tensors[name].shape == shape for name, (shape, _) in expected.items()
        )
