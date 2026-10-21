from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.artifact.codecs.row_split import decode_row_split_codes, encode_row_split
from tools.convert import ternary
from tools.convert.sources.gguf import (
    TYPE_IDS,
    TYPE_PQ2_0,
    GGUFFile,
    TernaryBlocks,
    encode_pq2_0,
    write_gguf,
)


def _ternary_rows(rows: int, columns: int, seed: int):
    generator = np.random.default_rng(seed)
    values = generator.integers(-1, 2, size=(rows, columns // 128, 128)).astype(np.int8)
    scales = generator.uniform(0.004, 0.03, size=(rows, columns // 128)).astype(
        np.float16
    )
    return values, scales


def _dense(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (values.astype(np.float32) * scales.astype(np.float32)[:, :, None]).reshape(
        values.shape[0], -1
    )


def _pq2_0_file(tmp_path, rows: int, columns: int, seed: int):
    values, scales = _ternary_rows(rows, columns, seed)
    blocks = encode_pq2_0(values, scales)
    path = tmp_path / "sample.gguf"
    write_gguf(
        path,
        {"general.architecture": "qwen35"},
        [("w", (rows, columns), TYPE_PQ2_0, blocks.tobytes())],
    )
    return path, values, scales


def test_ternary_source_passes_codes_through_and_decodes_exact_values(tmp_path):
    path, values, scales = _pq2_0_file(tmp_path, 12, 256, 3)
    with GGUFFile(path) as gguf:
        source = ternary.ternary_source(gguf, "w", (12, 256), ternary.rows())
        words = source.read_encoded(2, 7)
        assert words.format == "t2_g128_fp16"
        assert torch.equal(words.codes, torch.from_numpy(values[2:7]))
        assert torch.equal(words.scales, torch.from_numpy(scales[2:7]))
        expected = (
            values[2:7].astype(np.float32) * scales[2:7].astype(np.float32)[:, :, None]
        )
        assert torch.equal(
            source.rows(2, 7), torch.from_numpy(expected.reshape(5, 256))
        )
        payload = encode_row_split(words.codes, words.scales, "t2_g128_fp16", (5, 256))
        decoded_scales, decoded_codes = decode_row_split_codes(
            payload, "t2_g128_fp16", (5, 256)
        )
        assert torch.equal(decoded_codes, words.codes)
        assert torch.equal(decoded_scales, words.scales)


def test_selected_rows_follow_the_row_map(tmp_path):
    path, values, _ = _pq2_0_file(tmp_path, 16, 128, 5)
    order = np.array([9, 3, 12, 3], dtype=np.int64)
    with GGUFFile(path) as gguf:
        source = ternary.ternary_source(gguf, "w", (4, 128), lambda b, e: order[b:e])
        assert torch.equal(
            source.read_encoded(0, 4).codes, torch.from_numpy(values[order])
        )
        assert torch.equal(
            source.read_encoded(1, 3).codes, torch.from_numpy(values[order[1:3]])
        )


def test_grouped_value_heads_pair_with_their_key_head():
    permutation = ternary.tiled_to_grouped_permutation()
    assert sorted(permutation.tolist()) == list(range(48))
    for head in range(48):
        tiled = permutation[head]
        assert tiled % 16 == head // 3 and tiled // 16 == head % 3
    tiled = np.arange(48 * 2).reshape(96, 1)
    grouped = ternary.untile(tiled, 2)
    assert grouped[2:4, 0].tolist() == [32, 33]
    rows = ternary.untiled_rows(4096)(128, 130)
    assert rows.tolist() == [4096 + 16 * 128, 4096 + 16 * 128 + 1]


def test_query_and_gate_rows_split_each_interleaved_head():
    query = ternary.attention_rows(False)(250, 260)
    gate = ternary.attention_rows(True)(0, 3)
    assert query.tolist() == [250, 251, 252, 253, 254, 255, 512, 513, 514, 515]
    assert gate.tolist() == [256, 257, 258]


def test_sign_auxiliary_is_a_bf16_vector():
    signs = torch.tensor([1.0, -1.0] * 512)
    value = ternary.sign_auxiliary(signs)
    assert value.format == "bf16" and value.shape == (1024,)
    words = np.frombuffer(value.data, dtype="<u2")
    assert words[0] == 0x3F80 and words[1] == 0xBF80


def test_stored_ternary_keeps_every_code_of_a_larger_block_format():
    gguf = pytest.importorskip("gguf")
    rows, columns = 8, 512
    values, scales = _ternary_rows(rows, columns, 11)
    dense = _dense(values, scales)
    blocks = gguf.quantize(dense, gguf.GGMLQuantizationType.Q8_0)
    quantized = gguf.dequantize(blocks.reshape(rows, -1), gguf.GGMLQuantizationType.Q8_0)
    stored, clipped = ternary.stored_ternary(quantized.reshape(rows, columns))
    assert clipped == 0
    # The block type rounds each group's scale; every code survives, which is what the artifact
    # keeps. (The 2-bit sources measured in the fork's mix recover the codes exactly.)
    assert np.array_equal(
        stored.values.astype(np.int32).reshape(rows, columns),
        np.sign(dense).astype(np.int32),
    )
    assert np.allclose(
        stored.values.astype(np.float32) * stored.scales.astype(np.float32)[:, :, None],
        quantized.reshape(rows, columns // 128, 128),
        rtol=0.25,
        atol=0.0,
    )


def test_stored_ternary_stores_the_quantizer_fourth_level_at_the_group_scale():
    generator = np.random.default_rng(5)
    rows, groups = 4, 2
    scale = generator.uniform(0.01, 0.03, size=(rows, groups)).astype(np.float32)
    codes = (generator.integers(0, 3, size=(rows, groups, 128)) - 1).astype(np.int8)
    codes[:, :, :4] = 2  # the quantizer's +2 level, which the artifact's T2 codes cannot hold
    dense = (codes.astype(np.float32) * scale[:, :, None]).reshape(rows, groups * 128)
    blocks, clipped = ternary.stored_ternary(dense)
    assert clipped == rows * groups * 4
    assert np.array_equal(blocks.values, np.where(codes == 2, np.int8(1), codes))
    assert np.allclose(blocks.scales, scale, rtol=1e-3)


def test_stored_ternary_refuses_values_that_are_not_a_ternary_matrix():
    generator = np.random.default_rng(7)
    dense = generator.normal(0.0, 0.02, size=(8, 512)).astype(np.float32)
    with pytest.raises(ValueError, match="not a ternary matrix"):
        ternary.stored_ternary(dense)


def test_stored_ternary_source_reads_rows_and_selected_rows(tmp_path):
    gguf = pytest.importorskip("gguf")
    rows, columns = 16, 512
    values, scales = _ternary_rows(rows, columns, 23)
    blocks = gguf.quantize(_dense(values, scales), gguf.GGMLQuantizationType.Q8_0)
    path = tmp_path / "block.gguf"
    write_gguf(
        path,
        {"general.architecture": "qwen35"},
        [("w", (rows, columns), TYPE_IDS["Q8_0"], blocks.tobytes())],
    )
    order = np.array([9, 3, 12, 3], dtype=np.int64)
    with GGUFFile(path) as source_file:
        expected, _ = ternary.stored_ternary(source_file.read_dequantized("w"))
        source = ternary.stored_ternary_source(
            source_file, "w", (rows, columns), ternary.rows()
        )
        words = source.read_encoded(2, 6)
        assert words.format == "t2_g128_fp16"
        assert torch.equal(words.codes, torch.from_numpy(expected.values[2:6]))
        assert torch.equal(words.scales, torch.from_numpy(expected.scales[2:6]))
        selected = ternary.stored_ternary_source(
            source_file, "w", (4, columns), lambda begin, end: order[begin:end]
        )
        assert torch.equal(
            selected.read_encoded(1, 3).codes,
            torch.from_numpy(expected.values[order[1:3]]),
        )


def test_ternary_matrix_accepts_a_ternary_type_and_a_larger_block_type(tmp_path):
    gguf = pytest.importorskip("gguf")
    rows, columns = 8, 512
    values, scales = _ternary_rows(rows, columns, 31)
    path = tmp_path / "block.gguf"
    write_gguf(
        path,
        {"general.architecture": "qwen35"},
        [
            (
                "w",
                (rows, columns),
                TYPE_IDS["Q8_0"],
                gguf.quantize(
                    _dense(values, scales), gguf.GGMLQuantizationType.Q8_0
                ).tobytes(),
            )
        ],
    )
    pq2_path, pq2_values, pq2_scales = _pq2_0_file(tmp_path, rows, columns, 31)
    with GGUFFile(path) as block_file:
        block = ternary.ternary_matrix(block_file, "w", (rows, columns))
        assert "Q8_0" in block.label
        assert torch.equal(
            block.read_encoded(0, rows).codes,
            torch.from_numpy(ternary.stored_ternary(block_file.read_dequantized("w"))[0].values),
        )
    with GGUFFile(pq2_path) as block_file:
        stored = ternary.ternary_matrix(block_file, "w", (rows, columns))
        assert torch.equal(
            stored.read_encoded(0, rows).codes, torch.from_numpy(pq2_values)
        )
        assert torch.equal(
            stored.read_encoded(0, rows).scales, torch.from_numpy(pq2_scales)
        )


def test_a_control_projection_comes_back_untiled(tmp_path):
    rows, columns = 48, 128
    values, scales = _ternary_rows(rows, columns, 5)
    blocks = encode_pq2_0(values, scales)
    path = tmp_path / "controls.gguf"
    write_gguf(
        path,
        {"general.architecture": "qwen35"},
        [("w", (rows, columns), TYPE_PQ2_0, blocks.tobytes())],
    )
    expected = torch.from_numpy(ternary.untile(_dense(values, scales), 1)).to(
        torch.bfloat16
    )
    with GGUFFile(path) as source_file:
        got = ternary.control_source(source_file, "w").rows(0, rows)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, expected)


def test_a_bf16_control_projection_keeps_its_words(tmp_path):
    rows, columns = 48, 128
    generator = np.random.default_rng(13)
    dense = generator.normal(0.0, 0.02, size=(rows, columns)).astype(np.float32)
    words = (
        torch.from_numpy(dense).to(torch.bfloat16).view(torch.int16).numpy().astype("<u2")
    )
    path = tmp_path / "controls.gguf"
    write_gguf(
        path,
        {"general.architecture": "qwen35"},
        [("w", (rows, columns), TYPE_IDS["BF16"], words.tobytes())],
    )
    expected = torch.from_numpy(ternary.untile(dense, 1)).to(torch.bfloat16)
    with GGUFFile(path) as source_file:
        got = ternary.control_source(source_file, "w").rows(0, rows)
    assert torch.equal(got, expected)


def test_validate_accepts_the_quantizers_own_mix_per_role():
    storage = ternary._accepted_storage
    assert storage("output.weight", "PQ2_0") == ternary.MATRIX_STORAGE["output.weight"]
    assert storage("token_embd.weight", "PQ2_0") == ternary.MATRIX_STORAGE[
        "token_embd.weight"
    ]
    assert "Q6_K" in ternary.MATRIX_STORAGE["output.weight"]
    assert "Q4_K" in ternary.MATRIX_STORAGE["token_embd.weight"]
    assert storage("blk.7.ssm_alpha.weight", "BF16") == ternary.CONTROL_STORAGE
    assert storage("blk.7.ssm_beta.weight", "BF16") == ternary.CONTROL_STORAGE
    assert "PQ2_0" in ternary.CONTROL_STORAGE
    # Every other role is read in its registered type only.
    assert storage("blk.7.attn_qkv.weight", "PQ2_0") == frozenset({"PQ2_0"})
    assert storage("blk.7.ssm_a", "F32") == frozenset({"F32"})
    assert storage("output_norm.weight", "F32") == frozenset({"F32"})
