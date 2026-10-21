from __future__ import annotations

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from safetensors.torch import load_file

from tools.convert import split_mtp
from tools.convert.sources.gguf import (
    TYPE_IDS,
    TYPE_PQ2_0,
    GGUFFile,
    encode_pq2_0,
    write_gguf,
)

BLOCK = 1
MTP_NAMES = split_mtp.mtp_names(BLOCK)


def _mtp_payloads():
    generator = np.random.default_rng(17)
    payloads = []
    for name in sorted(MTP_NAMES):
        if name.endswith("attn_q.weight"):
            dense = generator.normal(0.0, 0.02, size=(8, 256)).astype(np.float32)
            blocks = gguf.quantize(dense, gguf.GGMLQuantizationType.Q8_0)
            payloads.append((name, (8, 256), TYPE_IDS["Q8_0"], blocks.tobytes()))
        elif name.endswith("eh_proj.weight"):
            values = generator.integers(-1, 2, size=(8, 1, 128)).astype(np.int8)
            scales = generator.uniform(0.01, 0.03, size=(8, 1)).astype(np.float16)
            blocks = encode_pq2_0(values, scales)
            payloads.append((name, (8, 128), TYPE_PQ2_0, blocks.tobytes()))
        elif "norm" in name:
            words = generator.normal(1.0, 0.05, size=8).astype(np.float32)
            payloads.append((name, (8,), TYPE_IDS["F32"], words.tobytes()))
        else:
            dense = generator.normal(0.0, 0.02, size=(4, 8)).astype(np.float32)
            payloads.append((name, (4, 8), TYPE_IDS["F32"], dense.tobytes()))
    return payloads


def _source(tmp_path, *, extra=(), rotated=()):
    tensors = [
        ("token_embd.weight", (4, 8), TYPE_IDS["F32"], np.ones((4, 8), np.float32).tobytes()),
        (
            "blk.0.attn_norm.weight",
            (8,),
            TYPE_IDS["F32"],
            np.ones(8, np.float32).tobytes(),
        ),
        *_mtp_payloads(),
        *extra,
    ]
    path = tmp_path / "source.gguf"
    write_gguf(
        path,
        {
            "general.architecture": "qwen35",
            "general.name": "split test",
            "general.sampling.top_p": 0.95,
            "qwen35.block_count": BLOCK + 1,
            "qwen35.nextn_predict_layers": 1,
            "qwen35.rope.dimension_sections": [11, 11, 10, 0],
            "tokenizer.ggml.tokens": ["a", "bb", "ccc"],
            **({"prism.hadamard.weight_names": list(rotated)} if rotated else {}),
        },
        tensors,
    )
    return path


def test_split_writes_a_text_only_gguf_and_the_mtp_weights(tmp_path):
    source = _source(tmp_path)
    text = tmp_path / "text.gguf"
    mtp = tmp_path / "mtp.safetensors"
    report = split_mtp.split(source, text=text, mtp=mtp)
    assert report["block"] == BLOCK
    assert report["mtp_tensors"] == len(MTP_NAMES)
    assert report["text_tensors"] == 2

    with GGUFFile(source) as original, GGUFFile(text) as split:
        assert set(split.tensors) == {"token_embd.weight", "blk.0.attn_norm.weight"}
        assert split.kv["qwen35.block_count"] == BLOCK
        assert "qwen35.nextn_predict_layers" not in split.kv
        # Every other key/value pair survives, in the source's own words: a list of strings, an
        # int array, a float and a name are all copied from the source's header bytes.
        expected = {
            key: value
            for key, value in original.kv.items()
            if key != "qwen35.nextn_predict_layers"
        }
        expected["qwen35.block_count"] = BLOCK
        assert split.kv == expected
        for key in ("general.name", "general.sampling.top_p", "tokenizer.ggml.tokens",
                    "qwen35.rope.dimension_sections"):
            begin, end = original.kv_spans[key]
            assert split.kv_spans[key][1] - split.kv_spans[key][0] == end - begin
        with text.open("rb") as handle:
            split_bytes = handle.read()
        with source.open("rb") as handle:
            source_bytes = handle.read()
        for key in ("general.name", "tokenizer.ggml.tokens"):
            begin, end = original.kv_spans[key]
            assert split_bytes[split.kv_spans[key][0]:split.kv_spans[key][1]] == source_bytes[begin:end]
        # Tensors keep their bytes, and the file's data section covers every one of them.
        for name in split.tensors:
            assert split.tensor_bytes(name).tobytes() == original.tensor_bytes(name).tobytes()
        end = max(info.offset + info.nbytes for info in split.tensors.values())
        assert split.data_bytes_available >= end

    weights = load_file(str(mtp), device="cpu")
    assert set(weights) == set(MTP_NAMES.values())
    with GGUFFile(source) as original:
        for gguf_name, source_name in MTP_NAMES.items():
            expected = split_mtp.tensor_values(original, gguf_name)
            assert np.array_equal(weights[source_name].float().numpy(), expected)


def test_split_infers_the_last_block_of_a_one_layer_prediction(tmp_path):
    with GGUFFile(_source(tmp_path)) as gguf_file:
        assert split_mtp.mtp_block(gguf_file, None) == BLOCK


def test_split_refuses_a_block_it_cannot_map(tmp_path):
    extra = [
        (
            f"blk.{BLOCK}.nextn.mystery.weight",
            (8,),
            TYPE_IDS["F32"],
            np.zeros(8, np.float32).tobytes(),
        )
    ]
    with GGUFFile(_source(tmp_path, extra=extra)) as gguf_file:
        with pytest.raises(ValueError, match="does not map"):
            split_mtp.split(
                gguf_file.path,
                text=tmp_path / "text.gguf",
                mtp=tmp_path / "mtp.safetensors",
            )


def test_split_refuses_a_rotated_mtp_block(tmp_path):
    name = f"blk.{BLOCK}.ffn_down.weight"
    with GGUFFile(_source(tmp_path, rotated=(name,))) as gguf_file:
        with pytest.raises(ValueError, match="Hadamard-rotated"):
            split_mtp.split(
                gguf_file.path,
                text=tmp_path / "text.gguf",
                mtp=tmp_path / "mtp.safetensors",
            )


def test_a_file_without_a_prediction_needs_an_explicit_block(tmp_path):
    path = tmp_path / "plain.gguf"
    write_gguf(
        path,
        {"general.architecture": "qwen35", "qwen35.block_count": 1},
        [("blk.0.attn_norm.weight", (8,), TYPE_IDS["F32"], np.zeros(8, np.float32).tobytes())],
    )
    with GGUFFile(path) as gguf_file:
        with pytest.raises(ValueError, match="name the MTP block with --block"):
            split_mtp.mtp_block(gguf_file, None)
        with pytest.raises(ValueError, match="no tensors under blk.7."):
            split_mtp.mtp_block(gguf_file, 7)
