"""The DFlash2 adapter and an MTP block read from their own GGUF, and the scale check.

Every fixture here is a real `.gguf` written by ``write_gguf``: the header keys a release states,
the tensors it stores in its own block formats, and the tiny model the adapter's parameters are
registered on. What is compared is what an artifact would store: encoded rows against the source's
own bytes, the small tensors against their F32 values, and the recipe's choices.
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
import torch

from tools.artifact.formats import GGUF_FORMATS_BY_TYPE
from tools.convert import gguf_blocks
from tools.convert.qwen3_5 import _Builder, text_config
from tools.convert.qwen3_5_gguf_companions import (
    apply_gguf_companions,
    dflash2_config,
    dflash2_sources,
    dflash2_tensor_map,
    geometry_of,
    mtp_block_of,
    validate_dflash2,
)
from tools.convert.recipe import Recipe
from tools.convert.model import Model
from tools.convert.sources.gguf import (
    SCALE_OFFSETS,
    TYPE_NAMES,
    TYPE_F32,
    TYPE_IDS,
    GGUFFile,
    write_gguf,
)

TYPE_Q2_K = TYPE_IDS["Q2_K"]
TYPE_Q3_K = TYPE_IDS["Q3_K"]
TYPE_Q4_K = TYPE_IDS["Q4_K"]
TYPE_Q6_K = TYPE_IDS["Q6_K"]

# A miniature DFlash2 adapter and Qwen3.5 tower that share one block-friendly geometry: hidden and
# intermediate are whole super-blocks, so every stored type is legal at these widths.
HIDDEN = 256
INTERMEDIATE = 512
HEAD_DIM = 128
HEADS = 2
KV_HEADS = 1
LAYERS = 2
RANK = 256
TOP_K = 2
CONV_KERNEL = 2
CONV_GROUP = 16
DRAFT_BLOCKS = 1
VOCABULARY = 64
MASK = 7


def _blocks(qtype: int, rows: int, columns: int, seed: int) -> bytes:
    spec = GGUF_FORMATS_BY_TYPE[qtype]
    assert columns % spec.block_elements == 0
    generator = np.random.default_rng(seed)
    count = rows * (columns // spec.block_elements)
    data = generator.integers(0, 256, size=(count, spec.block_bytes), dtype=np.uint8)
    # Every binary16 multiplier finite, so the stored rows are real weights: the layout states
    # where each type keeps them.
    scales = np.random.default_rng(seed + 1).uniform(0.002, 0.02, size=count).astype("<f2")
    words = scales.view(np.uint8).reshape(count, 2)
    for offset in SCALE_OFFSETS[TYPE_NAMES[qtype]]:
        data[:, offset : offset + 2] = words
    return data.tobytes()


def _f32(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype="<f4").tobytes()


def _dflash2_header(**overrides) -> dict:
    header = {
        "general.architecture": "dflash",
        "dflash.block_count": DRAFT_BLOCKS,
        "dflash.embedding_length": HIDDEN,
        "dflash.feed_forward_length": INTERMEDIATE,
        "dflash.attention.head_count": HEADS,
        "dflash.attention.head_count_kv": KV_HEADS,
        "dflash.attention.key_length": HEAD_DIM,
        "dflash.attention.value_length": HEAD_DIM,
        "dflash.context_length": 4096,
        "dflash.attention.layer_norm_rms_epsilon": 1e-6,
        "dflash.rope.freq_base": 1.0e7,
        "dflash.attention.causal": False,
        "dflash.conv_kernel_size": CONV_KERNEL,
        "dflash.conv_group_size": CONV_GROUP,
        "dflash.selector_rank": RANK,
        "dflash.selector_top_k": TOP_K,
        "dflash.target_layers": [2, 4],
        "dflash.attention.sliding_window": 512,
        "dflash.attention.sliding_window_pattern": [True] * DRAFT_BLOCKS,
        "tokenizer.ggml.mask_token_id": MASK,
    }
    header.update(overrides)
    return header


def _dflash2_tensors(qtype_of=None) -> list[tuple[str, tuple[int, ...], int, bytes]]:
    """Every stored tensor of one DFlash2 block, its types mixed as a Q2_K/Q3_K/Q4_K/Q6_K release."""

    def kind(name: str) -> int:
        if qtype_of is None:
            return TYPE_Q2_K
        return TYPE_Q2_K if "attn_q" in name else TYPE_Q4_K

    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []
    seed = 11
    for name, shape in (
        ("fc.weight", (HIDDEN, 2 * HIDDEN)),
        ("enc.output_norm.weight", (HIDDEN,)),
        ("output_norm.weight", (HIDDEN,)),
        ("selector_hidden.weight", (RANK, HIDDEN)),
        ("selector_predecessor.weight", (VOCABULARY, RANK)),
        ("selector_successor.weight", (VOCABULARY, RANK)),
    ):
        if name.endswith("_norm.weight"):
            tensors.append((name, shape, TYPE_F32, _f32(np.full(shape, 0.5))))
            continue
        qtype = kind(name)
        tensors.append((name, shape, qtype, _blocks(qtype, shape[0], shape[1], seed)))
        seed += 1
    for layer in range(DRAFT_BLOCKS):
        g = f"blk.{layer}."
        query_rows = HEADS * HEAD_DIM
        kv_rows = KV_HEADS * HEAD_DIM
        conv_rows = 2 * CONV_KERNEL * (HIDDEN // CONV_GROUP)
        stored = (
            (g + "attn_norm.weight", (HIDDEN,), TYPE_F32),
            (g + "ffn_norm.weight", (HIDDEN,), TYPE_F32),
            (g + "attn_q_norm.weight", (HEAD_DIM,), TYPE_F32),
            (g + "attn_k_norm.weight", (HEAD_DIM,), TYPE_F32),
            (g + "attn_conv_base", (2, CONV_KERNEL, HIDDEN), TYPE_F32),
            (g + "ffn_conv_base", (2, CONV_KERNEL, HIDDEN), TYPE_F32),
            (g + "attn_q.weight", (query_rows, HIDDEN), TYPE_Q2_K),
            (g + "attn_k.weight", (kv_rows, HIDDEN), TYPE_Q3_K),
            (g + "attn_v.weight", (kv_rows, HIDDEN), TYPE_Q4_K),
            (g + "attn_output.weight", (HIDDEN, query_rows), TYPE_Q3_K),
            (g + "ffn_gate.weight", (INTERMEDIATE, HIDDEN), TYPE_Q2_K),
            (g + "ffn_up.weight", (INTERMEDIATE, HIDDEN), TYPE_Q6_K),
            (g + "ffn_down.weight", (HIDDEN, INTERMEDIATE), TYPE_Q4_K),
            (g + "attn_conv_proj.weight", (conv_rows, HIDDEN), TYPE_Q2_K),
            (g + "ffn_conv_proj.weight", (conv_rows, HIDDEN), TYPE_Q4_K),
        )
        for name, shape, qtype in stored:
            if qtype == TYPE_F32:
                tensors.append((name, shape, TYPE_F32, _f32(np.full(shape, 0.75))))
                continue
            tensors.append((name, shape, qtype, _blocks(qtype, shape[0], shape[1], seed)))
            seed += 1
    return tensors


@pytest.fixture()
def dflash2_path(tmp_path):
    path = tmp_path / "draft.gguf"
    write_gguf(path, _dflash2_header(), _dflash2_tensors())
    return path


def _text_config() -> dict:
    return text_config(
        {
            "architectures": ["Qwen3_5ForCausalLM"],
            "hidden_size": HIDDEN,
            "vocab_size": VOCABULARY,
            "num_hidden_layers": 4,
            "max_position_embeddings": 4096,
            "layer_types": ["linear_attention", "full_attention"] * 2,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            # Rotary width 32 pairs, interleaved mRoPE: the sections a 4:1 pattern gives.
            "rope_parameters": {"partial_rotary_factor": 0.25, "mrope_section": [6, 5, 5]},
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 4,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 3,
            "intermediate_size": INTERMEDIATE,
        },
        mtp=False,
    )


def _model_with_draft(blocks: GGUFFile):
    """The adapter's parameters as the builder registers them, from the alias `--source` builds."""

    from tools.convert.__main__ import _companion_alias

    text = _text_config()
    draft = dflash2_config(blocks)
    model = Model(
        {"text": {"config": text}, "dflash2": {"config": draft, "target": "text"}},
        token_count=VOCABULARY,
    )
    alias = _companion_alias("dflash2", blocks, _Values(blocks))
    _Builder(model).draft(alias, draft, text, "dflash2")
    return model, draft


class _Values:
    """The name-mapped value surface of one GGUFFile, as `_companion_alias` expects."""

    def __init__(self, blocks: GGUFFile):
        self.path = blocks.path
        self.blocks = blocks

    def has(self, name: str) -> bool:
        return name in self.blocks.tensors

    def describe(self, name: str):
        from tools.convert.sources.gguf_source import TensorInfo

        info = self.blocks.info(name)
        return TensorInfo(info.shape, info.type_name)

    def read_flat(self, name, begin=0, end=None):
        raise AssertionError("a stored block companion never reads values")


def test_the_release_header_becomes_the_draft_config(dflash2_path):
    with GGUFFile(dflash2_path) as gguf:
        draft = dflash2_config(gguf)
        # One-based target layers become the target's indices, and the mask comes from the
        # tokenizer's own key when the release states no dflash.mask_token_id.
        assert draft["dflash_config"]["target_layer_ids"] == [1, 3]
        assert draft["dflash_config"]["mask_token_id"] == MASK
        assert draft["dflash_config"]["selector_rank"] == RANK
        assert draft["layer_types"] == ["sliding_attention"] * DRAFT_BLOCKS
        assert draft["sliding_window"] == 512
        assert draft["head_dim"] == HEAD_DIM
        assert draft["is_causal"] is False
        assert draft["architectures"] == ["DFlash2DraftModel"]


def test_a_plain_full_head_rope_and_an_explicit_mask_are_accepted(dflash2_path):
    with GGUFFile(dflash2_path) as gguf:
        accepted = dflash2_config(
            gguf,
        )
        assert "dflash.rope.dimension_sections" not in gguf.kv
        assert accepted["dflash_config"]["target_layer_ids"] == [1, 3]
    path = dflash2_path.parent / "sections.gguf"
    write_gguf(
        path,
        _dflash2_header(**{
            "dflash.rope.dimension_sections": [HEAD_DIM // 2, 0, 0, 0],
            "dflash.mask_token_id": 3,
        }),
        _dflash2_tensors(),
    )
    with GGUFFile(path) as gguf:
        draft = dflash2_config(gguf)
        assert draft["dflash_config"]["mask_token_id"] == 3
    path = dflash2_path.parent / "mrope.gguf"
    write_gguf(
        path,
        _dflash2_header(**{"dflash.rope.dimension_sections": [1, 1, 1, 0]}),
        _dflash2_tensors(),
    )
    with GGUFFile(path) as gguf, pytest.raises(ValueError, match="dimension_sections"):
        dflash2_config(gguf)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"dflash.attention.value_length": HEAD_DIM // 2}, "key/value length"),
        ({"dflash.target_layers": [0, 3]}, "1-based"),
        ({"dflash.attention.sliding_window_pattern": [True, False]}, "per block"),
        ({"general.architecture": "qwen35"}, "general.architecture"),
        ({"tokenizer.ggml.mask_token_id": None}, "mask_token_id"),
        ({"dflash.selector_top_k": 0}, "selector_top_k"),
    ],
)
def test_a_header_outside_the_adapter_is_refused(dflash2_path, override, message):
    path = dflash2_path.parent / "bad.gguf"
    header = {k: v for k, v in _dflash2_header(**override).items() if v is not None}
    write_gguf(path, header, _dflash2_tensors())
    with GGUFFile(path) as gguf, pytest.raises(ValueError, match=message):
        dflash2_config(gguf)


def test_the_adapter_imports_every_matrix_as_its_stored_blocks_and_its_small_tensors_as_bf16(
    dflash2_path,
):
    with GGUFFile(dflash2_path) as gguf:
        model, draft = _model_with_draft(gguf)
        validate_dflash2(model, gguf, draft)
        encoded, direct = dflash2_sources(model, gguf, draft)
        stored = {tensor for _, tensor in dflash2_tensor_map(draft).values()}
        assert len(encoded) + len(direct) == len(stored)
        for name, source in encoded.items():
            tensor = dflash2_tensor_map(draft)[name][1]
            words = source.read_encoded(0, source.shape[0])
            assert words.format == gguf_blocks.block_format(gguf, tensor)
            info = gguf.info(tensor)
            block = GGUF_FORMATS_BY_TYPE[info.type_id]
            rows = gguf.read_blocks(tensor)
            assert words.codes.numpy().tobytes() == rows.tobytes()
            assert block.block_bytes * (info.shape[1] // block.block_elements) == rows.shape[1]
        for name, source in direct.items():
            tensor = dflash2_tensor_map(draft)[name][1]
            values = source.values()
            assert values.dtype == torch.bfloat16
            # Logical values are flat; the stored tensor keeps its rank.
            assert np.array_equal(
                values.to(torch.float32).numpy(),
                gguf.read_direct(tensor).reshape(-1).astype(np.float32),
            )


def test_the_recipe_stores_the_adapter_blocks_shared_context_and_a8_permissions(dflash2_path):
    with GGUFFile(dflash2_path) as gguf:
        model, draft = _model_with_draft(gguf)
        recipe = Recipe(model)
        recipe.assign(
            "dflash2/feature_projection",
            format="q8_g32_fp16",
            layout="row_split_k128_v1",
            method="grouped_absmax",
        )
        apply_gguf_companions(model, recipe, {"dflash2": gguf})
        mapped = dflash2_tensor_map(draft)
        for name, choices in recipe.selections.items():
            if name not in mapped:  # the shared context projections have no tensor of their own
                continue
            tensor = mapped[name][1]
            info = gguf.info(tensor)
            if info.type_id in GGUF_FORMATS_BY_TYPE:
                assert len(choices) == 1
                assert choices[0].format == gguf_blocks.block_format(gguf, tensor)
                assert choices[0].source.read_encoded(0, 1).format == choices[0].format
            else:
                assert choices[0].format == "bf16"
        # Every Linear Use of the adapter permits the format's own activation arithmetic...
        for (name, _), policy in recipe.policies.items():
            if name in recipe.selections:
                assert policy == "AllowA8"
        # ...and one physical copy backs both Uses of a context projection.
        for layer in range(DRAFT_BLOCKS):
            prefix = f"dflash2/layers/{layer}/attention/"
            for role in ("key", "value"):
                assert recipe.aliases[prefix + "context_" + role] == prefix + role
                assert recipe.selections[prefix + "context_" + role] == recipe.selections[
                    prefix + role
                ]
        prepared = recipe.prepare(device="cpu")
        formats = {
            quantum.spec.format for quantum in prepared.weights
        }
        assert {"gguf_q2_k", "gguf_q3_k", "gguf_q4_k", "gguf_q6_k"} <= formats
        assert "bf16" in formats
        for weight in prepared.weights:
            if weight.spec.format.startswith("gguf_"):
                assert weight.spec.layout == "gguf_blocks_v1"


def test_a_doubled_query_matrix_is_not_a_draft_query(dflash2_path):
    tensors = []
    for name, shape, qtype, data in _dflash2_tensors():
        if name == "blk.0.attn_q.weight":
            shape = (2 * shape[0], shape[1])
            data = _blocks(qtype, shape[0], shape[1], 99)
        tensors.append((name, shape, qtype, data))
    path = dflash2_path.parent / "query-gate.gguf"
    write_gguf(path, _dflash2_header(), tensors)
    with GGUFFile(path) as gguf:
        model, draft = _model_with_draft(gguf)
        with pytest.raises(ValueError, match="attn_q.weight"):
            validate_dflash2(model, gguf, draft)


def test_a_tensor_of_another_shape_is_refused(dflash2_path):
    tensors = []
    for name, shape, qtype, data in _dflash2_tensors():
        if name == "blk.0.attn_conv_proj.weight":
            shape = (shape[0] // 2, shape[1])
            data = _blocks(TYPE_Q2_K, shape[0], shape[1], 99)
        tensors.append((name, shape, qtype, data))
    path = dflash2_path.parent / "narrow.gguf"
    write_gguf(path, _dflash2_header(), tensors)
    with GGUFFile(path) as gguf:
        model, draft = _model_with_draft(gguf)
        with pytest.raises(ValueError, match="attn_conv_proj"):
            validate_dflash2(model, gguf, draft)


def _small_geometry():
    return replace(
        gguf_blocks.GEOMETRY_27B,
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
        vocabulary=VOCABULARY,
        layers=4,
        attention_heads=HEADS,
        attention_head_dim=HEAD_DIM,
        attention_kv_heads=KV_HEADS,
        gdn_key_heads=1,
        gdn_value_heads=2,
        full_attention_interval=2,
    )


def _mtp_tensors(
    block: int, *, geometry, qtype: int = TYPE_Q6_K
) -> list[tuple[str, tuple[int, ...], int, bytes]]:
    p = f"blk.{block}."
    out: list[tuple[str, tuple[int, ...], int, bytes]] = [
        (p + "attn_norm.weight", (geometry.hidden,), TYPE_F32, _f32(np.full(geometry.hidden, 0.5))),
        (p + "post_attention_norm.weight", (geometry.hidden,), TYPE_F32,
         _f32(np.full(geometry.hidden, 0.5))),
        (p + "attn_q_norm.weight", (geometry.attention_head_dim,), TYPE_F32,
         _f32(np.full(geometry.attention_head_dim, 1.0))),
        (p + "attn_k_norm.weight", (geometry.attention_head_dim,), TYPE_F32,
         _f32(np.full(geometry.attention_head_dim, 1.0))),
        (p + "nextn.enorm.weight", (geometry.hidden,), TYPE_F32,
         _f32(np.full(geometry.hidden, 0.5))),
        (p + "nextn.hnorm.weight", (geometry.hidden,), TYPE_F32,
         _f32(np.full(geometry.hidden, 0.5))),
        (p + "nextn.shared_head_norm.weight", (geometry.hidden,), TYPE_F32,
         _f32(np.full(geometry.hidden, 0.5))),
    ]
    seed = 71
    for name, shape in (
        (p + "attn_q.weight", (2 * geometry.attention_query_rows, geometry.hidden)),
        (p + "attn_k.weight", (geometry.attention_kv_rows, geometry.hidden)),
        (p + "attn_v.weight", (geometry.attention_kv_rows, geometry.hidden)),
        (p + "attn_output.weight", (geometry.hidden, geometry.attention_query_rows)),
        (p + "ffn_gate.weight", (geometry.intermediate, geometry.hidden)),
        (p + "ffn_up.weight", (geometry.intermediate, geometry.hidden)),
        (p + "ffn_down.weight", (geometry.hidden, geometry.intermediate)),
        (p + "nextn.eh_proj.weight", (geometry.hidden, 2 * geometry.hidden)),
    ):
        out.append((name, shape, qtype, _blocks(qtype, shape[0], shape[1], seed)))
        seed += 1
    return out


def test_an_mtp_block_is_imported_from_its_own_file_with_or_without_the_tower(tmp_path):
    geometry = _small_geometry()
    header = geometry.expected_header() | {
        "qwen35.block_count": geometry.layers + 1,
        "qwen35.nextn_predict_layers": 1,
    }
    full = tmp_path / "full.gguf"
    tower = []
    for name, (shape, kind) in geometry.expected_tensors(False).items():
        if kind == "blocks":
            tower.append((name, shape, TYPE_IDS["Q8_0"],
                          _blocks(TYPE_IDS["Q8_0"], shape[0], shape[1], 3)))
        else:
            tower.append((name, shape, TYPE_F32, _f32(np.full(shape, 1.0))))
    write_gguf(full, header, tower + _mtp_tensors(geometry.layers, geometry=geometry))
    with GGUFFile(full) as gguf:
        assert mtp_block_of(gguf) == geometry.layers
        assert geometry.validate_mtp(gguf) == geometry.layers
        encoded, direct = gguf_blocks.mtp_sources(geometry, gguf, geometry.layers)
        # The exporter's `1 + w` offset is undone for every norm, and each matrix keeps its blocks.
        block = f"blk.{geometry.layers}."
        expected = {
            "mtp/input_projection": block + "nextn.eh_proj.weight",
            "mtp/layers/0/input_norm": block + "attn_norm.weight",
            "mtp/layers/0/post_attention_norm": block + "post_attention_norm.weight",
            "mtp/layers/0/attention/query": block + "attn_q.weight",
            "mtp/layers/0/attention/gate": block + "attn_q.weight",
            "mtp/layers/0/attention/key": block + "attn_k.weight",
            "mtp/layers/0/attention/value": block + "attn_v.weight",
            "mtp/layers/0/attention/output": block + "attn_output.weight",
            "mtp/layers/0/attention/query_norm": block + "attn_q_norm.weight",
            "mtp/layers/0/attention/key_norm": block + "attn_k_norm.weight",
            "mtp/embedding_norm": block + "nextn.enorm.weight",
            "mtp/hidden_norm": block + "nextn.hnorm.weight",
            "mtp/final_norm": block + "nextn.shared_head_norm.weight",
            "mtp/layers/0/mlp/gate": block + "ffn_gate.weight",
            "mtp/layers/0/mlp/up": block + "ffn_up.weight",
            "mtp/layers/0/mlp/down": block + "ffn_down.weight",
        }
        assert set(encoded) | set(direct) == set(expected)
        for name, tensor in expected.items():
            source = encoded.get(name) or direct[name]
            if source.read_encoded is None:
                continue
            assert source.read_encoded(0, 1).format == gguf_blocks.block_format(gguf, tensor)
        first = direct["mtp/embedding_norm"].values().to(torch.float32)
        assert float(first[0]) == pytest.approx(0.5 - 1.0, rel=0, abs=1e-3)
    block_only = tmp_path / "block.gguf"
    write_gguf(
        block_only,
        header | {"qwen35.block_count": geometry.layers + 1},
        _mtp_tensors(geometry.layers, geometry=geometry),
    )
    with GGUFFile(block_only) as gguf:
        assert geometry.validate_mtp(gguf) == geometry.layers
    # A file that declares no predicted layer, or holds a tensor the block never has, is refused.
    plain = tmp_path / "plain.gguf"
    write_gguf(plain, header | {"qwen35.nextn_predict_layers": 0},
               _mtp_tensors(geometry.layers, geometry=geometry))
    with GGUFFile(plain) as gguf, pytest.raises(ValueError, match="nextn_predict_layers"):
        geometry.validate_mtp(gguf)
    extra = tmp_path / "extra.gguf"
    write_gguf(extra, header,
               _mtp_tensors(geometry.layers, geometry=geometry) +
               [(f"blk.{geometry.layers}.attn_gate.weight",
                 (geometry.hidden, geometry.hidden), TYPE_IDS["Q8_0"],
                 _blocks(TYPE_IDS["Q8_0"], geometry.hidden, geometry.hidden, 5))])
    with GGUFFile(extra) as gguf, pytest.raises(ValueError, match="MTP block"):
        geometry.validate_mtp(gguf)


@pytest.mark.parametrize(
    "key", ["prism.hadamard.weight_names", "prism.hadamard.inverse_weight_names"]
)
def test_an_external_mtp_refuses_its_own_rotations_but_not_the_towers(tmp_path, key):
    geometry = _small_geometry()
    block = geometry.layers
    header = geometry.expected_header() | {
        "qwen35.block_count": block + 1,
        "qwen35.nextn_predict_layers": 1,
        key: ["blk.0.ffn_down.weight"],
    }
    tensors = _mtp_tensors(block, geometry=geometry)
    path = tmp_path / "unrotated-mtp.gguf"
    write_gguf(path, header, tensors)
    with GGUFFile(path) as gguf:
        assert geometry.validate_mtp(gguf) == block
        encoded, _ = gguf_blocks.mtp_sources(geometry, gguf, block)
        assert encoded["mtp/layers/0/mlp/down"].read_encoded(0, HIDDEN).codes.numpy().tobytes() == (
            gguf.read_blocks(f"blk.{block}.ffn_down.weight").tobytes()
        )
    rotated = tmp_path / "rotated-mtp.gguf"
    write_gguf(rotated, header | {key: [f"blk.{block}.ffn_down.weight"]}, tensors)
    with GGUFFile(rotated) as gguf, pytest.raises(ValueError, match="Hadamard-rotated"):
        geometry.validate_mtp(gguf)


def test_apply_gguf_companions_leaves_a_companion_it_was_not_given_alone(dflash2_path):
    with GGUFFile(dflash2_path) as gguf:
        model, _ = _model_with_draft(gguf)
        recipe = Recipe(model)
        replacements = {name: choices[0].format for name, choices in recipe.selections.items()}
        apply_gguf_companions(model, recipe, {})
        assert {
            name: choices[0].format for name, choices in recipe.selections.items()
        } == replacements


def test_a_stored_multiplier_that_is_not_finite_is_refused(tmp_path):
    tensors = _dflash2_tensors()
    name, shape, qtype, data = tensors[0]  # fc.weight, Q2_K: its d sits at offset 80
    raw = bytearray(data)
    raw[80:82] = np.array([0x7E00], dtype="<u2").tobytes()  # binary16 NaN
    tensors[0] = (name, shape, qtype, bytes(raw))
    path = tmp_path / "nan.gguf"
    write_gguf(path, _dflash2_header(), tensors)
    with GGUFFile(path) as gguf:
        count, first = gguf.non_finite_scales(name)
        assert count == 1 and first == 0
        with pytest.raises(ValueError, match="non-finite multiplier"):
            gguf.require_finite_scales([name])


def test_the_mlp_tower_geometry_is_read_from_the_text_config():
    text = _text_config()
    geometry = geometry_of(text)
    assert geometry.hidden == HIDDEN
    assert geometry.intermediate == INTERMEDIATE
    assert geometry.full_attention_interval == 2
    assert geometry.attention_heads == HEADS
