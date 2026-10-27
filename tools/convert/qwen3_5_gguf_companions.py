"""Qwen3.5/3.8 companions read from their own GGUF: the DFlash2 adapter and an MTP block.

A Qwen3.8-27B target keeps its low-precision drafter: ``z-lab``'s DFlash2 adapter is published as
a small GGUF (``general.architecture = dflash``, five blocks, 81 tensors), and a release such as
PrismML's ternary GGUF carries its MTP head as one more block (``blk.<layers>.*``nextn). Both
arrive here as stored ggml blocks, exactly as ``gguf_blocks`` imports the text tower, so no matrix
is re-quantised: a row of the artifact is a row of the source.

The adapter's tensors use their own names and conventions, none of them llama.cpp's Qwen3.5 ones:

* the feature projection (``fc.weight``), the context norm (``enc.output_norm.weight``) and the
  final norm (``output_norm.weight``) are not per-layer, and the norms carry no ``1 + w`` offset
  (verified against the release's own ``model.safetensors``);
* the two dynamic-convolution base kernels are stored C-order ``[2, 2, 5120]``, which is the
  artifact's own layout: no transpose, unlike the exporter's GDN convolution;
* the candidate selector is ``selector_hidden.weight``, ``selector_predecessor.weight`` and
  ``selector_successor.weight``, the codebooks one row per padded token id;
* ``dflash.target_layers`` counts the target's layers from one.

``draft_config`` stays the single authority for the adapter's mathematics: this module only
translates the GGUF's header into that function's input and refuses a file whose tensors are not
the logical parameters the builder registered.
"""

from __future__ import annotations

import numpy as np
import torch

from tools.artifact.formats import GGUF_FORMATS_BY_TYPE

from .gguf_blocks import Geometry, block_source, mtp_sources
from .methods import cast_direct, import_encoded
from .sources.gguf import GGUFFile
from .sources.logical import LogicalSource
from .ternary import _direct, rows

DFLASH2_ARCHITECTURE = "dflash"
MTP_ARCHITECTURE = "qwen35"
MTP_LAYERS_KEY = "qwen35.nextn_predict_layers"
BLOCK_COUNT_KEY = "qwen35.block_count"


def _integer(gguf: GGUFFile, key: str) -> int:
    value = gguf.kv.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{gguf.path}: {key} = {value!r}, expected a positive integer")
    return value


def _real(gguf: GGUFFile, key: str) -> float:
    value = gguf.kv.get(key)
    if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{gguf.path}: {key} = {value!r}, expected a positive real")
    return float(value)


def full_attention_interval(kinds: list[str]) -> int:
    """The interval whose every last layer is a full-attention block."""

    for interval in range(1, len(kinds) + 1):
        if all(
            (kind == "full_attention") == (index % interval == interval - 1)
            for index, kind in enumerate(kinds)
        ):
            return interval
    raise ValueError("the text layer types have no single full-attention interval")


def geometry_of(config: dict) -> Geometry:
    """One Qwen3.5/3.8 text tower's geometry, as the GGUF block importers describe it."""

    return Geometry(
        hidden=config["hidden_size"],
        intermediate=config["intermediate_size"],
        vocabulary=config["vocab_size"],
        layers=config["num_hidden_layers"],
        attention_heads=config["num_attention_heads"],
        attention_head_dim=config["head_dim"],
        attention_kv_heads=config["num_key_value_heads"],
        gdn_key_heads=config["linear_num_key_heads"],
        gdn_value_heads=config["linear_num_value_heads"],
        gdn_head_dim=config["linear_value_head_dim"],
        gdn_taps=config["linear_conv_kernel_dim"],
        full_attention_interval=full_attention_interval(config["layer_types"]),
        tied=config["tie_word_embeddings"],
    )


def dflash2_tensor_map(config: dict) -> dict[str, tuple[str, str]]:
    """Every DFlash2 logical parameter's (checkpoint name, GGUF tensor name).

    The checkpoint names are what ``qwen3_5._Builder.draft`` addresses its companion with; the
    GGUF names are what the release stores. The context key/value projections read their layer's
    own key/value tensors, so they need no entry of their own.
    """

    out: dict[str, tuple[str, str]] = {
        "dflash2/feature_projection": ("fc.weight", "fc.weight"),
        "dflash2/context_norm": ("hidden_norm.weight", "enc.output_norm.weight"),
        "dflash2/final_norm": ("norm.weight", "output_norm.weight"),
        "dflash2/candidate_selector/hidden_projection": (
            "candidate_selector.hidden_projection.weight",
            "selector_hidden.weight",
        ),
        "dflash2/candidate_selector/predecessor_codebook": (
            "candidate_selector.predecessor_codebook",
            "selector_predecessor.weight",
        ),
        "dflash2/candidate_selector/successor_codebook": (
            "candidate_selector.successor_codebook",
            "selector_successor.weight",
        ),
    }
    for layer in range(config["num_hidden_layers"]):
        p, hf, g = f"dflash2/layers/{layer}/", f"layers.{layer}.", f"blk.{layer}."
        out[p + "input_norm"] = (hf + "input_layernorm.weight", g + "attn_norm.weight")
        out[p + "post_attention_norm"] = (
            hf + "post_attention_layernorm.weight",
            g + "ffn_norm.weight",
        )
        for role, source, tensor in (
            ("query", "self_attn.q_proj.weight", "attn_q.weight"),
            ("key", "self_attn.k_proj.weight", "attn_k.weight"),
            ("value", "self_attn.v_proj.weight", "attn_v.weight"),
            ("output", "self_attn.o_proj.weight", "attn_output.weight"),
        ):
            out[p + "attention/" + role] = (hf + source, g + tensor)
        for role, source, tensor in (
            ("query_norm", "self_attn.q_norm.weight", "attn_q_norm.weight"),
            ("key_norm", "self_attn.k_norm.weight", "attn_k_norm.weight"),
        ):
            out[p + "attention/" + role] = (hf + source, g + tensor)
        for role in ("gate", "up", "down"):
            out[p + "mlp/" + role] = (
                hf + f"mlp.{role}_proj.weight",
                g + f"ffn_{role}.weight",
            )
        for branch, prefix, tensor in (
            ("attention_conv", "attn", "attn_conv"),
            ("mlp_conv", "ffn", "ffn_conv"),
        ):
            out[p + branch + "/base_kernel"] = (
                hf + branch + ".base_kernel",
                g + tensor + "_base",
            )
            out[p + branch + "/kernel_projection"] = (
                hf + branch + ".kernel_projection.weight",
                g + prefix + "_conv_proj.weight",
            )
    return out


def dflash2_config(gguf: GGUFFile) -> dict:
    """The release's header as the checkpoint names ``qwen3_5.draft_config`` reads."""

    if gguf.kv.get("general.architecture") != DFLASH2_ARCHITECTURE:
        raise ValueError(
            f"{gguf.path}: general.architecture = {gguf.kv.get('general.architecture')!r}, "
            f"expected {DFLASH2_ARCHITECTURE!r}"
        )
    layers = _integer(gguf, "dflash.block_count")
    head_dim = _integer(gguf, "dflash.attention.key_length")
    value_length = gguf.kv.get("dflash.attention.value_length", head_dim)
    if value_length != head_dim:
        raise ValueError(
            f"{gguf.path}: key/value length {head_dim}/{value_length}; the adapter rotates whole "
            "heads and needs one width"
        )
    causal = gguf.kv.get("dflash.attention.causal", False)
    if causal is not False:
        raise ValueError(f"{gguf.path}: dflash.attention.causal = {causal!r}, expected False")
    sections = gguf.kv.get("dflash.rope.dimension_sections")
    if sections is not None:
        rotated = [head_dim // 2, 0, 0, 0]
        if list(sections) not in (rotated, rotated[:1]):
            # Anything else is an M-RoPE schedule this adapter's plain head rope cannot express.
            raise ValueError(
                f"{gguf.path}: dflash.rope.dimension_sections = {list(sections)!r}; the adapter "
                f"rotates whole heads ({rotated!r} or {rotated[:1]!r})"
            )
    taps = gguf.kv.get("dflash.target_layers")
    if (
        not isinstance(taps, list)
        or not taps
        or any(type(index) is not int or index < 1 for index in taps)
        or len(set(taps)) != len(taps)
    ):
        raise ValueError(f"{gguf.path}: dflash.target_layers = {taps!r} is not a set of 1-based ids")
    pattern = gguf.kv.get("dflash.attention.sliding_window_pattern")
    window = gguf.kv.get("dflash.attention.sliding_window")
    if pattern is None:
        pattern = [window is not None] * layers
    if (
        not isinstance(pattern, list)
        or len(pattern) != layers
        or any(type(flag) is not bool for flag in pattern)
    ):
        raise ValueError(f"{gguf.path}: dflash.attention.sliding_window_pattern is not per block")
    layer_types = [
        "sliding_attention" if flag else "full_attention" for flag in pattern
    ]
    result = {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "hidden_act": "silu",
        "attention_bias": False,
        "is_causal": causal,
        "hidden_size": _integer(gguf, "dflash.embedding_length"),
        "intermediate_size": _integer(gguf, "dflash.feed_forward_length"),
        "num_attention_heads": _integer(gguf, "dflash.attention.head_count"),
        "num_key_value_heads": _integer(gguf, "dflash.attention.head_count_kv"),
        "head_dim": head_dim,
        "num_hidden_layers": layers,
        "max_position_embeddings": _integer(gguf, "dflash.context_length"),
        "rms_norm_eps": _real(gguf, "dflash.attention.layer_norm_rms_epsilon"),
        "rope_parameters": {"rope_theta": _real(gguf, "dflash.rope.freq_base")},
        "layer_types": layer_types,
    }
    if "sliding_attention" in layer_types:
        result["sliding_window"] = _integer(gguf, "dflash.attention.sliding_window")
    mask = gguf.kv.get("dflash.mask_token_id", gguf.kv.get("tokenizer.ggml.mask_token_id"))
    if type(mask) is not int or mask < 0:
        raise ValueError(
            f"{gguf.path}: no dflash.mask_token_id or tokenizer.ggml.mask_token_id to read"
        )
    result["dflash_config"] = {
        "target_layer_ids": [index - 1 for index in taps],
        "mask_token_id": mask,
        "conv_kernel_size": _integer(gguf, "dflash.conv_kernel_size"),
        "conv_group_size": _integer(gguf, "dflash.conv_group_size"),
        "selector_rank": _integer(gguf, "dflash.selector_rank"),
        "selector_top_k": _integer(gguf, "dflash.selector_top_k"),
    }
    return result


def validate_dflash2(model, gguf: GGUFFile, config: dict) -> None:
    """Refuse a GGUF that is not this adapter's weights for this target's geometry."""

    expected = dflash2_tensor_map(config)
    stored = {tensor for _, tensor in expected.values()}
    missing = sorted(stored - set(gguf.tensors))[:5]
    extra = sorted(set(gguf.tensors) - stored)[:5]
    if missing or extra:
        raise ValueError(f"{gguf.path}: DFlash2 tensor set mismatch (missing {missing}, extra {extra})")
    for name, (_, tensor) in expected.items():
        parameter = model.parameters[name]
        info = gguf.info(tensor)
        shape = parameter.shape
        if info.shape != shape:
            raise ValueError(
                f"{gguf.path}: {tensor} is {info.shape}, but {name} is {shape}"
            )
        if info.type_id in GGUF_FORMATS_BY_TYPE:
            continue
        if info.type_name != "F32":
            raise ValueError(f"{gguf.path}: {tensor} is {info.type_name}, expected F32 or a block")
    end = max(info.offset + info.nbytes for info in gguf.tensors.values())
    if gguf.data_bytes_available < end:
        raise ValueError(f"{gguf.path}: the data section is truncated")
    gguf.require_finite_scales(stored)


def dflash2_sources(
    model, gguf: GGUFFile, config: dict
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    """Encoded block rows of every quantized adapter matrix, and its norms and base kernels."""

    encoded: dict[str, LogicalSource] = {}
    direct: dict[str, LogicalSource] = {}
    for name, (_, tensor) in dflash2_tensor_map(config).items():
        shape = model.parameters[name].shape
        if gguf.info(tensor).type_id in GGUF_FORMATS_BY_TYPE:
            encoded[name] = block_source(gguf, tensor, shape, rows())
            continue
        values = gguf.read_direct(tensor).astype(np.float32)
        direct[name] = _direct(values, torch.bfloat16, tensor)
    return encoded, direct


def mtp_block_of(gguf: GGUFFile) -> int:
    """The MTP block a GGUF declares: one predicted layer, appended after the text tower."""

    if gguf.kv.get(MTP_LAYERS_KEY) != 1:
        raise ValueError(
            f"{gguf.path}: {MTP_LAYERS_KEY} = {gguf.kv.get(MTP_LAYERS_KEY)!r}, expected 1"
        )
    count = gguf.kv.get(BLOCK_COUNT_KEY)
    if type(count) is not int or count < 1:
        raise ValueError(f"{gguf.path}: {BLOCK_COUNT_KEY} = {count!r}")
    return count - 1


def _store_encoded(recipe, encoded: dict[str, LogicalSource]) -> None:
    for name, source in encoded.items():
        recipe.assign(
            name,
            format=source.read_encoded(0, 1).format,
            layout="gguf_blocks_v1",
            method=import_encoded,
            source=source,
            activation_policy="AllowA8",
        )


def _store_direct(model, recipe, direct: dict[str, LogicalSource]) -> None:
    for name, source in direct.items():
        recipe.assign(
            name,
            format=model.parameters[name].direct_format,
            method=cast_direct,
            source=source,
        )


def apply_gguf_companions(model, recipe, companions) -> None:
    """Store every companion given as a GGUF file in the block formats it was published in.

    Applied after the selected recipe, so a complete assignment replaces whatever default format
    that recipe chose for the same matrices: the release's own blocks are the point.
    """

    dflash2 = companions.get("dflash2")
    if dflash2 is not None and "dflash2" in model.components:
        config = model.components["dflash2"]["config"]
        validate_dflash2(model, dflash2, config)
        encoded, direct = dflash2_sources(model, dflash2, config)
        _store_encoded(recipe, encoded)
        _store_direct(model, recipe, direct)
        # One physical copy per context K/V projection, both Uses keeping their own permission.
        for layer in range(config["num_hidden_layers"]):
            prefix = f"dflash2/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)
                name = prefix + "context_" + role
                for input_name in model.parameters[name].inputs:
                    recipe.use(name, input_name, activation_policy="AllowA8")
    mtp = companions.get("mtp")
    if mtp is not None and "mtp" in model.components:
        geometry = geometry_of(model.config)
        block = geometry.validate_mtp(mtp, mtp_block_of(mtp))
        encoded, direct = mtp_sources(geometry, mtp, block)
        _store_encoded(recipe, encoded)
        _store_direct(model, recipe, direct)


__all__ = [
    "DFLASH2_ARCHITECTURE",
    "apply_gguf_companions",
    "dflash2_config",
    "dflash2_sources",
    "dflash2_tensor_map",
    "full_attention_interval",
    "geometry_of",
    "mtp_block_of",
    "validate_dflash2",
]
