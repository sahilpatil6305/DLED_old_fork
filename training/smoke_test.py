#!/usr/bin/env python3
"""Offline, single-batch DLED runtime smoke test (synthetic inputs only)."""

import hashlib
import os
from pathlib import Path
import sys

import torch
import torch.nn as nn
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT))

# Import the same entrypoint/model/data packages used by the DLED runner.
import train as training_entrypoint  # noqa: F401,E402
import dataset as dataset_package  # noqa: F401,E402
import detectors as detector_package  # noqa: F401,E402
from clip import clip as clip_module  # noqa: E402
from detectors.dled import (  # noqa: E402
    DLED_Detector,
    _validate_lora_checkpoint_compatibility,
)
from loralib.layers import MultiheadAttentionLoRA  # noqa: E402
from loralib.utils import (  # noqa: E402
    INDEX_POSITIONS_TEXT,
    INDEX_POSITIONS_VISION,
    _get_lora_r,
    _get_lora_targets,
    apply_lora,
)


def _load_config():
    detector_path = ROOT / "training" / "config" / "detector" / "dled.yaml"
    train_config_path = ROOT / "training" / "config" / "train_config.yaml"
    with detector_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    with train_config_path.open(encoding="utf-8") as stream:
        train_config = yaml.safe_load(stream)
    if "label_dict" in config:
        train_config["label_dict"] = config["label_dict"]
    config.update(train_config)

    # Match the NOEFS arguments assembled by training/train.py and train.sh.
    config.update(
        {
            "train_dataset": ["DF40_FF_train_NOEFS"],
            "test_dataset": ["DF40_FF_test_NOFE_all"],
            "training_data_ratio": 1.0,
            "max_samples_per_class": 13504,
            "detect_novel": False,
            "num_classes": 3,
            "training_mode": "multi",
            "device_choice": "old",
            "avu_reg": True,
            "kl_reg": False,
        }
    )
    config.setdefault("backbone_config", {})["num_classes"] = 3
    return config


def _require_cached_clip(config):
    root = Path(
        os.environ.get(
            "DLED_PRETRAINED_ROOT",
            ROOT / "training" / "pretrained" / "clip",
        )
    ).expanduser().resolve()
    os.environ["DLED_PRETRAINED_ROOT"] = str(root)

    backbone = config["MODEL"]["BACKBONE"]["NAME"]
    if backbone not in clip_module._MODELS:
        raise RuntimeError(f"No bundled CLIP URL for configured backbone {backbone!r}")
    url = clip_module._MODELS[backbone]
    checkpoint = root / os.path.basename(url)
    expected_sha256 = url.rstrip("/").split("/")[-2]
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Cached CLIP checkpoint is required; refusing download: {checkpoint}"
        )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(
            f"Cached CLIP checksum mismatch for {checkpoint}: "
            f"expected {expected_sha256}, got {digest}"
        )
    return root, url, checkpoint


def _assert_no_download(root, url, checkpoint):
    """Replace CLIP's downloader during initialization with a cache-only guard."""
    expected_sha256 = url.rstrip("/").split("/")[-2]
    original_download = clip_module._download
    cache_root = root

    def cache_only_download(requested_url, root=None):
        if requested_url != url:
            raise RuntimeError(f"Unexpected CLIP download request: {requested_url}")
        candidate = Path(root or cache_root) / checkpoint.name
        if candidate.resolve() != checkpoint.resolve():
            raise RuntimeError(f"Unexpected CLIP cache path: {candidate}")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected_sha256:
            raise RuntimeError(f"Cached CLIP checkpoint changed during initialization: {candidate}")
        return str(candidate)

    clip_module._download = cache_only_download
    return original_download


def _blocks(encoder):
    transformer = getattr(encoder, "transformer", None)
    resblocks = getattr(transformer, "resblocks", None)
    if resblocks is None:
        raise AssertionError(f"No transformer.resblocks on {type(encoder).__name__}")
    return resblocks


def _verify_single_attention_replacement(device):
    """Regression for re-wrapping a newly installed MHA LoRA child."""
    class ResidualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(32, 4, dropout=0.2)

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.resblocks = nn.Sequential(ResidualBlock())

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = Transformer()

    config = {
        "TRAINER": {
            "RANK": 16,
            "ENABLE_LORA": ["q", "k", "v", "o"],
            "LORA_ALPHA": 1,
        }
    }
    encoder = Encoder().to(device)
    source_attention = encoder.transformer.resblocks[0].attn
    source_state = {
        name: tensor.detach().clone()
        for name, tensor in source_attention.state_dict().items()
    }

    layers = apply_lora(config, encoder, [0], dtype=source_attention.in_proj_weight.dtype)
    if len(layers) != 1:
        raise AssertionError(f"Expected exactly one replacement, got {len(layers)}")
    replacement = encoder.transformer.resblocks[0].attn
    if not isinstance(replacement, MultiheadAttentionLoRA):
        raise AssertionError(f"Unexpected replacement type: {type(replacement).__name__}")
    if sum(isinstance(module, MultiheadAttentionLoRA) for module in encoder.modules()) != 1:
        raise AssertionError("One source attention was wrapped more than once")

    for name, tensor in source_state.items():
        if not torch.equal(replacement.state_dict()[name], tensor):
            raise AssertionError(f"Base attention state changed during replacement: {name}")
    for projection in ("q", "k", "v", "o"):
        a = getattr(replacement, f"lora_{projection}_A")
        b = getattr(replacement, f"lora_{projection}_B")
        if tuple(a.shape) != (16, 32) or tuple(b.shape) != (32, 16):
            raise AssertionError(
                f"{projection} adapter shapes are {tuple(a.shape)}, {tuple(b.shape)}"
            )
        if not a.requires_grad or not b.requires_grad:
            raise AssertionError(f"{projection} adapter parameters must be trainable")
        if torch.count_nonzero(b).item() != 0:
            raise AssertionError(f"{projection} B factor must start at zero")

    for name, parameter in replacement.named_parameters():
        if "lora_" not in name and parameter.requires_grad:
            raise AssertionError(f"Base attention parameter unexpectedly trainable: {name}")

    reference = nn.MultiheadAttention(32, 4, dropout=0.2).to(device)
    reference.load_state_dict(source_state, strict=True)
    reference.eval()
    replacement.eval()
    sequence_length, batch_size = 4, 2
    query = torch.randn(sequence_length, batch_size, 32, device=device)
    key_padding_mask = torch.tensor(
        [[False, False, False, True], [False, True, False, False]],
        dtype=torch.bool,
        device=device,
    )
    attn_mask = torch.triu(
        torch.full((sequence_length, sequence_length), float("-inf"), device=device), 1
    )
    with torch.no_grad():
        expected, expected_weights = reference(
            query, query, query, key_padding_mask=key_padding_mask,
            need_weights=True, attn_mask=attn_mask,
        )
        actual, actual_weights = replacement(
            query, query, query, key_padding_mask=key_padding_mask,
            need_weights=True, attn_mask=attn_mask,
        )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_weights, expected_weights)

    try:
        apply_lora(config, encoder, [0], dtype=source_attention.in_proj_weight.dtype)
    except ValueError as error:
        if "already contains" not in str(error):
            raise AssertionError(f"Repeated injection did not give a clear error: {error}") from error
    else:
        raise AssertionError("Repeated apply_lora unexpectedly wrapped an adapted block")


def _verify_zero_delta_parity(replacements, device):
    """Compare each zero-adapter replacement against its frozen base MHA."""
    mha_backend = getattr(torch.backends, "mha", None)
    set_fastpath = getattr(mha_backend, "set_fastpath_enabled", None)
    get_fastpath = getattr(mha_backend, "get_fastpath_enabled", None)
    prior_fastpath = get_fastpath() if get_fastpath is not None else None
    prior_training_modes = [(attention, attention.training) for _, _, attention in replacements]
    if set_fastpath is not None:
        set_fastpath(False)

    try:
        with torch.no_grad():
            for name, block_index, attention in replacements:
                base_state = {
                    key: value.detach().clone()
                    for key, value in attention.state_dict().items()
                    if "lora_" not in key
                }
                reference = nn.MultiheadAttention(
                    attention.embed_dim,
                    attention.num_heads,
                    dropout=attention.dropout,
                    bias=attention.in_proj_bias is not None,
                    add_bias_kv=attention.bias_k is not None,
                    add_zero_attn=attention.add_zero_attn,
                    kdim=attention.kdim,
                    vdim=attention.vdim,
                    batch_first=attention.batch_first,
                ).to(device=device, dtype=attention.in_proj_weight.dtype)
                reference.load_state_dict(base_state, strict=True)
                reference.eval()
                attention.eval()

                length, batch_size = 3, 2
                if attention.batch_first:
                    shape = (batch_size, length, attention.embed_dim)
                else:
                    shape = (length, batch_size, attention.embed_dim)
                query = torch.randn(shape, device=device, dtype=attention.in_proj_weight.dtype)
                key_padding_mask = torch.zeros((batch_size, length), device=device, dtype=torch.bool)
                attn_mask = None
                if name.endswith("text"):
                    attn_mask = torch.triu(
                        torch.ones(length, length, device=device, dtype=torch.bool),
                        diagonal=1,
                    )

                expected = reference(
                    query,
                    query,
                    query,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                    attn_mask=attn_mask,
                )[0]
                actual = attention(
                    query,
                    query,
                    query,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                    attn_mask=attn_mask,
                )[0]
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=2e-3,
                    atol=2e-3,
                    msg=f"Zero-delta attention parity failed for {name} block {block_index}",
                )
                del reference, query, expected, actual
    finally:
        for attention, training_mode in prior_training_modes:
            attention.train(training_mode)
        if set_fastpath is not None and prior_fastpath is not None:
            set_fastpath(prior_fastpath)


def _verify_legacy_checkpoint_rejection(model):
    """Exercise the guard with a synthetic rank-4/QV checkpoint state."""
    current_state = model.state_dict()
    legacy_state = {}
    for name, tensor in current_state.items():
        if ".lora_q_" in name or ".lora_v_" in name:
            if name.endswith("_A"):
                legacy_state[name] = tensor[:4].clone()
            else:
                legacy_state[name] = tensor[:, :4].clone()

    try:
        _validate_lora_checkpoint_compatibility(current_state, legacy_state)
    except RuntimeError as error:
        message = str(error)
        if "Missing keys" not in message or "shape mismatches" not in message:
            raise AssertionError(f"Checkpoint guard omitted mismatch details: {message}") from error
    else:
        raise AssertionError("Checkpoint guard accepted a synthetic rank-4/QV checkpoint")


def _verify_lora(detector, config, device):
    model = detector.model
    expected_rank = _get_lora_r(config)
    expected_targets = set(_get_lora_targets(config))
    if expected_rank != 16:
        raise AssertionError(f"NOEFS config contract requires LoRA rank 16, got {expected_rank}")
    if expected_targets != {"q", "k", "v", "o"}:
        raise AssertionError(f"NOEFS config contract requires Q/K/V/O, got {sorted(expected_targets)}")

    text_indices = set(INDEX_POSITIONS_TEXT[config["TRAINER"]["POSITION"]])
    visual_indices = set(
        INDEX_POSITIONS_VISION[config["MODEL"]["BACKBONE"]["NAME"]][
            config["TRAINER"]["POSITION"]
        ]
    )
    encoders = {
        "spatial_text": (model.spatial_text_encoder, text_indices),
        "spatial_vision": (model.spatial_image_encoder, visual_indices),
        "frequency_text": (model.frequency_text_encoder, text_indices),
        "frequency_vision": (model.frequency_image_encoder, visual_indices),
    }
    replacements = []
    for name, (encoder, expected_indices) in encoders.items():
        blocks = _blocks(encoder)
        actual_indices = {
            index
            for index, block in enumerate(blocks)
            if isinstance(getattr(block, "attn", None), MultiheadAttentionLoRA)
        }
        if actual_indices != expected_indices:
            raise AssertionError(
                f"{name}: expected LoRA blocks {sorted(expected_indices)}, "
                f"found {sorted(actual_indices)}"
            )
        if len(actual_indices) != 12:
            raise AssertionError(f"{name}: expected 12 replacements, found {len(actual_indices)}")
        replacements.extend(
            (name, index, block.attn)
            for index, block in enumerate(blocks)
            if isinstance(getattr(block, "attn", None), MultiheadAttentionLoRA)
        )

    if len(replacements) != 48:
        raise AssertionError(f"Expected 48 ViT-B/16 attention replacements, found {len(replacements)}")

    expected_dtype = model.dtype
    expected_adapter_numel = 0
    for name, block_index, attention in replacements:
        actual_targets = set(attention.enable_lora)
        if attention.r != expected_rank or actual_targets != expected_targets:
            raise AssertionError(
                f"{name} block {block_index}: expected rank/targets "
                f"{expected_rank}/{sorted(expected_targets)}, got "
                f"{attention.r}/{sorted(actual_targets)}"
            )
        if attention.lora_alpha != config["TRAINER"]["LORA_ALPHA"]:
            raise AssertionError(f"{name} block {block_index}: LORA_ALPHA was not propagated")
        if abs(attention.scaling - attention.lora_alpha / (attention.r ** 0.5)) > 1e-12:
            raise AssertionError(f"{name} block {block_index}: scaling convention changed")

        for projection in ("q", "k", "v", "o"):
            has_adapter = hasattr(attention, f"lora_{projection}_A")
            if has_adapter != (projection in expected_targets):
                raise AssertionError(
                    f"{name} block {block_index}: adapter presence disagrees for {projection}"
                )
            if not has_adapter:
                continue
            expected_shapes = ((expected_rank, attention.embed_dim), (attention.embed_dim, expected_rank))
            actual_shapes = (
                tuple(getattr(attention, f"lora_{projection}_A").shape),
                tuple(getattr(attention, f"lora_{projection}_B").shape),
            )
            if actual_shapes != expected_shapes:
                raise AssertionError(
                    f"{name} block {block_index} {projection}: adapter shapes "
                    f"{actual_shapes}, expected {expected_shapes}"
                )
            if torch.count_nonzero(getattr(attention, f"lora_{projection}_B")).item() != 0:
                raise AssertionError(
                    f"{name} block {block_index} {projection}: expected zero-initialized B"
                )
            expected_adapter_numel += 2 * expected_rank * attention.embed_dim

        for parameter_name, parameter in attention.named_parameters():
            if parameter.device != device:
                raise AssertionError(
                    f"{name} block {block_index}.{parameter_name} is on {parameter.device}, "
                    f"expected {device}"
                )
            if parameter.dtype != expected_dtype:
                raise AssertionError(
                    f"{name} block {block_index}.{parameter_name} has {parameter.dtype}, "
                    f"expected {expected_dtype}"
                )
            if "lora_" not in parameter_name and parameter.requires_grad:
                raise AssertionError(
                    f"Frozen base attention parameter is trainable: {name}.{parameter_name}"
                )

    actual_adapter_numel = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    )
    if actual_adapter_numel != expected_adapter_numel:
        raise AssertionError(
            f"LoRA trainable parameter count {actual_adapter_numel}; "
            f"expected {expected_adapter_numel} from configured rank/targets"
        )

    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    unexpected_trainable = {
        name
        for name in trainable
        if "spatial_prompt_learner" not in name
        and "frequency_prompt_learner" not in name
        and "lora_" not in name
    }
    if unexpected_trainable:
        raise AssertionError(f"Unexpected trainable parameters: {sorted(unexpected_trainable)}")
    if not any("spatial_prompt_learner.ctx" in name for name in trainable):
        raise AssertionError("Spatial prompt context is not trainable")
    if not any("frequency_prompt_learner.ctx" in name for name in trainable):
        raise AssertionError("Frequency prompt context is not trainable")
    if not any("lora_" in name for name in trainable):
        raise AssertionError("No LoRA parameters are trainable")

    frozen_base = [
        name
        for name, parameter in model.named_parameters()
        if "lora_" not in name
        and "spatial_prompt_learner" not in name
        and "frequency_prompt_learner" not in name
        and parameter.requires_grad
    ]
    if frozen_base:
        raise AssertionError(f"Frozen CLIP base parameters are trainable: {frozen_base[:20]}")

    _verify_zero_delta_parity(replacements, device)
    _verify_legacy_checkpoint_rejection(model)

    return trainable, actual_adapter_numel


def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.cuda.current_device())
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    print(f"Running DLED runtime smoke test on {device}.")
    _verify_single_attention_replacement(device)
    print(f"Focused attention replacement regression passed on {device}.")

    config = _load_config()
    if device.type == "cpu":
        # CLIP's configured fp16 path is intended for CUDA; use the detector's
        # existing fp32 option on CPU without changing model structure.
        config["TRAINER"]["PREC"] = "fp32"
    cache_root, clip_url, checkpoint = _require_cached_clip(config)
    print(f"Using checksum-verified cached CLIP checkpoint: {checkpoint}")

    original_download = _assert_no_download(cache_root, clip_url, checkpoint)
    try:
        detector = DLED_Detector(config)
    finally:
        clip_module._download = original_download

    if detector.device != device:
        raise AssertionError(f"Detector selected {detector.device}, expected {device}")

    detector.model.train()
    trainable, adapter_numel = _verify_lora(detector, config, device)
    labels = torch.tensor([0, 1], dtype=torch.long, device=device)
    batch = {
        "image": torch.randn(2, 3, 224, 224, device=device),
        "freq": torch.randn(2, 3, 224, 224, device=device),
        "label": labels,
    }

    predictions = detector(batch, "fusion")
    for key in ("spatial_pred_alpha", "frequency_pred_alpha", "pred_alpha"):
        if predictions[key].shape != (2, 3):
            raise AssertionError(f"Unexpected {key} shape: {tuple(predictions[key].shape)}")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if not torch.isfinite(value).all():
                raise AssertionError(f"Non-finite forward output: {key}")

    predictions["epoch"] = 0
    predictions["total_epoch"] = 1
    losses = detector.get_losses(batch, predictions, mode="train")
    loss = losses["overall"]
    if not torch.isfinite(loss):
        raise AssertionError(f"Non-finite smoke-test loss: {loss}")
    loss.backward()

    missing_gradients = [
        name
        for name, parameter in detector.model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing_gradients:
        raise AssertionError(f"Trainable parameters without gradients: {missing_gradients[:20]}")
    nonfinite_gradients = [
        name
        for name, parameter in detector.model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if nonfinite_gradients:
        raise AssertionError(f"Non-finite gradients: {nonfinite_gradients[:20]}")
    for projection in ("q", "k", "v", "o"):
        b_gradients = [
            parameter.grad
            for name, parameter in detector.model.named_parameters()
            if f"lora_{projection}_B" in name and parameter.grad is not None
        ]
        if not b_gradients or not any(torch.count_nonzero(grad).item() for grad in b_gradients):
            raise AssertionError(f"No nonzero first-backward gradient reached {projection} B factors")
    frozen_with_gradients = [
        name
        for name, parameter in detector.model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if frozen_with_gradients:
        raise AssertionError(f"Frozen parameters received gradients: {frozen_with_gradients[:20]}")

    # Exercise an actual optimizer step on the same trainable model parameters.
    optimizer = torch.optim.SGD(
        [parameter for parameter in detector.model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    before = next(
        parameter.detach().clone()
        for name, parameter in detector.model.named_parameters()
        if "lora_q_B" in name and parameter.requires_grad
    )
    optimizer.step()
    after = next(
        parameter.detach()
        for name, parameter in detector.model.named_parameters()
        if "lora_q_B" in name and parameter.requires_grad
    )
    if not torch.isfinite(after).all() or torch.equal(before, after):
        raise AssertionError("Optimizer step did not make a finite LoRA parameter update")

    print(
        "RUNTIME SMOKE PASSED: imports, cached CLIP, DLED init, 48 attention "
        f"replacements, rank=16 Q/K/V/O, forward, backward; "
        f"{len(trainable)} trainable parameter tensors ({adapter_numel} LoRA scalars) "
        "received finite gradients and an SGD step updated a LoRA parameter. "
        "Zero-initialized A gradients may be zero on the first pass; B gradients "
        "must demonstrate adapter connectivity. No epoch or real-data training was run."
    )


if __name__ == "__main__":
    main()
