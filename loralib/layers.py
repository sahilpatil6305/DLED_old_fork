"""
loralib/layers.py
=================
LoRA layer implementations compatible with CLIP's MultiheadAttention.
Based on CLIP-LoRA (MaxZanella/CLIP-LoRA), adapted for the DLED project.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Base LoRA mixin
# ---------------------------------------------------------------------------

class LoRALayer:
    def __init__(self, r: int, lora_alpha: int, fan_in_fan_out: bool = False,
                 dropout_rate: float = 0.0):
        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout_rate
        if r > 0:
            self.scaling = lora_alpha / math.sqrt(r)
        self.merged = False
        self.fan_in_fan_out = fan_in_fan_out
        self.params_with_lora = {}

    def register_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            w = getattr(self, param_name)
            assert len(w.size()) == 2
            self.register_parameter(
                f'{lora_name}_lora_A',
                nn.Parameter(w.new_zeros((self.r, w.size(1))))
            )
            self.register_parameter(
                f'{lora_name}_lora_B',
                nn.Parameter(w.new_zeros((w.size(0), self.r)))
            )
            w.requires_grad = False

    def init_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            if hasattr(self, f'{lora_name}_lora_A'):
                nn.init.kaiming_uniform_(
                    getattr(self, f'{lora_name}_lora_A'), a=math.sqrt(5)
                )
                nn.init.zeros_(getattr(self, f'{lora_name}_lora_B'))

    def transpose(self, w: torch.Tensor):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def merge_BA(self, param_name: str):
        lora_name = self.params_with_lora[param_name]
        A = getattr(self, f'{lora_name}_lora_A')
        B = getattr(self, f'{lora_name}_lora_B')
        return self.transpose((B @ A).view(getattr(self, param_name).shape))

    def merge_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            p = getattr(self, param_name)
            p_new = p.detach() + self.merge_BA(param_name) * self.scaling
            delattr(self, param_name)
            setattr(self, param_name, p_new)

    def add_lora_data(self):
        for param_name in self.params_with_lora:
            getattr(self, param_name).data += self.merge_BA(param_name) * self.scaling

    def sub_lora_data(self):
        for param_name in self.params_with_lora:
            getattr(self, param_name).data -= self.merge_BA(param_name) * self.scaling

    def lora_train(self, mode: bool = True):
        if mode:
            if self.merged and self.r > 0:
                self.sub_lora_data()
            self.merged = False
        else:
            if not self.merged and self.r > 0:
                self.add_lora_data()
            self.merged = True


# ---------------------------------------------------------------------------
# MultiheadAttention with independently selected projection adapters
# ---------------------------------------------------------------------------

class MultiheadAttentionLoRA(nn.MultiheadAttention, LoRALayer):
    """
    Drop-in replacement for CLIP's packed-QKV nn.MultiheadAttention.

    The adapter targets are supplied explicitly. Q/K/V updates are applied to
    their respective packed in_proj_weight slices, while O updates the actual
    out_proj weight. Base parameters are copied and frozen.
    """

    def __init__(self, original_attn: nn.MultiheadAttention, r: int,
                 lora_alpha: int = 1, dropout_rate: float = 0.0,
                 dtype=None, enable_lora=None):
        source_weight = original_attn.in_proj_weight
        if source_weight is None or not original_attn._qkv_same_embed_dim:
            raise ValueError(
                "MultiheadAttentionLoRA requires CLIP-style packed Q/K/V in_proj_weight"
            )
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r!r}")
        if enable_lora is None:
            raise ValueError("enable_lora targets must be provided explicitly")
        supported_targets = {'q', 'k', 'v', 'o'}
        targets = tuple(enable_lora)
        if (
            not targets
            or any(not isinstance(target, str) for target in targets)
            or len(set(targets)) != len(targets)
            or set(targets) - supported_targets
        ):
            raise ValueError(f"Invalid LoRA projection targets: {targets!r}")

        embed_dim = original_attn.embed_dim
        num_heads = original_attn.num_heads
        dropout = original_attn.dropout
        bias = original_attn.in_proj_bias is not None
        kdim = original_attn.kdim
        vdim = original_attn.vdim
        source_state = {
            name: tensor.detach().clone()
            for name, tensor in original_attn.state_dict().items()
        }

        nn.MultiheadAttention.__init__(
            self, embed_dim, num_heads, dropout=dropout, bias=bias,
            add_bias_kv=original_attn.bias_k is not None,
            add_zero_attn=original_attn.add_zero_attn,
            kdim=kdim, vdim=vdim, batch_first=original_attn.batch_first,
        )
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha,
                           dropout_rate=dropout_rate)

        # Strictly preserve the complete source attention state.
        self.load_state_dict(source_state, strict=True)

        if dtype is not None and dtype != source_weight.dtype:
            raise ValueError(
                f"Requested LoRA dtype {dtype} does not match source attention dtype "
                f"{source_weight.dtype}"
            )
        self.to(device=source_weight.device, dtype=source_weight.dtype)
        self.enable_lora = targets

        copied_state = self.state_dict()
        for name, source_tensor in source_state.items():
            if not torch.equal(copied_state[name], source_tensor.to(copied_state[name].device)):
                raise RuntimeError(f"Replacement changed source attention parameter {name}")

        # Freeze all parameters initially
        for p in self.parameters():
            p.requires_grad = False

        dim = embed_dim
        for projection in targets:
            base_weight = self.out_proj.weight if projection == 'o' else self.in_proj_weight
            setattr(
                self,
                f'lora_{projection}_A',
                nn.Parameter(base_weight.new_empty((r, dim))),
            )
            setattr(
                self,
                f'lora_{projection}_B',
                nn.Parameter(base_weight.new_zeros((dim, r))),
            )
            # Local convention: Kaiming-uniform A, zero B; initial delta is zero.
            nn.init.kaiming_uniform_(getattr(self, f'lora_{projection}_A'), a=math.sqrt(5))
            nn.init.zeros_(getattr(self, f'lora_{projection}_B'))

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (tensor.transpose(1, 0) for tensor in (query, key))
                    value = key
            else:
                query, key, value = (
                    tensor.transpose(1, 0) for tensor in (query, key, value)
                )

        # Add Q/K/V deltas to packed rows and O delta to the output projection.
        if self.r > 0 and not self.merged and any(
            projection in self.enable_lora for projection in ('q', 'k', 'v')
        ):
            dim = self.embed_dim
            weight = self.in_proj_weight.clone()
            for index, projection in enumerate(('q', 'k', 'v')):
                if projection in self.enable_lora:
                    delta = (
                        getattr(self, f'lora_{projection}_B')
                        @ getattr(self, f'lora_{projection}_A')
                    ) * self.scaling
                    weight[index * dim:(index + 1) * dim] = (
                        weight[index * dim:(index + 1) * dim] + delta
                    )
        else:
            weight = self.in_proj_weight

        out_proj_weight = self.out_proj.weight
        if self.r > 0 and not self.merged and 'o' in self.enable_lora:
            delta_o = (self.lora_o_B @ self.lora_o_A) * self.scaling
            out_proj_weight = out_proj_weight + delta_o

        output, attention_weights = F.multi_head_attention_forward(
            query, key, value,
            self.embed_dim, self.num_heads,
            weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, out_proj_weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )
        if self.batch_first and is_batched:
            output = output.transpose(1, 0)
        return output, attention_weights


# ---------------------------------------------------------------------------
# Compatibility alias retained for existing imports.
# ---------------------------------------------------------------------------
PlainMultiheadAttentionLoRA = MultiheadAttentionLoRA
