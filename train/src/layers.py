import inspect
import math
from typing import Callable, List, Optional, Tuple, Union
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import RMSNorm


class LoRALinearLayer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int = 4,
            network_alpha: Optional[float] = None,
            device: Optional[Union[torch.device, str]] = None,
            dtype: Optional[torch.dtype] = None,
            cond_width=512,
            cond_height=512,
            number=0,
            n_loras=1
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

        self.cond_height = cond_height
        self.cond_width = cond_width
        self.number = number
        self.n_loras = n_loras

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        batch_size = hidden_states.shape[0]
        cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64
        block_size = hidden_states.shape[1] - cond_size * self.n_loras
        shape = (batch_size, hidden_states.shape[1], 3072)
        mask = torch.ones(shape, device=hidden_states.device, dtype=dtype)
        mask[:, :block_size + self.number * cond_size, :] = 0
        mask[:, block_size + (self.number + 1) * cond_size:, :] = 0
        hidden_states = mask * hidden_states

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class MultiSingleStreamBlockLoraProcessor(nn.Module):
    def __init__(
            self,
            dim: int,
            ranks=[],
            lora_weights=[],
            network_alphas=[],
            device=None,
            dtype=None,
            cond_width=512,
            cond_height=512,
            n_loras=1,
            scale=1.0,
            cross_attention_dim=None,
            heads=24
    ):
        super().__init__()
        self.n_loras = n_loras
        self.cond_width = cond_width
        self.cond_height = cond_height

        self.q_loras = nn.ModuleList([
            LoRALinearLayer(
                dim, dim, ranks[i], network_alphas[i],
                device=device, dtype=dtype,
                cond_width=cond_width, cond_height=cond_height,
                number=i, n_loras=n_loras
            ) for i in range(n_loras)
        ])
        self.k_loras = nn.ModuleList([
            LoRALinearLayer(
                dim, dim, ranks[i], network_alphas[i],
                device=device, dtype=dtype,
                cond_width=cond_width, cond_height=cond_height,
                number=i, n_loras=n_loras
            ) for i in range(n_loras)
        ])
        self.v_loras = nn.ModuleList([
            LoRALinearLayer(
                dim, dim, ranks[i], network_alphas[i],
                device=device, dtype=dtype,
                cond_width=cond_width, cond_height=cond_height,
                number=i, n_loras=n_loras
            ) for i in range(n_loras)
        ])
        self.lora_weights = lora_weights

        self.hidden_size = dim
        self.scale = scale
        self.cross_attention_dim = cross_attention_dim or dim
        self.heads = heads
        self.head_dim = dim // heads

        self.to_k_ip = nn.Linear(
            self.cross_attention_dim, dim, bias=False,
            device=device, dtype=dtype
        )
        self.to_v_ip = nn.Linear(
            self.cross_attention_dim, dim, bias=False,
            device=device, dtype=dtype
        )
        self.norm_added_k = RMSNorm(128, eps=1e-5, elementwise_affine=False)

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            use_cond=False,
            image_emb: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0]
        cond_size = (self.cond_width // 8) * (self.cond_height // 8) * 16 // 64
        block_size = hidden_states.shape[1] - cond_size * self.n_loras

        original_query = attn.to_q(hidden_states[:, :block_size, :])
        original_query = original_query.view(
            batch_size, -1, self.heads, self.head_dim
        ).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        for i in range(self.n_loras):
            query = query + self.lora_weights[i] * self.q_loras[i](hidden_states)
            key = key + self.lora_weights[i] * self.k_loras[i](hidden_states)
            value = value + self.lora_weights[i] * self.v_loras[i](hidden_states)

        head_dim = self.hidden_size // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        ip_hidden_states = None
        if image_emb is not None:
            ip_key = self.to_k_ip(image_emb)
            ip_value = self.to_v_ip(image_emb)

            ip_key = ip_key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            ip_key = self.norm_added_k(ip_key)

            ip_hidden_states = F.scaled_dot_product_attention(
                original_query,
                ip_key,
                ip_value,
                dropout_p=0.0,
                is_causal=False
            )

            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
                batch_size, -1, self.heads * head_dim
            ).to(query.dtype)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        scaled_seq_len = query.shape[2]

        mask = torch.ones((scaled_seq_len, scaled_seq_len), device=hidden_states.device)
        mask[:block_size, :] = 0
        for i in range(self.n_loras):
            start = i * cond_size + block_size
            end = (i + 1) * cond_size + block_size
            mask[start:end, start:end] = 0
        mask = mask * -1e20
        mask = mask.to(query.dtype)

        hidden_states_attn = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False
        )
        hidden_states_attn = hidden_states_attn.transpose(1, 2).reshape(
            batch_size, -1, self.heads * head_dim
        ).to(query.dtype)

        cond_hidden_states = hidden_states_attn[:, block_size:, :]
        main_hidden_states = hidden_states_attn[:, :block_size, :]

        if ip_hidden_states is not None:
            main_hidden_states = main_hidden_states + self.scale * ip_hidden_states

        if use_cond:
            return main_hidden_states, cond_hidden_states
        else:
            return main_hidden_states


class MultiDoubleStreamBlockLoraProcessor(nn.Module):
    def __init__(
            self,
            dim: int,
            ranks=[],
            lora_weights=[],
            network_alphas=[],
            device=None,
            dtype=None,
            cond_width=512,
            cond_height=512,
            n_loras=1,
            scale=1.0,
            cross_attention_dim=None,
            heads=24
    ):
        super().__init__()
        self.n_loras = n_loras
        self.cond_width = cond_width
        self.cond_height = cond_height

        self.q_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i], network_alphas[i],
                            device=device, dtype=dtype,
                            cond_width=cond_width, cond_height=cond_height,
                            number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.k_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i], network_alphas[i],
                            device=device, dtype=dtype,
                            cond_width=cond_width, cond_height=cond_height,
                            number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.v_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i], network_alphas[i],
                            device=device, dtype=dtype,
                            cond_width=cond_width, cond_height=cond_height,
                            number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.proj_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i], network_alphas[i],
                            device=device, dtype=dtype,
                            cond_width=cond_width, cond_height=cond_height,
                            number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.lora_weights = lora_weights

        self.hidden_size = dim
        self.scale = scale
        self.cross_attention_dim = cross_attention_dim or dim
        self.heads = heads
        self.head_dim = dim // heads

        self.to_k_ip = nn.Linear(
            self.cross_attention_dim, dim, bias=False,
            device=device, dtype=dtype
        )
        self.to_v_ip = nn.Linear(
            self.cross_attention_dim, dim, bias=False,
            device=device, dtype=dtype
        )
        self.norm_added_k = RMSNorm(128, eps=1e-5, elementwise_affine=False)

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            use_cond=False,
            image_emb: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0]
        cond_size = (self.cond_width // 8) * (self.cond_height // 8) * 16 // 64
        block_size = hidden_states.shape[1] - cond_size * self.n_loras

        original_query = attn.to_q(hidden_states[:, :block_size, :])
        original_query = original_query.view(
            batch_size, -1, self.heads, self.head_dim
        ).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        for i in range(self.n_loras):
            query = query + self.lora_weights[i] * self.q_loras[i](hidden_states)
            key = key + self.lora_weights[i] * self.k_loras[i](hidden_states)
            value = value + self.lora_weights[i] * self.v_loras[i](hidden_states)

        head_dim = self.hidden_size // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        ip_hidden_states = None
        if image_emb is not None:
            ip_key = self.to_k_ip(image_emb)
            ip_value = self.to_v_ip(image_emb)

            ip_key = ip_key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            ip_key = self.norm_added_k(ip_key)

            ip_hidden_states = F.scaled_dot_product_attention(
                original_query,
                ip_key,
                ip_value,
                dropout_p=0.0,
                is_causal=False
            )

            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
                batch_size, -1, self.heads * head_dim
            ).to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.view(
                batch_size, -1, self.heads, head_dim
            ).transpose(1, 2)
            encoder_key = encoder_key.view(
                batch_size, -1, self.heads, head_dim
            ).transpose(1, 2)
            encoder_value = encoder_value.view(
                batch_size, -1, self.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=2)
            key = torch.cat([encoder_key, key], dim=2)
            value = torch.cat([encoder_value, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        scaled_seq_len = query.shape[2]

        mask = torch.ones((scaled_seq_len, scaled_seq_len), device=hidden_states.device)
        mask[:scaled_seq_len - cond_size * self.n_loras, :] = 0
        for i in range(self.n_loras):
            start = i * cond_size + (scaled_seq_len - cond_size * self.n_loras)
            end = (i + 1) * cond_size + (scaled_seq_len - cond_size * self.n_loras)
            mask[start:end, start:end] = 0
        mask = mask * -1e20
        mask = mask.to(query.dtype)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, self.heads * head_dim
        ).to(query.dtype)

        encoder_hidden_states_out, hidden_states = (
            hidden_states[:, :encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1]:]
        )

        hidden_states = attn.to_out[0](hidden_states)
        for i in range(self.n_loras):
            hidden_states = hidden_states + self.lora_weights[i] * self.proj_loras[i](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)

        cond_hidden_states = hidden_states[:, block_size:, :]
        hidden_states = hidden_states[:, :block_size, :]

        if ip_hidden_states is not None:
            hidden_states = hidden_states + self.scale * ip_hidden_states

        if use_cond:
            return hidden_states, encoder_hidden_states_out, cond_hidden_states
        else:
            return encoder_hidden_states_out, hidden_states