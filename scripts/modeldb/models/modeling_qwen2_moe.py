from ...utils.Injector import Injector
from ...utils.Namespace import UniConfig
from ...utils.ContextList import ContextList
from ...utils.Timer import Timer
from ...utils.pq_utils import DynamicPQCache
from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeForCausalLM,
    Qwen2MoeSdpaAttention,
    logger,
    apply_rotary_pos_emb,
    repeat_kv,
    Cache,
)

import torch
from typing import Optional, Tuple
from .ModelContext import ModelContext

def save_forward(
    self: Qwen2MoeSdpaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **qwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "Qwen2MoeModel is using Qwen2MoeSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    cos, sin = self.rotary_emb(value_states, position_ids)


    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Save key and value states to reservoir
    bs, nh, _, head_size = key_states.shape
    config = UniConfig()
    
    for b in range(bs):
        for h in range(nh):
            if config.merged_training is True:
                config.key_reservoir.batch_add(key_states[b, h].view(-1, head_size).detach().cpu())
                config.value_reservoir.batch_add(value_states[b, h].view(-1, head_size).detach().cpu())
            else:
                config.key_reservoir[self.layer_idx].batch_add(key_states[b, h].view(-1, head_size).detach().cpu())
                config.value_reservoir[self.layer_idx].batch_add(value_states[b, h].view(-1, head_size).detach().cpu())


    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal=self.is_causal and attention_mask is None and q_len > 1,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

def pq_forward(
    self: Qwen2MoeSdpaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    distort_recent = UniConfig().distort_recent
    with Timer("LlamaSdpaAttention.forward"):
        with Timer("qkv_proj"):
            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            torch.cuda.synchronize()

        with Timer("rotary_emb"):
            # cos, sin = self.rotary_emb(value_states, seq_len=key_states.shape[-2])
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            torch.cuda.synchronize()

        with Timer("update_cache"):
            if use_cache:
                assert DynamicPQCache.has_instance(), 'Cache must be set before using it.'
                cache = DynamicPQCache()
                key_states, value_states = cache.update(key_states, value_states, self.layer_idx, distort_recent=distort_recent)
            torch.cuda.synchronize()
        
        with Timer("repeat_kv"):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            torch.cuda.synchronize()

        with Timer("causal_mask"):
            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
            torch.cuda.synchronize()

        with Timer("contiguous"):
            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and causal_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()
            torch.cuda.synchronize()

        with Timer("sdpa"):
            # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
            # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
            is_causal = True if causal_mask is None and q_len > 1 else False

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            torch.cuda.synchronize()

        with Timer("contiguous"):
            attn_output = attn_output.transpose(1, 2).contiguous()
            torch.cuda.synchronize()

        with Timer("o_proj"):
            attn_output = attn_output.view(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            torch.cuda.synchronize()

        return attn_output, None, past_key_value


model_context = ModelContext(
    init_context=ContextList([]),
    baseline_context=ContextList([]),
    sampling_context=ContextList([
        Injector(Qwen2MoeSdpaAttention, 'forward', save_forward),
    ]),
    evaluation_context=ContextList([
        Injector(Qwen2MoeSdpaAttention, 'forward', pq_forward),
    ]),
)



