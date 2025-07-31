from ...utils.Injector import Injector
from ...utils.Namespace import UniConfig
from ...utils.ContextList import ContextList
from ...utils.Timer import Timer
from ...utils.pq_utils import DynamicPQCache
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2LMHeadModel,
    GPT2Attention,
    GPT2SdpaAttention,
    logger,
)

import torch
from typing import Optional, Tuple
from .ModelContext import ModelContext
from typing import Union


def save_forward_legacy(
    self: GPT2Attention,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
    if encoder_hidden_states is not None:
        if not hasattr(self, "q_attn"):
            raise ValueError(
                "If class is used as cross attention, the weights `q_attn` have to be defined. "
                "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
            )

        query = self.q_attn(hidden_states)
        key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
        attention_mask = encoder_attention_mask
    else:
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

    query = self._split_heads(query, self.num_heads, self.head_dim)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    value = self._split_heads(value, self.num_heads, self.head_dim)

    # save to disk
    bs, nh, _, head_size = key.shape
    config = UniConfig()
    
    for b in range(bs):
        for h in range(nh):
            if config.merged_training is True:
                config.key_reservoir.batch_add(key[b, h].view(-1, head_size).detach().cpu())
                config.value_reservoir.batch_add(value[b, h].view(-1, head_size).detach().cpu())
            else:
                config.key_reservoir[self.layer_idx].batch_add(key[b, h].view(-1, head_size).detach().cpu())
                config.value_reservoir[self.layer_idx].batch_add(value[b, h].view(-1, head_size).detach().cpu())

    if layer_past is not None:
        past_key, past_value = layer_past
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)

    if use_cache is True:
        present = (key, value)
    else:
        present = None

    if self.reorder_and_upcast_attn:
        attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
    else:
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

    attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs  # a, present, (attentions)

def pq_forward_legacy(
    self: GPT2Attention,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
    if encoder_hidden_states is not None:
        if not hasattr(self, "q_attn"):
            raise ValueError(
                "If class is used as cross attention, the weights `q_attn` have to be defined. "
                "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
            )

        query = self.q_attn(hidden_states)
        key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
        attention_mask = encoder_attention_mask
    else:
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

    query = self._split_heads(query, self.num_heads, self.head_dim)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    value = self._split_heads(value, self.num_heads, self.head_dim)

    assert DynamicPQCache.has_instance(), 'Cache must be set before using it.'
    cache = DynamicPQCache()
    distort_recent = UniConfig().distort_recent
    key, value = cache.update(key, value, self.layer_idx, distort_recent=distort_recent)

    if use_cache is True:
        present = (key, value)
    else:
        present = None

    if self.reorder_and_upcast_attn:
        attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
    else:
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

    attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs  # a, present, (attentions)

def save_forward(
    self: GPT2SdpaAttention,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
    if output_attentions or head_mask is not None:
        logger.warning_once(
            "`GPT2SdpaAttention` is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
            "`output_attentions=True` or `head_mask`. Falling back to the manual attention implementation, but "
            "specifying the manual implementation will be required from Transformers version v5.0.0 onwards. "
            'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

    bsz, q_len, _ = hidden_states.size()

    # Initial attention projections
    is_cross_attention = encoder_hidden_states is not None
    if is_cross_attention:
        if not hasattr(self, "q_attn"):
            raise ValueError(
                "If class is used as cross attention, the weights `q_attn` have to be defined. "
                "Please make sure to instantiate class with `GPT2SdpaAttention(..., is_cross_attention=True)`."
            )

        query = self.q_attn(hidden_states)
        key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
        attention_mask = encoder_attention_mask
    else:
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

    query = self._split_heads(query, self.num_heads, self.head_dim)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    value = self._split_heads(value, self.num_heads, self.head_dim)

    # Save key and value states to reservoir
    bs, nh, _, head_size = key.shape
    config = UniConfig()
    
    for b in range(bs):
        for h in range(nh):
            if config.merged_training is True:
                config.key_reservoir.batch_add(key[b, h].view(-1, head_size).detach().cpu())
                config.value_reservoir.batch_add(value[b, h].view(-1, head_size).detach().cpu())
            else:
                config.key_reservoir[self.layer_idx].batch_add(key[b, h].view(-1, head_size).detach().cpu())
                config.value_reservoir[self.layer_idx].batch_add(value[b, h].view(-1, head_size).detach().cpu())


    # Optional kv caching
    if layer_past is not None:
        past_key = layer_past[0]
        past_value = layer_past[1]
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)

    present = None
    if use_cache is True:
        present = (key, value)

    # Avoid torch==2.1.2 specific bug for the memory-efficient backend in SDPA
    if self.require_contiguous_qkv and query.device.type == "cuda" and attention_mask is not None:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if attention_mask is None and q_len > 1 and not is_cross_attention else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=self.attn_dropout.p if self.training else 0.0,
        is_causal=is_causal,
    )

    # Reshape outputs
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.embed_dim)

    # Final projection
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    return attn_output, present, None

def pq_forward(
    self: GPT2SdpaAttention,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
    if output_attentions or head_mask is not None:
        logger.warning_once(
            "`GPT2SdpaAttention` is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
            "`output_attentions=True` or `head_mask`. Falling back to the manual attention implementation, but "
            "specifying the manual implementation will be required from Transformers version v5.0.0 onwards. "
            'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

    bsz, q_len, _ = hidden_states.size()

    # Initial attention projections
    is_cross_attention = encoder_hidden_states is not None
    if is_cross_attention:
        if not hasattr(self, "q_attn"):
            raise ValueError(
                "If class is used as cross attention, the weights `q_attn` have to be defined. "
                "Please make sure to instantiate class with `GPT2SdpaAttention(..., is_cross_attention=True)`."
            )

        query = self.q_attn(hidden_states)
        key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
        attention_mask = encoder_attention_mask
    else:
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

    query = self._split_heads(query, self.num_heads, self.head_dim)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    value = self._split_heads(value, self.num_heads, self.head_dim)

    distort_recent = UniConfig().distort_recent
    assert DynamicPQCache.has_instance(), 'Cache must be set before using it.'
    cache = DynamicPQCache()
    key, value = cache.update(key, value, self.layer_idx, distort_recent=distort_recent)


    present = None
    if use_cache is True:
        present = (key, value)

    # Avoid torch==2.1.2 specific bug for the memory-efficient backend in SDPA
    if self.require_contiguous_qkv and query.device.type == "cuda" and attention_mask is not None:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if attention_mask is None and q_len > 1 and not is_cross_attention else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=self.attn_dropout.p if self.training else 0.0,
        is_causal=is_causal,
    )

    # Reshape outputs
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.embed_dim)

    # Final projection
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    return attn_output, present, None


model_context = ModelContext(
    init_context=ContextList([]),
    baseline_context=ContextList([]),
    sampling_context= ContextList([
        Injector(GPT2SdpaAttention, 'forward', save_forward),    
    ]),
    evaluation_context=ContextList([
        Injector(GPT2SdpaAttention, 'forward', pq_forward),
    ]),
)