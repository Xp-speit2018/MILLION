from ...utils.Injector import Injector
from ...utils.Namespace import UniConfig
from ...utils.ContextList import ContextList
from ...utils.Timer import Timer
from ...utils.pq_utils import DynamicPQCache
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2LMHeadModel,
    GPT2Attention,
    logger,
)

import torch
from typing import Optional, Tuple
from .ModelContext import ModelContext
from typing import Union


def save_forward(
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
                breakpoint()
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

def pq_forward(
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

model_context = ModelContext(
    init_context=ContextList([]),
    baseline_context=ContextList([]),
    sampling_context= ContextList([
        Injector(GPT2Attention, 'forward', save_forward),    
    ]),
    evaluation_context=ContextList([
        Injector(GPT2Attention, 'forward', pq_forward),
    ]),
)