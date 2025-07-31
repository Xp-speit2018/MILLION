from ...utils.Injector import Injector
from ...utils.Namespace import UniConfig
from ...utils.ContextList import ContextList
from ...utils.Timer import Timer
from ...utils.pq_utils import DynamicPQCache
from transformers.models.mpt.modeling_mpt import (
    MptForCausalLM,
    MptAttention,
    CrossEntropyLoss,
    CausalLMOutputWithCrossAttentions,
)

import torch
from typing import Optional, Tuple
from .ModelContext import ModelContext
from torch import nn

from typing import Union


def tag_idx_forward(
    self: MptForCausalLM,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
        `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
        are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
    """
    for i, layer in enumerate(self.transformer.blocks):
        layer.attn.layer_idx = i
    
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = transformer_outputs[0]

    lm_logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        batch_size, seq_length, vocab_size = shift_logits.shape
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
        )

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithCrossAttentions(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )
    
def save_forward(
    self: MptAttention,
    hidden_states: torch.Tensor,
    position_bias: torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    batch_size, seq_length = hidden_states.shape[:2]

    mixed_qkv = self.Wqkv(hidden_states)
    query_states, key_states, value_states = mixed_qkv.chunk(3, dim=2)
    query_states = query_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)

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
        if len(past_key_value) != 0:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states)
    else:
        past_key_value = (key_states, value_states)

    attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.softmax_scale

    query_length = seq_length if past_key_value is None else seq_length + past_key_value[0].shape[2]

    if position_bias is not None:
        if len(position_bias.shape) != 3:
            raise ValueError(f"Expecting position_bias shape to be 3 dimensions, got {len(position_bias.shape)}")
        key_length = key_states.shape[-2]

        position_bias_query_index = max(0, position_bias.size(1) - query_length)
        position_bias_key_index = max(0, position_bias.size(2) - key_length)

        position_bias = position_bias[:, position_bias_query_index:, position_bias_key_index:]

        attention_scores = attention_scores + position_bias

    if attention_mask is not None:
        attention_scores = attention_scores.masked_fill(attention_mask, torch.finfo(query_states.dtype).min)

    # (batch_size, n_heads, seq_length, key_length)
    attn_weights = nn.functional.softmax(attention_scores.float(), dim=-1).to(value_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attn_dropout_p, training=self.training)

    context_states = torch.matmul(attn_weights, value_states)
    context_states = context_states.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
    attn_output = self.out_proj(context_states)

    return attn_output, attn_weights, past_key_value

def pq_forward(
    self: MptAttention,
    hidden_states: torch.Tensor,
    position_bias: torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    batch_size, seq_length = hidden_states.shape[:2]

    mixed_qkv = self.Wqkv(hidden_states)
    query_states, key_states, value_states = mixed_qkv.chunk(3, dim=2)
    query_states = query_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)

    distort_recent = UniConfig().distort_recent
    assert DynamicPQCache.has_instance(), 'Cache must be set before using it.'
    cache = DynamicPQCache()
    key_states, value_states = cache.update(key_states, value_states, self.layer_idx, distort_recent=distort_recent)

    attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.softmax_scale

    query_length = seq_length if past_key_value is None else seq_length + past_key_value[0].shape[2]

    if position_bias is not None:
        if len(position_bias.shape) != 3:
            raise ValueError(f"Expecting position_bias shape to be 3 dimensions, got {len(position_bias.shape)}")
        key_length = key_states.shape[-2]

        position_bias_query_index = max(0, position_bias.size(1) - query_length)
        position_bias_key_index = max(0, position_bias.size(2) - key_length)

        position_bias = position_bias[:, position_bias_query_index:, position_bias_key_index:]

        attention_scores = attention_scores + position_bias

    if attention_mask is not None:
        attention_scores = attention_scores.masked_fill(attention_mask, torch.finfo(query_states.dtype).min)

    # (batch_size, n_heads, seq_length, key_length)
    attn_weights = nn.functional.softmax(attention_scores.float(), dim=-1).to(value_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attn_dropout_p, training=self.training)

    context_states = torch.matmul(attn_weights, value_states)
    context_states = context_states.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
    attn_output = self.out_proj(context_states)

    return attn_output, attn_weights, past_key_value
    
model_context = ModelContext(
    init_context=ContextList([
        # Injector(MptForCausalLM, "forward", prepare_inputs_for_generation),
    ]),
    baseline_context=ContextList([]),
    sampling_context=ContextList([
        Injector(MptAttention, "forward", save_forward),
        Injector(MptForCausalLM, "forward", tag_idx_forward),
    ]),
    evaluation_context=ContextList([
        Injector(MptAttention, "forward", pq_forward),
        Injector(MptForCausalLM, "forward", tag_idx_forward),
    ]),
)