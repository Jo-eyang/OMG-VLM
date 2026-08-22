# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# QWEN_LICENSE file in the root directory of this source tree.

import importlib
import math
from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.cuda.amp import autocast

from torch.nn import CrossEntropyLoss
from transformers import PreTrainedTokenizer, GenerationConfig, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

try:
    from einops import rearrange
except ImportError:
    rearrange = None
from torch import nn

SUPPORT_CUDA = torch.cuda.is_available()
SUPPORT_BF16 = SUPPORT_CUDA and torch.cuda.is_bf16_supported()
SUPPORT_FP16 = SUPPORT_CUDA and torch.cuda.get_device_capability(0)[0] >= 7

from .configuration_qwen import QWenConfig
from .qwen_generation_utils import (
    HistoryType,
    make_context,
    decode_tokens,
    get_stop_words_ids,
    StopWordsLogitsProcessor,
)
from omg_vlm.image import GraphAwareVisualAdapter
from omg_vlm.text import TargetAwareTextualAggregation
from .visual import VisionTransformer


logger = logging.get_logger(__name__)


def _debug_print(*args, **kwargs):
    """Silence verbose research-time tracing unless edited locally."""
    return None

_CHECKPOINT_FOR_DOC = "qwen"
_CONFIG_FOR_DOC = "QWenConfig"

QWen_PRETRAINED_MODEL_ARCHIVE_LIST = ["qwen-7b"]

_ERROR_BAD_CHAT_FORMAT = """\
We detect you are probably using the pretrained model (rather than chat model) for chatting, since the chat_format in generation_config is not "chatml".
If you are directly using the model downloaded from Huggingface, please make sure you are using our "Qwen/Qwen-7B-Chat" Huggingface model (rather than "Qwen/Qwen-7B") when you call model.chat().
"""

_SENTINEL = object()
_ERROR_STREAM_IN_CHAT = """\
Pass argument `stream` to model.chat() is buggy, deprecated, and marked for removal. Please use model.chat_stream(...) instead of model.chat(..., stream=True).
"""

apply_rotary_emb_func = None
rms_norm = None


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    


class QWenAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)
        self.seq_length = config.seq_length

        self.hidden_size = config.hidden_size
        self.split_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.scale_attn_weights = True

        self.projection_size = config.kv_channels * config.num_attention_heads

        assert self.projection_size % config.num_attention_heads == 0
        self.hidden_size_per_attention_head = (
            self.projection_size // config.num_attention_heads
        )

        self.c_attn = nn.Linear(config.hidden_size, 3 * self.projection_size)

        self.c_proj = nn.Linear(
            config.hidden_size, self.projection_size, bias=not config.no_bias
        )

        self.is_fp32 = not (config.bf16 or config.fp16)
        self.bf16 = config.bf16

        self.use_dynamic_ntk = config.use_dynamic_ntk
        self.use_logn_attn = config.use_logn_attn

        logn_list = [
            math.log(i, self.seq_length) if i > self.seq_length else 1
            for i in range(1, 32768)
        ]
        self.logn_tensor = torch.tensor(logn_list)[None, :, None, None]

        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)

    def _attn(self, query, key, value, registered_causal_mask, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [],
                value.size(-1) ** 0.5,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )

        query_length, key_length = query.size(-2), key.size(-2)
        # causal_mask = self.bias[
        #     :, :, key_length - query_length : key_length, :key_length
        # ]
        # mask_value = torch.finfo(attn_weights.dtype).min
        # mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
        #     attn_weights.device
        # )
        # attn_weights = torch.where(
        #     causal_mask, attn_weights.to(attn_weights.dtype), mask_value
        # )
        attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(
        self, query, key, value, registered_causal_mask, attention_mask=None, head_mask=None
    ):
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        attn_weights = torch.empty(
            bsz * num_heads,
            q_seq_len,
            k_seq_len,
            dtype=torch.float32,
            device=query.device,
        )

        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(
                -1, dk, k_seq_len
            )
            attn_weights = torch.baddbmm(
                attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor
            )
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = registered_causal_mask[
            :, :, key_length - query_length : key_length, :key_length
        ]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device
        )
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if attn_weights.dtype != torch.float32:
            raise RuntimeError(
                "Error with upcasting, attn_weights does not have dtype torch.float32"
            )
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        rotary_pos_emb: Optional[List[torch.Tensor]] = None,
        registered_causal_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):

        mixed_x_layer = self.c_attn(hidden_states)

        query, key, value = mixed_x_layer.split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if rotary_pos_emb is not None:
            cur_len = query.shape[1]
            rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
            rotary_pos_emb = (rotary_pos_emb,) * 2
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # Slice the pos emb for current inference
            query = apply_rotary_pos_emb(query, q_pos_emb)
            key = apply_rotary_pos_emb(key, k_pos_emb)

        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)

        if use_cache:
            present = (key, value)
        else:
            present = None

        if self.use_logn_attn and not self.training:
            if self.logn_tensor.device != query.device or self.logn_tensor.dtype != query.dtype:
                self.logn_tensor = self.logn_tensor.to(query.device).type_as(query)
            seq_start = key.size(1) - query.size(1)
            seq_end = key.size(1)
            logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :]
            query = query * logn_tensor.expand_as(query)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        attn_output, attn_weight = self._attn(
            query, key, value, registered_causal_mask, attention_mask, head_mask
        )
        context_layer = self._merge_heads(
            attn_output, self.num_heads, self.head_dim
        )

        attn_output = self.c_proj(context_layer)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weight,)

        return outputs


class QWenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(
            config.hidden_size, config.intermediate_size // 2, bias=not config.no_bias
        )
        self.w2 = nn.Linear(
            config.hidden_size, config.intermediate_size // 2, bias=not config.no_bias
        )
        ff_dim_in = config.intermediate_size // 2
        self.c_proj = nn.Linear(ff_dim_in, config.hidden_size, bias=not config.no_bias)

    def forward(self, hidden_states):
        a1 = self.w1(hidden_states)
        a2 = self.w2(hidden_states)
        intermediate_parallel = a1 * F.silu(a2)
        output = self.c_proj(intermediate_parallel)
        return output

class QWenBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.bf16 = config.bf16

        self.ln_1 = RMSNorm(
            hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.attn = QWenAttention(config)
        self.ln_2 = RMSNorm(
            hidden_size,
            eps=config.layer_norm_epsilon,
        )

        self.mlp = QWenMLP(config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        rotary_pos_emb: Optional[List[torch.Tensor]] = None,
        registered_causal_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        layernorm_output = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            layernorm_output,
            rotary_pos_emb,
            registered_causal_mask=registered_causal_mask,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        residual = hidden_states
        layernorm_input = attn_output + residual

        layernorm_output = self.ln_2(layernorm_input)

        residual = layernorm_input
        mlp_output = self.mlp(layernorm_output)
        hidden_states = residual + mlp_output

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class QWenPreTrainedModel(PreTrainedModel):
    config_class = QWenConfig
    base_model_prefix = "transformer"
    is_parallelizable = False
    supports_gradient_checkpointing = True
    _no_split_modules = ["QWenBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                p.data.normal_(
                    mean=0.0,
                    std=(
                        self.config.initializer_range
                        / math.sqrt(2 * self.config.num_hidden_layers)
                    ),
                )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, QWenModel):
            module.gradient_checkpointing = value


class QWenModel(QWenPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_dim = config.hidden_size

        self.gradient_checkpointing = False
        self.use_dynamic_ntk = config.use_dynamic_ntk
        self.seq_length = config.seq_length

        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)

        self.drop = nn.Dropout(config.emb_dropout_prob)

        if config.rotary_pct == 1.0:
            self.rotary_ndims = None
        else:
            assert config.rotary_pct < 1
            self.rotary_ndims = int(
                config.kv_channels * config.rotary_pct
            )
        dim = (
            self.rotary_ndims
            if self.rotary_ndims is not None
            else config.kv_channels
        )
        self.rotary_emb = RotaryEmbedding(dim, base=config.rotary_emb_base)

        self.use_flash_attn = config.use_flash_attn
        self.is_fp32 = not (config.bf16 or config.fp16)
        self.registered_causal_mask = None
        # if (
        #     self.use_flash_attn
        #     and flash_attn_unpadded_func is not None
        #     and not self.is_fp32
        # ):
        #     self.registered_causal_mask = None
        # else:
        #     max_positions = config.max_position_embeddings
        #     self.register_buffer(
        #         "registered_causal_mask",
        #         torch.tril(
        #             torch.ones((max_positions, max_positions), dtype=torch.bool)
        #         ).view(1, 1, max_positions, max_positions),
        #         persistent=False,
        #     )

        self.h = nn.ModuleList(
            [
                QWenBlock(
                    config
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = RMSNorm(
            self.embed_dim,
            eps=config.layer_norm_epsilon,
        )

        self.visual = VisionTransformer(**config.visual)
        
        visual_adapter_config = {
            'hidden_size': config.visual.get('output_dim', 4096),
            'num_queries': getattr(config, 'visual_adapter_num_queries', 32),
            'num_heads': getattr(config, 'visual_adapter_num_heads', 32),
            'num_layers': getattr(config, 'visual_adapter_num_layers', 1),
            'dropout': 0.1,
            'use_visual_compressor': getattr(config, 'use_visual_compressor', True),
        }
        self.visual_adapter = GraphAwareVisualAdapter(**visual_adapter_config)

        textual_aggregation_heads = getattr(config, 'textual_aggregation_num_heads', 16)
        textual_aggregation_layers = getattr(config, 'textual_aggregation_pool_layers', 1)
        textual_aggregation_mlp_ratio = getattr(config, 'textual_aggregation_pool_mlp_ratio', 4.0)
        textual_aggregation_dropout = getattr(config, 'textual_aggregation_dropout', 0.1)
        textual_aggregation_queries = getattr(config, 'textual_aggregation_query_tokens', getattr(config, 'textual_aggregation_context_tokens', 16))

        self.textual_aggregation = TargetAwareTextualAggregation(
            embedding_layer=self.wte,
            hidden_size=self.embed_dim,
            num_attention_heads=textual_aggregation_heads,
            pool_num_layers=textual_aggregation_layers,
            pool_mlp_ratio=textual_aggregation_mlp_ratio,
            dropout=textual_aggregation_dropout,
            query_token_count=textual_aggregation_queries,
        )
        self.textual_aggregation_query_tokens = textual_aggregation_queries

        neighbor_info_ids = self.config.visual.get('neighbor_info_token_ids') if hasattr(self.config, 'visual') else None
        self.neighbor_info_token_ids = neighbor_info_ids if neighbor_info_ids else None
        if self.neighbor_info_token_ids is not None:
            self.neighbor_info_token_length = len(self.neighbor_info_token_ids)
        else:
            self.neighbor_info_token_length = self.config.visual.get('neighbor_info_token_length', 0) if hasattr(self.config, 'visual') else 0
        self._neighbor_info_token_tensor = (
            torch.tensor(self.neighbor_info_token_ids, dtype=torch.long)
            if self.neighbor_info_token_ids is not None
            else None
        )
        self.neighbor_info_string = self.config.visual.get('neighbor_info_string', " Neighbor Text Info:") if hasattr(self.config, 'visual') else " Neighbor Text Info:"
        _debug_print(f"  - neighbor bridge tokens: {self.neighbor_info_token_length}")
        
        # Option to freeze VisionTransformer parameters for efficient fine-tuning
        self.fix_vit = getattr(config, 'fix_vit', False)
        if self.fix_vit:
            for param in self.visual.parameters():
                param.requires_grad = False
                
        self.post_init()

    def set_text_info_data(self, text_info_data: dict, neighbor_data: dict = None):
        """Set text info and neighbor data for text graph node processing."""
        self.text_info_data = text_info_data
        if neighbor_data:
            self.neighbor_data = neighbor_data
        _debug_print(f"Set text_info_data with {len(text_info_data)} entries")
        if neighbor_data:
            _debug_print(f"Set neighbor_data with {len(neighbor_data)} entries")

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings
        if hasattr(self, 'textual_aggregation'):
            self.textual_aggregation.set_embedding_layer(new_embeddings)
    
    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        text_neighbor_ids: Optional[List[List[List[Union[torch.Tensor, List[int]]]]]] = None,  
    ):
        """
        Args:
            text_neighbor_ids: Nested list structure containing token ids for text node neighbors:
                - Batch level: List of samples
                - Sample level: List of text nodes in this sample
                - Node level: List of neighbor sequences (each sequence is a tensor or list of token ids)
                
                Example for batch_size=2:
                [
                    [[[11, 22], [33, 44]], [[55, 66, 77]]],  # Sample 0: 2 text nodes
                    [[[88, 99]]]                              # Sample 1: 1 text node
                ]
        """
        _debug_print(f"Transformer forward started")
        _debug_print(f"Input IDs shape: {input_ids.shape if input_ids is not None else 'None'}")
        _debug_print(f"Past key values: {'Present' if past_key_values is not None else 'None'}")
        _debug_print(f"Training mode: {self.training}")
        _debug_print(f"text_neighbor_ids: {'Present' if text_neighbor_ids is not None else 'None'}")
        if text_neighbor_ids is not None:
            _debug_print(f"  text_neighbor_ids structure: {len(text_neighbor_ids)} samples")
        
        if past_key_values is None and torch.any(input_ids == self.config.visual['image_start_id']):
            _debug_print(f"Found image tokens in input, processing images...")
            _debug_print(f"Image start ID: {self.config.visual['image_start_id']}")
            _debug_print(f"Image end ID: {self.config.visual['image_start_id'] + 1}")
            
            bos_pos = torch.where(input_ids == self.config.visual['image_start_id'])
            eos_pos = torch.where(input_ids == self.config.visual['image_start_id'] + 1)
            
            _debug_print(f"Found {len(bos_pos[0])} image start positions")
            _debug_print(f"Found {len(eos_pos[0])} image end positions")
            
            # Handle mismatched image start/end tokens
            if len(bos_pos[0]) != len(eos_pos[0]):
                _debug_print(f"Warning: Warning: Mismatched image tokens - {len(bos_pos[0])} start tokens vs {len(eos_pos[0])} end tokens")
                # Take the minimum length to avoid dimension mismatch
                min_len = min(len(bos_pos[0]), len(eos_pos[0]))
                bos_pos = (bos_pos[0][:min_len], bos_pos[1][:min_len])
                eos_pos = (eos_pos[0][:min_len], eos_pos[1][:min_len])
                _debug_print(f"Using {min_len} matching image token pairs")
                eos_pos = (eos_pos[0][:min_len], eos_pos[1][:min_len])
            
            # Additional safety check
            if len(bos_pos[0]) == 0 or len(eos_pos[0]) == 0:
                _debug_print("Warning: No valid image token pairs found, falling back to training mode")
                # Fall back to training mode processing
                fake_images = torch.zeros(1,3,224,224).to(
                    dtype=self.visual.conv1.weight.dtype, device=self.visual.conv1.weight.device)
                _debug_print(f"Creating fake images for fallback: {fake_images.shape}")
                # Create fake concatenated features for training
                main_fake = self.visual(fake_images)  # [1, 256, 4096]
                _debug_print(f"Fake main features shape: {main_fake.shape}")
                # Use only enhanced features (256 tokens) - no fake neighbor features needed
                images = main_fake  # [1, 256, 4096]
                _debug_print(f"Final fake features shape: {images.shape}")
            else:
                assert (bos_pos[0] == eos_pos[0]).all()
                img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
                _debug_print(f"Image position stack shape: {img_pos.shape}")
                
                # Extract main image paths and neighbor paths from tokenizer
                main_image_paths = []
                neighbor_image_paths = []
                
                for idx, (i, a, b) in enumerate(img_pos):
                    _debug_print(f"Processing image {idx}: batch {i}, tokens {a}-{b}")
                    # Extract encoded path data
                    encoded_data = input_ids[i][a + 1 : b - 1].tolist()
                    _debug_print(f"Encoded data length: {len(encoded_data)}")
                    _debug_print(f"First 10 encoded tokens: {encoded_data[:10]}")
                    
                    end_idx = encoded_data.index(self.config.visual['image_start_id'] + 2)
                    encoded_data = encoded_data[:end_idx]
                    _debug_print(f"Trimmed encoded data length: {len(encoded_data)}")
                    
                    # Decode path string
                    path_string = bytes(encoded_data).decode('utf-8')
                    _debug_print(f"Decoded path string: '{path_string[:100]}...'" if len(path_string) > 100 else f"Decoded path string: '{path_string}'")
                    
                    # Parse main and neighbor paths
                    if '|' in path_string:
                        # Format: "main_path|neighborID1,neighborID2,..." (IDs only)
                        parts = path_string.split('|', 1)
                        main_path = parts[0]
                        raw_ids = parts[1].split(',') if parts[1] else []
                        # Reconstruct full neighbor paths in same directory as main
                        import os
                        main_dir = os.path.dirname(main_path)
                        neighbor_paths = []
                        for nid in raw_ids:
                            nid = nid.strip()
                            if not nid:
                                continue
                            # Check if nid already has extension to avoid double .jpg
                            if '.' in nid and nid.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                                neighbor_paths.append(os.path.join(main_dir, nid))
                            else:
                                neighbor_paths.append(os.path.join(main_dir, f"{nid}.jpg"))
                        _debug_print(f"Parsed ID format - Main: '{main_path}', Neighbor IDs: {raw_ids} -> Paths: {neighbor_paths}")
                    else:
                        # Image node without encoded neighbor ids.
                        main_path = path_string
                        neighbor_paths = []
                        _debug_print(f"Image node without neighbor ids: '{main_path}'")
                    
                    main_image_paths.append(main_path)
                    neighbor_image_paths.append(neighbor_paths)
                
                _debug_print(f"Total main images: {len(main_image_paths)}")
                _debug_print(f"Neighbor groups: {[len(n) for n in neighbor_image_paths]}")
                
                # Extract main image features using VisionTransformer
                _debug_print(f"Encoding main images with VisionTransformer...")
                main_images = self.visual.encode(main_image_paths)  # [batch, 256, 4096]
                _debug_print(f"Main images encoded shape: {main_images.shape}")
                _debug_print(f"Main images dtype: {main_images.dtype}")
                _debug_print(f"Main images range: [{main_images.min():.6f}, {main_images.max():.6f}]")
                
                # Process each image through enhanced graph-aware visual adapter
                _debug_print(f"Processing images through enhanced GraphAwareVisualAdapter...")
                enhanced_features_list = []
                
                for idx, (main_feat, neighbor_paths) in enumerate(zip(main_images, neighbor_image_paths)):
                    _debug_print(f"Processing image {idx}: main shape {main_feat.shape}, {len(neighbor_paths)} neighbors")
                    
                    # Encode neighbor images if any
                    neighbor_features = None
                    if neighbor_paths and neighbor_paths != ['']:
                        _debug_print(f"Encoding {len(neighbor_paths)} neighbor images...")
                        neighbor_features = []
                        for neighbor_path in neighbor_paths:
                            try:
                                # Validate neighbor path exists before encoding
                                import os
                                if not os.path.exists(neighbor_path):
                                    _debug_print(f"Warning: Neighbor image not found, skipping: {neighbor_path}")
                                    continue
                                neighbor_feat = self.visual.encode([neighbor_path])  # [1, 256, 4096]
                                neighbor_features.append(neighbor_feat.squeeze(0))  # [256, 4096]
                            except Exception as e:
                                _debug_print(f"Warning: Failed to encode neighbor {neighbor_path}: {e}")
                                continue
                        _debug_print(f"Successfully encoded: {len(neighbor_features)} neighbor features")
                        # If no valid neighbors were encoded, set to None
                        if not neighbor_features:
                            neighbor_features = None
                    
                    # Process through enhanced graph-aware visual adapter
                    enhanced_feat = self.visual_adapter(
                        center_features=main_feat,  # [256, 4096]
                        neighbor_features=neighbor_features  # List of [256, 4096] or None
                    )  # [256, 4096]
                    enhanced_features_list.append(enhanced_feat)
                    _debug_print(f"Enhanced features shape: {enhanced_feat.shape}")
                
                # Use graph-aware visual features directly.
                enhanced_features = torch.stack(enhanced_features_list, dim=0)  # [batch, 256, 4096]
                _debug_print(f"Enhanced features shape: {enhanced_features.shape}")
                
                # Use enhanced features as the main output
                images = enhanced_features
                _debug_print(f"Final features shape: {images.shape}")
                _debug_print(f"Expected shape: [batch, 256, 4096] - Enhanced features only")
                
                assert images.shape[0] == len(main_image_paths)
                assert images.shape[1] == 256  # 256 enhanced tokens only
                _debug_print(f"Shape assertions passed - batch size: {images.shape[0]}, feature dims: {images.shape[1]}")
                fake_images = None
        elif self.training:
            _debug_print(f"Training mode: creating fake images and features")
            fake_images=torch.zeros(1,3,224,224).to(
                dtype=self.visual.conv1.weight.dtype, device=self.visual.conv1.weight.device)
            _debug_print(f"Fake training images shape: {fake_images.shape}")
            # Create fake concatenated features for training
            main_fake = self.visual(fake_images)  # [1, 256, 4096]
            _debug_print(f"Training main fake features shape: {main_fake.shape}")
            # Use only main features for training mode (256 tokens)
            images = main_fake  # [1, 256, 4096]
            _debug_print(f"Training fake features shape: {images.shape}")
        else:
            _debug_print(f"Inference mode without images: setting fake_images and images to None")
            fake_images = None
            images = None
        
        # Process text graph nodes with <nbr> placeholders
        text_context_jobs: List[dict] = []

        if past_key_values is None and hasattr(self.config, 'visual') and 'text_start_id' in self.config.visual:
            text_start_id = self.config.visual['text_start_id']
            text_end_id = self.config.visual.get('text_end_id', text_start_id + 1)
            nbr_token_id = self.config.visual.get('nbr_token_id', None)  # <nbr> token ID

            if torch.any(input_ids == text_start_id):
                _debug_print(f"Found text tokens in input, processing text nodes with <nbr> placeholders...")
                _debug_print(f"Text start ID: {text_start_id}")
                _debug_print(f"Text end ID: {text_end_id}")
                if nbr_token_id is not None:
                    _debug_print(f"<nbr> token ID: {nbr_token_id}")
                
                text_bos_pos = torch.where(input_ids == text_start_id)
                text_eos_pos = torch.where(input_ids == text_end_id)
                
                _debug_print(f"Found {len(text_bos_pos[0])} text start positions")
                _debug_print(f"Found {len(text_eos_pos[0])} text end positions")
                
                # Handle mismatched text start/end tokens
                if len(text_bos_pos[0]) != len(text_eos_pos[0]):
                    _debug_print(f"Warning: Warning: Mismatched text tokens")
                    min_len = min(len(text_bos_pos[0]), len(text_eos_pos[0]))
                    text_bos_pos = (text_bos_pos[0][:min_len], text_bos_pos[1][:min_len])
                    text_eos_pos = (text_eos_pos[0][:min_len], text_eos_pos[1][:min_len])
                
                if len(text_bos_pos[0]) > 0 and len(text_eos_pos[0]) > 0:
                    assert (text_bos_pos[0] == text_eos_pos[0]).all()
                    text_pos = torch.stack((text_bos_pos[0], text_bos_pos[1], text_eos_pos[1]), dim=1)
                    _debug_print(f"Text position stack shape: {text_pos.shape}")
                    
                    for idx, (i, a, b) in enumerate(text_pos):
                        batch_idx = i.item()
                        _debug_print(f"Processing text {idx}: batch {batch_idx}, tokens {a}-{b}")

                        text_token_ids = input_ids[batch_idx][a + 1 : b]
                        if nbr_token_id is None:
                            _debug_print(f"No <nbr> token configured; skipping text node {idx}")
                            continue

                        nbr_positions = (text_token_ids == nbr_token_id).nonzero(as_tuple=True)[0]
                        num_nbr_in_sequence = len(nbr_positions)
                        _debug_print(f"Found {num_nbr_in_sequence} <nbr> placeholders in sequence")

                        if num_nbr_in_sequence == 0:
                            _debug_print(f"Skipping text node {idx}: no <nbr> placeholder present")
                            continue

                        first_nbr_pos = nbr_positions[0].item()
                        bridge_len = getattr(self, 'neighbor_info_token_length', 0)
                        center_end = first_nbr_pos - bridge_len if bridge_len > 0 else first_nbr_pos
                        center_end = max(0, center_end)
                        center_token_ids = text_token_ids[:center_end]
                        if center_token_ids.numel() == 0:
                            # Fall back to everything before the placeholder if bridge tokens are missing
                            center_token_ids = text_token_ids[:first_nbr_pos]

                        if bridge_len > 0 and first_nbr_pos >= bridge_len:
                            bridge_slice = text_token_ids[first_nbr_pos - bridge_len:first_nbr_pos]
                            bridge_tensor = self._neighbor_info_token_tensor
                            if (
                                bridge_tensor is not None
                                and bridge_tensor.numel() == bridge_len
                                and bridge_slice.shape[0] == bridge_len
                            ):
                                expected = bridge_tensor.to(bridge_slice.device)
                                if not torch.equal(bridge_slice.detach(), expected):
                                    _debug_print(f"Warning: Bridge token mismatch detected for text node {idx}")

                        neighbor_token_sequences: List[Union[torch.Tensor, List[int]]] = []
                        if text_neighbor_ids is not None and batch_idx < len(text_neighbor_ids):
                            sample_neighbor_ids = text_neighbor_ids[batch_idx]
                            if idx < len(sample_neighbor_ids):
                                neighbor_token_sequences = sample_neighbor_ids[idx]
                        _debug_print(f"Retrieved {len(neighbor_token_sequences)} neighbor token sequences from argument")

                        fused_vectors = self.textual_aggregation(
                            target_ids=center_token_ids,
                            neighbor_ids=neighbor_token_sequences,
                        )
                        if fused_vectors.dim() == 3:
                            fused_vectors = fused_vectors.squeeze(0)
                        elif fused_vectors.dim() == 1:
                            fused_vectors = fused_vectors.unsqueeze(0)

                        vector_count = fused_vectors.shape[0]
                        required_slots = getattr(self, 'textual_aggregation_query_tokens', vector_count)
                        if vector_count != required_slots:
                            _debug_print(
                                f"Warning: Textual aggregation produced {vector_count} vectors, expected {required_slots}; proceeding with available vectors"
                            )

                        injection_positions = [a + 1 + pos.item() for pos in nbr_positions]

                        if len(injection_positions) == 1 and fused_vectors.shape[0] > 1:
                            # Single-slot data: collapse multiple context vectors to one.
                            fused_vectors = fused_vectors.mean(dim=0, keepdim=True)
                            vector_count = 1
                            _debug_print(
                                f"Warning: Only one <nbr> placeholder, collapsing {required_slots} fused vectors into a single mean vector"
                            )

                        if len(injection_positions) > fused_vectors.shape[0]:
                            pad = len(injection_positions) - fused_vectors.shape[0]
                            last_vec = fused_vectors[-1:].expand(pad, -1)
                            fused_vectors = torch.cat([fused_vectors, last_vec], dim=0)
                        elif len(injection_positions) < fused_vectors.shape[0]:
                            fused_vectors = fused_vectors[: len(injection_positions)]

                        text_context_jobs.append(
                            {
                                "batch_idx": batch_idx,
                                "positions": injection_positions[: fused_vectors.shape[0]],
                                "fused": fused_vectors,
                                "num_neighbors": len(neighbor_token_sequences),
                            }
                        )
                        _debug_print(
                            f"Prepared fused neighbor vectors for text node {idx} -> positions {injection_positions[: fused_vectors.shape[0]]}"
                        )

                    _debug_print(f"Prepared {len(text_context_jobs)} textual aggregation jobs so far")

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        encoder_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if batch_size <= 0:
            raise ValueError("batch_size has to be defined and > 0")
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_length
        )

        hidden_states = inputs_embeds

        kv_seq_len = hidden_states.size()[1]
        if past_key_values[0] is not None:
            # past key values[0][0] shape: bs * seq_len * head_num * dim
            kv_seq_len += past_key_values[0][0].shape[1]
        if (
            self.use_dynamic_ntk
            and kv_seq_len == hidden_states.size()[1]
            and not self.training
        ):
            context_value = math.log(kv_seq_len / self.seq_length, 2) + 1
            ntk_alpha = 2 ** math.ceil(context_value) - 1
            ntk_alpha = max(ntk_alpha, 1)
        else:
            ntk_alpha = self.rotary_emb._ntk_alpha_cached

        rotary_pos_emb = self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha)
        for idx in range(len(rotary_pos_emb)):
            rotary_pos_emb[idx] = rotary_pos_emb[idx].to(hidden_states.device)

        hidden_states = self.drop(hidden_states).clone()
        _debug_print(f"Hidden states after dropout shape: {hidden_states.shape}")
        _debug_print(f"Hidden states dtype: {hidden_states.dtype}")
        _debug_print(f"Hidden states range: [{hidden_states.min():.6f}, {hidden_states.max():.6f}]")
        
        if fake_images is not None:
            _debug_print(f"Adding fake images mean to hidden states (training mode)")
            _debug_print(f"Images mean value: {images.mean():.6f}")
            hidden_states = hidden_states + images.mean()*0
            _debug_print(f"Hidden states after fake image addition: [{hidden_states.min():.6f}, {hidden_states.max():.6f}]")
        elif images is not None:
            _debug_print(f"Injecting real image features into hidden states")
            _debug_print(f"Processing {len(img_pos)} image positions")
            for idx, (i, a, b) in enumerate(img_pos):
                _debug_print(f"Injecting image {idx}: batch {i}, positions {a+1}:{b}")
                _debug_print(f"Image features shape for injection: {images[idx].shape}")
                _debug_print(f"Target hidden state slice shape: {hidden_states[i][a + 1 : b].shape}")
                _debug_print(f"Image features range: [{images[idx].min():.6f}, {images[idx].max():.6f}]")
                
                # Store original hidden states for comparison
                original_slice = hidden_states[i][a + 1 : b].clone()
                
                # Inject image features
                hidden_states[i][a + 1 : b] = images[idx]
                
                _debug_print(f"Original hidden states range: [{original_slice.min():.6f}, {original_slice.max():.6f}]")
                _debug_print(f"After injection range: [{hidden_states[i][a + 1 : b].min():.6f}, {hidden_states[i][a + 1 : b].max():.6f}]")
                _debug_print(f"Successfully injected features for image {idx}")
            
            _debug_print(f"All image feature injections completed")
            _debug_print(f"Final hidden states shape: {hidden_states.shape}")
            _debug_print(f"Final hidden states range: [{hidden_states.min():.6f}, {hidden_states.max():.6f}]")
        else:
            _debug_print(f"No images to inject, continuing with text-only hidden states")
        
        # Inject fused neighbor vectors at each <nbr> placeholder
        if text_context_jobs:
            _debug_print(f"Injecting {len(text_context_jobs)} fused neighbor vectors")
            seq_len = hidden_states.shape[1]
            for job_idx, job in enumerate(text_context_jobs):
                batch_idx = job["batch_idx"]
                fused_vecs = job["fused"]
                if fused_vecs.dim() == 1:
                    fused_vecs = fused_vecs.unsqueeze(0)
                fused_vecs = fused_vecs.to(hidden_states.device).to(hidden_states.dtype)

                positions = job["positions"]
                valid_len = min(len(positions), fused_vecs.shape[0])
                if valid_len == 0:
                    _debug_print(f"Warning: Job {job_idx}: no valid positions for injection")
                    continue

                positions = positions[:valid_len]
                vecs_to_inject = fused_vecs[:valid_len]

                positions_are_consecutive = all(
                    positions[i] + 1 == positions[i + 1] for i in range(len(positions) - 1)
                )

                if positions_are_consecutive and 0 <= positions[0] and positions[-1] < seq_len:
                    start = positions[0]
                    end = start + valid_len
                    hidden_states[batch_idx, start:end, :] = vecs_to_inject
                    _debug_print(
                        f"Job {job_idx}: injected {valid_len} fused vectors (neighbors={job['num_neighbors']}) "
                        f"into batch {batch_idx}, positions {start}-{end - 1}"
                    )
                else:
                    for local_idx, pos in enumerate(positions):
                        if 0 <= pos < seq_len:
                            hidden_states[batch_idx, pos, :] = vecs_to_inject[local_idx]
                            _debug_print(
                                f"Job {job_idx}: injected fused vector idx {local_idx} (neighbors={job['num_neighbors']}) "
                                f"into batch {batch_idx}, position {pos}"
                            )
                        else:
                            _debug_print(f"Warning: Job {job_idx}: position {pos} out of bounds (seq_len={seq_len})")
            _debug_print(f"All fused neighbor injections completed")
        
        output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    rotary_pos_emb,
                    self.registered_causal_mask,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_reentrant=False,  # Required for DDP + LoRA compatibility
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    rotary_pos_emb=rotary_pos_emb,
                    registered_causal_mask=self.registered_causal_mask,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, presents, all_hidden_states] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class QWenLMHeadModel(QWenPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.rotary_emb\.inv_freq"]
    _keys_to_ignore_on_load_unexpected = [r"h\.\d+\.attn\.masked_bias"]

    def __init__(self, config):
        super().__init__(config)
        assert (
            config.bf16 + config.fp16 + config.fp32 <= 1
        ), "Only one of \"bf16\", \"fp16\", \"fp32\" can be true"

        autoset_precision = config.bf16 + config.fp16 + config.fp32 == 0

        if autoset_precision:
            if SUPPORT_BF16:
                logger.warn(
                    "The model is automatically converting to bf16 for faster inference. "
                    "If you want to disable the automatic precision, please manually add bf16/fp16/fp32=True to \"AutoModelForCausalLM.from_pretrained\"."
                )
                config.bf16 = True
            elif SUPPORT_FP16:
                logger.warn(
                    "The model is automatically converting to fp16 for faster inference. "
                    "If you want to disable the automatic precision, please manually add bf16/fp16/fp32=True to \"AutoModelForCausalLM.from_pretrained\"."
                )
                config.fp16 = True
            else:
                config.fp32 = True
        # if autoset_precision:
        #     if SUPPORT_BF16 and config.bf16:  # Honor explicit BF16 if set
        #         logger.warn("Using BF16 as configured.")
        #     elif SUPPORT_FP16 and config.fp16:  # Honor explicit FP16 from args
        #         logger.warn("Using FP16 as configured for better stability.")
        #     else:
        #         config.fp32 = True
        #         logger.warn("Falling back to FP32.")

        if config.bf16 and SUPPORT_CUDA and not SUPPORT_BF16:
            logger.warn("Your device does NOT seem to support bf16, you can switch to fp16 or fp32 by by passing fp16/fp32=True in \"AutoModelForCausalLM.from_pretrained\".")
        if config.fp16 and SUPPORT_CUDA and not SUPPORT_FP16:
            logger.warn("Your device does NOT support faster inference with fp16, please switch to fp32 which is likely to be faster")
        if config.fp32:
            if SUPPORT_BF16:
                logger.warn("Your device support faster inference by passing bf16=True in \"AutoModelForCausalLM.from_pretrained\".")
            elif SUPPORT_FP16:
                logger.warn("Your device support faster inference by passing fp16=True in \"AutoModelForCausalLM.from_pretrained\".")

        self.transformer = QWenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.bf16:
            self.transformer.bfloat16()
            self.lm_head.bfloat16()
        if config.fp16:
            self.transformer.half()
            self.lm_head.half()
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        _debug_print(f"Preparing inputs for generation")
        _debug_print(f"Input IDs shape: {input_ids.shape if input_ids is not None else 'None'}")
        _debug_print(f"Past key values: {'Present' if past_key_values is not None else 'None'}")
        _debug_print(f"Inputs embeds: {'Present' if inputs_embeds is not None else 'None'}")
        
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            _debug_print(f"Using past key values, taking last token only")
            input_ids = input_ids[:, -1].unsqueeze(-1)
            _debug_print(f"Truncated input IDs shape: {input_ids.shape}")
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        _debug_print(f"Attention mask shape: {attention_mask.shape if attention_mask is not None else 'None'}")
        _debug_print(f"Position IDs: {'Present' if position_ids is not None else 'None'}")

        if attention_mask is not None and position_ids is None:
            _debug_print(f"Computing position IDs from attention mask")
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            _debug_print(f"Computed position IDs shape: {position_ids.shape}")
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
                _debug_print(f"Truncated position IDs shape: {position_ids.shape}")
        else:
            position_ids = None

        if inputs_embeds is not None and past_key_values is None:
            _debug_print(f"Using inputs_embeds for model inputs")
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            _debug_print(f"Using input_ids for model inputs")
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "text_neighbor_ids": kwargs.get("text_neighbor_ids"),  # 
            }
        )
        _debug_print(f"Model inputs prepared with keys: {list(model_inputs.keys())}")
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        text_neighbor_ids: Optional[List[List[List[Union[torch.Tensor, List[int]]]]]] = None,  
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            text_neighbor_ids=text_neighbor_ids,  # 
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:

        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past_key_values
        )

    def chat(
        self,
        tokenizer: PreTrainedTokenizer,
        query: str,
        history: Optional[HistoryType],
        system: str = "You are a helpful assistant.",
        append_history: bool = True,
        stream: Optional[bool] = _SENTINEL,
        stop_words_ids: Optional[List[List[int]]] = None,
        generation_config: Optional[GenerationConfig] = None,
        text_neighbor_ids: Optional[List[List[List[Union[torch.Tensor, List[int]]]]]] = None,  
        **kwargs,
    ) -> Tuple[str, HistoryType]:
        _debug_print(f"Chat method started")
        _debug_print(f"Input query type: {type(query)}")
        _debug_print(f"Input query length: {len(str(query))}")
        _debug_print(f"Query content preview: {str(query)[:300]}...")
        _debug_print(f"History: {history}")
        _debug_print(f"System: {system}")
        _debug_print(f"Kwargs: {kwargs}")
        
        generation_config = generation_config if generation_config is not None else self.generation_config
        _debug_print(f"Using generation config:")
        _debug_print(f"   - chat_format: {generation_config.chat_format}")
        _debug_print(f"   - max_window_size: {getattr(generation_config, 'max_window_size', 'not set')}")
        _debug_print(f"   - top_p: {generation_config.top_p}")
        _debug_print(f"   - temperature: {getattr(generation_config, 'temperature', 'not set')}")
        _debug_print(f"   - max_new_tokens: {getattr(generation_config, 'max_new_tokens', 'not set')}")

        assert stream is _SENTINEL, _ERROR_STREAM_IN_CHAT
        assert generation_config.chat_format == 'chatml', _ERROR_BAD_CHAT_FORMAT
        if history is None:
            history = []
        if stop_words_ids is None:
            stop_words_ids = []

        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None:
            max_window_size = generation_config.max_window_size
        
        _debug_print(f"Making context with max_window_size: {max_window_size}")
        raw_text, context_tokens = make_context(
            tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
            chat_format=generation_config.chat_format,
        )
        
        _debug_print(f"Context creation completed:")
        _debug_print(f"   - raw_text length: {len(raw_text)}")
        _debug_print(f"   - context_tokens length: {len(context_tokens)}")
        _debug_print(f"   - raw_text preview: {raw_text[:200]}...")
        _debug_print(f"   - context_tokens preview: {context_tokens[:20]}...")

        stop_words_ids.extend(get_stop_words_ids(
            generation_config.chat_format, tokenizer
        ))
        _debug_print(f"Stop words IDs: {stop_words_ids}")
        
        input_ids = torch.tensor([context_tokens]).to(self.device)
        _debug_print(f"Input IDs created:")
        _debug_print(f"   - shape: {input_ids.shape}")
        _debug_print(f"   - device: {input_ids.device}")
        _debug_print(f"   - dtype: {input_ids.dtype}")
        _debug_print(f"   - first 10 tokens: {input_ids[0][:10].tolist()}")
        _debug_print(f"   - last 10 tokens: {input_ids[0][-10:].tolist()}")
        
        _debug_print(f"Starting generation...")
        outputs = self.generate(
                    input_ids,
                    stop_words_ids=stop_words_ids,
                    return_dict_in_generate=False,
                    generation_config=generation_config,
                    text_neighbor_ids=text_neighbor_ids,  # 
                    **kwargs,
                )
        
        _debug_print(f"Generation completed:")
        _debug_print(f"   - output shape: {outputs.shape}")
        _debug_print(f"   - output device: {outputs.device}")
        _debug_print(f"   - output dtype: {outputs.dtype}")
        _debug_print(f"   - generated tokens (first sequence): {outputs[0].tolist()}")

        _debug_print(f"Decoding tokens...")
        response = decode_tokens(
            outputs[0],
            tokenizer,
            raw_text_len=len(raw_text),
            context_length=len(context_tokens),
            chat_format=generation_config.chat_format,
            verbose=False,
            errors='replace'
        )
        
        _debug_print(f"Decoding completed:")
        _debug_print(f"   - response type: {type(response)}")
        _debug_print(f"   - response length: {len(response)}")
        _debug_print(f"   - response content: {response}")

        if append_history:
            history.append((query, response))
            _debug_print(f"History updated, new length: {len(history)}")

        _debug_print(f"Chat method completed successfully")
        return response, history

    def chat_stream(
            self,
            tokenizer: PreTrainedTokenizer,
            query: str,
            history: Optional[HistoryType],
            system: str = "You are a helpful assistant.",
            stop_words_ids: Optional[List[List[int]]] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            generation_config: Optional[GenerationConfig] = None,
            **kwargs,
    ) -> Generator[str, Any, None]:
        generation_config = generation_config if generation_config is not None else self.generation_config
        assert generation_config.chat_format == 'chatml', _ERROR_BAD_CHAT_FORMAT
        if history is None:
            history = []
        if stop_words_ids is None:
            stop_words_ids = []

        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None:
            max_window_size = generation_config.max_window_size
        raw_text, context_tokens = make_context(
            tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
            chat_format=generation_config.chat_format,
        )

        stop_words_ids.extend(get_stop_words_ids(
            generation_config.chat_format, tokenizer
        ))
        if stop_words_ids is not None:
            stop_words_logits_processor = StopWordsLogitsProcessor(
                stop_words_ids=stop_words_ids,
                eos_token_id=generation_config.eos_token_id,
            )
            if logits_processor is None:
                logits_processor = LogitsProcessorList([stop_words_logits_processor])
            else:
                logits_processor.append(stop_words_logits_processor)
        input_ids = torch.tensor([context_tokens]).to(self.device)

        from transformers_stream_generator.main import NewGenerationMixin, StreamGenerationConfig
        self.__class__.generate_stream = NewGenerationMixin.generate
        self.__class__.sample_stream = NewGenerationMixin.sample_stream
        stream_config = StreamGenerationConfig(**generation_config.to_dict(), do_stream=True)

        def stream_generator():
            outputs = []
            for token in self.generate_stream(
                    input_ids,
                    return_dict_in_generate=False,
                    generation_config=stream_config,
                    logits_processor=logits_processor,
                    seed=-1,
                    **kwargs):
                outputs.append(token.item())
                yield tokenizer.decode(outputs, skip_special_tokens=True, errors='ignore', keep_image_special=True)

        return stream_generator()

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        _debug_print(f"Generate method started")
        _debug_print(f"Input shape: {inputs.shape if inputs is not None else 'None'}")
        _debug_print(f"Input device: {inputs.device if inputs is not None else 'None'}")
        _debug_print(f"Input dtype: {inputs.dtype if inputs is not None else 'None'}")
        _debug_print(f"Kwargs: {list(kwargs.keys())}")
        
        generation_config = generation_config if generation_config is not None else self.generation_config
        _debug_print(f"Generation config loaded")

        # Process stop_words_ids.
        stop_words_ids = kwargs.pop("stop_words_ids", None)
        if stop_words_ids is None and generation_config is not None:
            stop_words_ids = getattr(generation_config, "stop_words_ids", None)
        if stop_words_ids is None:
            stop_words_ids = getattr(generation_config, "stop_words_ids", None)

        _debug_print(f"Stop words IDs processed: {stop_words_ids}")

        if stop_words_ids is not None:
            stop_words_logits_processor = StopWordsLogitsProcessor(
                stop_words_ids=stop_words_ids,
                eos_token_id=generation_config.eos_token_id,
            )
            if logits_processor is None:
                logits_processor = LogitsProcessorList([stop_words_logits_processor])
            else:
                logits_processor.append(stop_words_logits_processor)
            _debug_print(f"Stop words logits processor added")

        _debug_print(f"Calling super().generate()...")
        result = super().generate(
            inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            **kwargs,
        )
        
        _debug_print(f"Super().generate() completed")
        _debug_print(f"Result type: {type(result)}")
        _debug_print(f"Result shape: {result.shape if hasattr(result, 'shape') else 'no shape'}")
        if hasattr(result, 'shape') and len(result.shape) > 0:
            _debug_print(f"Generated sequence length: {result.shape[-1]}")
            _debug_print(f"Generated tokens: {result[0].tolist() if result.shape[0] > 0 else 'empty'}")
        
        _debug_print(f"Generate method completed")
        return result


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        if importlib.util.find_spec("einops") is None:
            raise RuntimeError("einops is required for Rotary Embedding")

        self._rotary_pos_emb_cache = None
        self._seq_len_cached = 0
        self._ntk_alpha_cached = 1.0

    def update_rotary_pos_emb_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
        seqlen = max_seq_len + offset
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, self.dim, 2, device=self.inv_freq.device).float()
                    / self.dim
                )
            )
            self._seq_len_cached = max(2 * seqlen, 16)
            self._ntk_alpha_cached = ntk_alpha
            seq = torch.arange(self._seq_len_cached, device=self.inv_freq.device)
            freqs = torch.outer(seq.type_as(self.inv_freq), self.inv_freq)
            
            emb = torch.cat((freqs, freqs), dim=-1)
            from einops import rearrange

            emb = rearrange(emb, "n d -> 1 n 1 d")

            cos, sin = emb.cos(), emb.sin()
            self._rotary_pos_emb_cache = [cos, sin]

    def forward(self, max_seq_len, offset=0, ntk_alpha=1.0):
        self.update_rotary_pos_emb_cache(max_seq_len, offset, ntk_alpha)
        cos, sin = self._rotary_pos_emb_cache
        return [cos[:, offset : offset + max_seq_len], sin[:, offset : offset + max_seq_len]]


def _rotate_half(x):
    from einops import rearrange

    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    cos, sin = freqs
    if apply_rotary_emb_func is not None and t.is_cuda:
        t_ = t.float()
        cos = cos.squeeze(0).squeeze(1)[:, : cos.shape[-1] // 2]
        sin = sin.squeeze(0).squeeze(1)[:, : sin.shape[-1] // 2]
        output = apply_rotary_emb_func(t_, cos, sin).type_as(t)
        return output
    else:
        rot_dim = freqs[0].shape[-1]
        cos, sin = freqs
        t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
        t_ = t_.float()
        t_pass_ = t_pass_.float()
        t_ = (t_ * cos) + (_rotate_half(t_) * sin)
        return torch.cat((t_, t_pass_), dim=-1).type_as(t)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if rms_norm is not None and x.is_cuda:
            return rms_norm(x, self.weight, self.eps)
        else:
            output = self._norm(x.float()).type_as(x)
            return output * self.weight
