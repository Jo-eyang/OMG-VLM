# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import PretrainedConfig


class QWenConfig(PretrainedConfig):
    model_type = "qwen"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        emb_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        layer_norm_epsilon=1e-6,
        initializer_range=0.02,
        max_position_embeddings=8192,
        seq_length=2048,
        scale_attn_weights=True,
        use_cache=True,
        bf16=False,
        fp16=False,
        fp32=False,
        kv_channels=128,
        rotary_pct=1.0,
        rotary_emb_base=10000,
        use_dynamic_ntk=True,
        use_logn_attn=True,
        use_flash_attn="auto",
        intermediate_size=22016,
        no_bias=True,
        tie_word_embeddings=False,
        # Graph-aware visual adapter configurations.
        visual_adapter_num_layers=1,
        use_visual_compressor=True,
        visual_adapter_num_queries=32,
        visual_adapter_num_heads=32,
        textual_aggregation_num_heads=16,
        textual_aggregation_pool_layers=1,
        textual_aggregation_pool_mlp_ratio=4.0,
        textual_aggregation_context_tokens=16,
        textual_aggregation_separator_tokens=4,
        textual_aggregation_dropout=0.1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.emb_dropout_prob = emb_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.seq_length = seq_length
        self.bf16 = bf16
        self.fp16 = fp16
        self.fp32 = fp32
        self.kv_channels = kv_channels
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.use_dynamic_ntk = use_dynamic_ntk
        self.use_logn_attn = use_logn_attn
        self.use_flash_attn = use_flash_attn
        self.no_bias = no_bias
        # Graph-aware visual adapter settings.
        self.visual_adapter_num_layers = visual_adapter_num_layers
        self.use_visual_compressor = use_visual_compressor
        self.visual_adapter_num_queries = visual_adapter_num_queries
        self.visual_adapter_num_heads = visual_adapter_num_heads
        # Target-aware textual aggregation settings
        self.textual_aggregation_num_heads = textual_aggregation_num_heads
        self.textual_aggregation_pool_layers = textual_aggregation_pool_layers
        self.textual_aggregation_pool_mlp_ratio = textual_aggregation_pool_mlp_ratio
        self.textual_aggregation_context_tokens = textual_aggregation_context_tokens
        self.textual_aggregation_separator_tokens = textual_aggregation_separator_tokens
        self.textual_aggregation_dropout = textual_aggregation_dropout
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
