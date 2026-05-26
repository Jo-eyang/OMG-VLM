# Copyright (c) Alibaba Cloud.
# OMG-VLM graph-aware image modules.

from typing import List, Optional

import torch
from torch import nn

from Qwen_VL_Chat.visual import VisionTransformer


class PerNeighborVisualCompressor(nn.Module):
    """
    Per-Neighbor Visual Compressor module for compressing neighbor image features.
    
    Each neighbor feature [256, 4096] is compressed to [num_queries, 4096] using learnable queries.
    Optimized for GPU efficiency with batched processing.
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        num_queries: int = 32,
        num_heads: int = 32,
        dropout: float = 0.1,
        debug: bool = False,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.debug = debug
        
        # Ensure hidden_size is divisible by num_heads
        assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        
        # Learnable queries for compression
        self.queries = nn.Parameter(torch.empty(num_queries, hidden_size))
        nn.init.xavier_uniform_(self.queries)
        
        # Cross-attention for neighbor compression - use batch_first=True for efficiency
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # More efficient memory layout
        )
        
        # Layer normalization
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key_value_norm = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, neighbor_feature: torch.Tensor) -> torch.Tensor:
        """
        Compress a single neighbor feature using learnable queries.
        
        Args:
            neighbor_feature: [256, 4096] neighbor feature from VisionTransformer
            
        Returns:
            torch.Tensor: [num_queries, 4096] compressed neighbor feature
        """
        # Wrap single neighbor as batch and use batched forward
        return self.forward_batch(neighbor_feature.unsqueeze(0)).squeeze(0)

    def forward_batch(self, neighbor_features: torch.Tensor) -> torch.Tensor:
        """
        Compress multiple neighbor features in a single batched operation.
        This is the primary method - optimized for GPU efficiency.
        
        Args:
            neighbor_features: [N, 256, 4096] stacked neighbor features
            
        Returns:
            torch.Tensor: [N, num_queries, 4096] compressed neighbor features
        """
        batch_size = neighbor_features.shape[0]
        device = neighbor_features.device
        dtype = neighbor_features.dtype
        
        # Prepare queries: [num_queries, hidden] -> [N, num_queries, hidden]
        queries = self.query_norm(self.queries.to(dtype=dtype, device=device))
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)  # [N, 32, 4096]
        
        # Normalize neighbor features: [N, 256, 4096]
        neighbor_norm = self.key_value_norm(neighbor_features)  # [N, 256, 4096]
        
        # Batched cross-attention with batch_first=True
        # Query: [N, 32, 4096], Key/Value: [N, 256, 4096]
        attn_output, _ = self.cross_attention(
            query=queries,          # [N, 32, 4096]
            key=neighbor_norm,      # [N, 256, 4096]
            value=neighbor_norm,    # [N, 256, 4096]
            need_weights=False,     # Skip weight computation for speed
        )
        # attn_output: [N, 32, 4096]
        
        # Apply dropout, projection, and norm
        output = self.dropout(attn_output)
        output = self.output_proj(output)
        output = self.output_norm(output)
        
        if self.debug:
            print(f"PerNeighborVisualCompressor: [{batch_size}, 256, 4096] -> [{batch_size}, {self.num_queries}, 4096]")
        
        return output


class GraphAwareVisualAdapter(nn.Module):
    """
    Enhanced Graph-Aware Visual Adapter module with optional per-neighbor visual compressor.
    
    Architecture:
    1. Center image features [256, 4096] -> Self-attention -> center_features [256, 4096]
    2. Neighbor features processed by PerNeighborVisualCompressor (if enabled) or used directly
    3. Cross-attention between center_features and neighbor queries/features -> mixed_feature [256, 4096]
    4. Residual connection: output = mixed_feature + center_features (input)
    5. Multiple layers can be stacked
    
    Args:
        hidden_size (int): Hidden dimension size (default: 4096)
        num_queries (int): Number of neighbor queries (default: 32 when compressor enabled, 256 when disabled)
        num_heads (int): Number of attention heads (default: 32)
        num_layers (int): Number of GraphAwareVisualAdapter layers (default: 1)
        dropout (float): Dropout probability (default: 0.1)
        use_visual_compressor (bool): Whether to use per-neighbor visual compressor (default: True)
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        num_queries: int = 32,
        num_heads: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_visual_compressor: bool = True,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_visual_compressor = use_visual_compressor
        
        # Ensure hidden_size is divisible by num_heads
        assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        
        # Initialize PerNeighborVisualCompressor if enabled
        if self.use_visual_compressor:
            self.visual_compressor = PerNeighborVisualCompressor(
                hidden_size=hidden_size,
                num_queries=num_queries,
                num_heads=num_heads,
                dropout=dropout,
            )
        
        # GraphAwareVisualAdapter layers
        self.layers = nn.ModuleList([
            CenterConditionedVisualFusionLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
            ) for _ in range(num_layers)
        ])
        
    def forward(self, center_features: torch.Tensor, neighbor_features: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass of enhanced GraphAwareVisualAdapter. Optimized for GPU efficiency.
        
        Args:
            center_features: [256, 4096] center image features from VisionTransformer
            neighbor_features: List of neighbor features, each [256, 4096], or None
            
        Returns:
            torch.Tensor: [256, 4096] enhanced center features
        """
        # Process neighbor features
        neighbor_queries = None
        if neighbor_features is not None and len(neighbor_features) > 0:
            if self.use_visual_compressor:
                # Use BATCHED per-neighbor visual compressor for better GPU utilization
                # Stack neighbors: List[Tensor[256, 4096]] -> Tensor[N, 256, 4096]
                stacked_neighbors = torch.stack(neighbor_features, dim=0)
                
                # Batched compression: [N, 256, 4096] -> [N, num_queries, 4096]
                compressed_batch = self.visual_compressor.forward_batch(stacked_neighbors)
                
                # Flatten to [N * num_queries, 4096] for cross-attention
                neighbor_queries = compressed_batch.reshape(-1, compressed_batch.shape[-1])
            else:
                # Use original neighbor features: [256, 4096] for each neighbor
                neighbor_queries = torch.cat(neighbor_features, dim=0)  # [num_neighbors * 256, 4096]
        
        # Process through GraphAwareVisualAdapter layers
        current_features = center_features  # [256, 4096]
        
        for layer in self.layers:
            current_features = layer(current_features, neighbor_queries)
        
        return current_features


class CenterConditionedVisualFusionLayer(nn.Module):
    """
    Single GraphAwareVisualAdapter layer implementation. Optimized for GPU efficiency.
    
    Architecture for one layer:
    1. Self-attention on center features
    2. Cross-attention with neighbor queries/features  
    3. Residual connection + FFN
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Self-attention for center features - use batch_first=True for efficiency
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention with neighbor features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.self_norm = nn.LayerNorm(hidden_size)
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, center_features: torch.Tensor, neighbor_queries: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Optimized forward pass with minimal overhead.
        
        Args:
            center_features: [256, 4096] center image features
            neighbor_queries: [num_neighbors * queries_per_neighbor, 4096] neighbor queries/features or None
            
        Returns:
            torch.Tensor: [256, 4096] enhanced center features
        """
        # Save input for residual connection
        input_features = center_features  # [256, 4096]
        
        # Step 1: Self-attention on center features
        # Add batch dim for batch_first attention: [256, 4096] -> [1, 256, 4096]
        center_norm = self.self_norm(center_features).unsqueeze(0)
        
        self_attn_output, _ = self.self_attention(
            query=center_norm,
            key=center_norm,
            value=center_norm,
            need_weights=False,
        )
        self_attn_output = self.dropout(self_attn_output).squeeze(0)  # [256, 4096]
        
        # Step 2: Cross-attention with neighbor features (if available)
        if neighbor_queries is not None:
            center_norm_cross = self.cross_norm(self_attn_output).unsqueeze(0)  # [1, 256, 4096]
            neighbor_norm = neighbor_queries.unsqueeze(0)  # [1, N, 4096]
            
            cross_attn_output, _ = self.cross_attention(
                query=center_norm_cross,  # [1, 256, 4096]
                key=neighbor_norm,        # [1, N, 4096]
                value=neighbor_norm,      # [1, N, 4096]
                need_weights=False,
            )
            mixed_features = self.dropout(cross_attn_output).squeeze(0)  # [256, 4096]
        else:
            mixed_features = self_attn_output  # [256, 4096]
        
        # Step 3: Residual connection
        residual_output = mixed_features + input_features  # [256, 4096]
        
        # Step 4: Feed-forward network with residual
        ffn_output = self.ffn(self.output_norm(residual_output))
        final_output = residual_output + ffn_output  # [256, 4096]
        
        return final_output


class GraphAwareVisionModel(nn.Module):
    """
    Comprehensive vision model combining VisionTransformer and enhanced Graph-Aware Visual Adapter
    for main and neighbor image feature extraction and processing.
    
    This model processes main images and their neighbors according to the new Graph-Aware Visual Adapter architecture:
    - Main image features go through Graph-Aware Visual Adapter layers
    - Neighbor features are compressed (if enabled) and used in cross-attention
    - Output maintains the same shape as main image features [256, 4096]
    """
    
    def __init__(
        self,
        vision_config: dict,
        visual_adapter_config: dict = None,
        fix_vit: bool = True,
    ):
        super().__init__()
        
        # Initialize VisionTransformer
        self.vision_transformer = VisionTransformer(**vision_config)
        
        # Initialize enhanced Graph-Aware Visual Adapter with default config if not provided
        if visual_adapter_config is None:
            visual_adapter_config = {
                'hidden_size': vision_config.get('output_dim', 4096),
                'num_queries': 32,
                'num_heads': 32,
                'num_layers': 1,
                'dropout': 0.1,
                'use_visual_compressor': True,
            }
        
        self.visual_adapter = GraphAwareVisualAdapter(**visual_adapter_config)
        
        # Option to freeze VisionTransformer parameters
        self.fix_vit = fix_vit
        if fix_vit:
            for param in self.vision_transformer.parameters():
                param.requires_grad = False
    
    def forward(
        self, 
        main_image_path: str, 
        neighbor_image_paths: List[str] = None
    ) -> torch.Tensor:
        """
        Forward pass processing main image and neighbors with new architecture.
        
        Args:
            main_image_path (str): Path to main image
            neighbor_image_paths (List[str], optional): List of neighbor image paths
            
        Returns:
            torch.Tensor: Enhanced main image features from Graph-Aware Visual Adapter [1, 256, hidden_size]
        """
        # Extract main features using VisionTransformer
        main_features = self.vision_transformer.encode([main_image_path])  # [1, 256, hidden_size]
        
        # Extract neighbor features if provided
        neighbor_features = None
        if neighbor_image_paths and len(neighbor_image_paths) > 0:
            # Encode each neighbor separately
            neighbor_features = []
            for neighbor_path in neighbor_image_paths:
                neighbor_feat = self.vision_transformer.encode([neighbor_path])  # [1, 256, hidden_size]
                neighbor_features.append(neighbor_feat.squeeze(0))  # [256, hidden_size]
        
        # Process through enhanced Graph-Aware Visual Adapter
        enhanced_features = self.visual_adapter(
            center_features=main_features.squeeze(0),  # [256, hidden_size]
            neighbor_features=neighbor_features
        )  # [256, hidden_size]
        
        return enhanced_features.unsqueeze(0)  # [1, 256, hidden_size]


def create_graph_aware_vision_model(
    image_size: int = 448,
    patch_size: int = 14, 
    width: int = 1664,
    layers: int = 48,
    heads: int = 16,
    mlp_ratio: float = 4.9231,
    n_queries: int = 256,
    output_dim: int = 4096,
    visual_adapter_num_queries: int = 32,
    visual_adapter_num_heads: int = 32,
    visual_adapter_num_layers: int = 1,
    use_visual_compressor: bool = True,
    fix_vit: bool = True,
    **kwargs
) -> GraphAwareVisionModel:
    """
    Factory function to create GraphAwareVisionModel with enhanced Graph-Aware Visual Adapter configurations.
    
    Args:
        Standard VisionTransformer parameters...
        visual_adapter_num_queries (int): Number of Graph-Aware Visual Adapter neighbor queries (default: 32)
        visual_adapter_num_heads (int): Number of Graph-Aware Visual Adapter attention heads (default: 32) 
        visual_adapter_num_layers (int): Number of Graph-Aware Visual Adapter layers (default: 1)
        use_visual_compressor (bool): Whether to use per-neighbor visual compressor (default: True)
        fix_vit (bool): Whether to freeze VisionTransformer parameters (default: True)
        
    Returns:
        GraphAwareVisionModel: Configured model ready for training/inference
    """
    
    vision_config = {
        'image_size': image_size,
        'patch_size': patch_size,
        'width': width,
        'layers': layers,
        'heads': heads,
        'mlp_ratio': mlp_ratio,
        'n_queries': n_queries,
        'output_dim': output_dim,
        **kwargs
    }
    
    visual_adapter_config = {
        'hidden_size': output_dim,
        'num_queries': visual_adapter_num_queries,
        'num_heads': visual_adapter_num_heads,
        'num_layers': visual_adapter_num_layers,
        'dropout': 0.1,
        'use_visual_compressor': use_visual_compressor,
    }
    
    return GraphAwareVisionModel(
        vision_config=vision_config,
        visual_adapter_config=visual_adapter_config,
        fix_vit=fix_vit
    )


__all__ = [
    "CenterConditionedVisualFusionLayer",
    "GraphAwareVisionModel",
    "GraphAwareVisualAdapter",
    "PerNeighborVisualCompressor",
    "create_graph_aware_vision_model",
]
