"""
RoPE Enhanced Position Encoding for Qwen3 G2P Model
Utilize Qwen3's Rotary Position Embedding for better position awareness.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class RoPEEnhancedPositionModule(nn.Module):
    """Enhanced position module that leverages Qwen3's RoPE."""
    
    def __init__(self, hidden_size: int, max_position_embeddings: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        
        # RoPE parameters
        self.rope_theta = 10000.0
        self.rope_scaling = None
        
        # Create rotation matrix cache
        self._create_rope_cache()
        
        # Position-aware character embedding enhancement
        self.position_char_fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.position_dropout = nn.Dropout(0.1)
        
    def _create_rope_cache(self):
        """Create RoPE rotation matrix cache."""
        # Calculate dimension for RoPE (typically half of hidden_size)
        self.rope_dim = self.hidden_size // 2
        
        # Create frequency tensor
        inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, self.rope_dim, 2).float() / self.rope_dim
        ))
        self.register_buffer("inv_freq", inv_freq)
        
        # Pre-compute rotation matrices for common positions
        max_seq_len = self.max_position_embeddings
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Create cos and sin matrices
        cos_cached = torch.cos(freqs)
        sin_cached = torch.sin(freqs)
        
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)
    
    def _apply_rope_rotation(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Apply RoPE rotation to input tensor."""
        batch_size, seq_len, hidden_size = x.shape
        
        # Split into two halves for rotation
        x1 = x[..., :self.rope_dim]
        x2 = x[..., self.rope_dim:self.rope_dim*2]
        
        # Get cos and sin for positions
        cos = self.cos_cached[position_ids]  # [batch_size, seq_len, rope_dim//2]
        sin = self.sin_cached[position_ids]
        
        # Apply rotation
        # RoPE formula: [x1*cos - x2*sin, x1*sin + x2*cos]
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        
        # Concatenate rotated parts with remaining dimensions
        if hidden_size > self.rope_dim * 2:
            x_remaining = x[..., self.rope_dim*2:]
            x_rotated = torch.cat([x1_rot, x2_rot, x_remaining], dim=-1)
        else:
            x_rotated = torch.cat([x1_rot, x2_rot], dim=-1)
        
        return x_rotated
    
    def enhance_character_position(self, 
                                 char_hidden: torch.Tensor,
                                 position_ids: torch.Tensor,
                                 query_positions: torch.Tensor) -> torch.Tensor:
        """
        Enhance character representation with RoPE-aware position information.
        
        Args:
            char_hidden: Character hidden states [batch_size, hidden_size]
            position_ids: Position IDs in sequence [batch_size]
            query_positions: Query character positions [batch_size]
        
        Returns:
            Enhanced character representation
        """
        batch_size = char_hidden.shape[0]
        
        # Expand for RoPE application
        char_expanded = char_hidden.unsqueeze(1)  # [batch_size, 1, hidden_size]
        pos_expanded = position_ids.unsqueeze(1)  # [batch_size, 1]
        
        # Apply RoPE rotation to character representation
        char_rope = self._apply_rope_rotation(char_expanded, pos_expanded)
        char_rope = char_rope.squeeze(1)  # [batch_size, hidden_size]
        
        # Create query position representation
        query_expanded = query_positions.unsqueeze(1)  # [batch_size, 1]
        query_hidden = char_hidden.unsqueeze(1)  # [batch_size, 1, hidden_size]
        query_rope = self._apply_rope_rotation(query_hidden, query_expanded)
        query_rope = query_rope.squeeze(1)  # [batch_size, hidden_size]
        
        # Fuse character and query position information
        fused_input = torch.cat([char_rope, query_rope], dim=-1)  # [batch_size, hidden_size*2]
        enhanced_char = self.position_char_fusion(fused_input)
        enhanced_char = self.position_dropout(enhanced_char)
        
        return enhanced_char
    
    def compute_relative_position_bias(self, 
                                     char_positions: torch.Tensor,
                                     context_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute relative position bias using RoPE.
        
        Args:
            char_positions: Target character positions [batch_size]
            context_positions: Context positions [batch_size, context_len]
        
        Returns:
            Relative position bias [batch_size, context_len]
        """
        batch_size, context_len = context_positions.shape
        
        # Calculate relative distances
        char_pos_expanded = char_positions.unsqueeze(1)  # [batch_size, 1]
        relative_distances = torch.abs(context_positions - char_pos_expanded)  # [batch_size, context_len]
        
        # Apply RoPE-based distance encoding
        # Use cosine similarity of RoPE embeddings as position bias
        char_rope_emb = self.cos_cached[char_positions]  # [batch_size, rope_dim//2]
        context_rope_emb = self.cos_cached[context_positions]  # [batch_size, context_len, rope_dim//2]
        
        # Compute cosine similarity as position bias
        char_rope_expanded = char_rope_emb.unsqueeze(1)  # [batch_size, 1, rope_dim//2]
        position_bias = torch.cosine_similarity(
            char_rope_expanded, context_rope_emb, dim=-1
        )  # [batch_size, context_len]
        
        # Apply distance decay
        distance_decay = torch.exp(-relative_distances.float() / 10.0)
        position_bias = position_bias * distance_decay
        
        return position_bias

class RoPEAwareConditionalLayer(nn.Module):
    """RoPE-aware conditional weight layer for G2P."""
    
    def __init__(self, hidden_size: int, num_labels: int, num_chars: int, num_pos: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        
        # RoPE enhanced position module
        self.rope_position = RoPEEnhancedPositionModule(hidden_size)
        
        # Enhanced character embedding with position awareness
        self.char_embedding = nn.Embedding(num_chars, hidden_size)
        self.pos_embedding = nn.Embedding(num_pos, hidden_size)
        
        # Position-aware cross interaction
        self.char_pos_cross = nn.Linear(hidden_size * 2, hidden_size)
        self.position_gate = nn.Linear(hidden_size, 1)
        
        # Final conditional weight generation
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
        
    def forward(self, 
                char_ids: torch.Tensor,
                pos_ids: torch.Tensor,
                position_ids: torch.Tensor,
                query_positions: torch.Tensor,
                context_hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate conditional weights with RoPE position awareness.
        
        Args:
            char_ids: Character IDs [batch_size]
            pos_ids: POS tag IDs [batch_size]  
            position_ids: Position in sequence [batch_size]
            query_positions: Query character positions [batch_size]
            context_hidden: Context hidden states [batch_size, hidden_size]
        
        Returns:
            Conditional weights [batch_size, num_labels]
        """
        batch_size = char_ids.shape[0]
        
        # Get basic embeddings
        char_emb = self.char_embedding(char_ids)  # [batch_size, hidden_size]
        pos_emb = self.pos_embedding(pos_ids)    # [batch_size, hidden_size]
        
        # Enhance character embedding with RoPE position information
        char_enhanced = self.rope_position.enhance_character_position(
            char_emb, position_ids, query_positions
        )
        
        # Character-POS cross interaction with position awareness
        char_pos_input = torch.cat([char_enhanced, pos_emb], dim=-1)
        char_pos_cross = self.char_pos_cross(char_pos_input)
        
        # Position gate to control position influence
        position_gate = torch.sigmoid(self.position_gate(char_enhanced))
        char_pos_gated = char_pos_cross * position_gate
        
        # Combine all features
        if context_hidden is not None:
            combined_features = torch.cat([char_enhanced, char_pos_gated, context_hidden], dim=-1)
        else:
            # Use char_enhanced as context if not provided
            combined_features = torch.cat([char_enhanced, char_pos_gated, char_enhanced], dim=-1)
        
        # Generate conditional weights
        conditional_weights = self.weight_generator(combined_features)
        
        # Apply sigmoid to get soft weights
        conditional_weights = torch.sigmoid(conditional_weights)
        
        return conditional_weights

def integrate_rope_into_g2pw(model, config):
    """
    Integrate RoPE enhancements into existing G2PW model.
    
    Args:
        model: Existing G2PW model
        config: Configuration object
    
    Returns:
        Enhanced model with RoPE features
    """
    # Add RoPE-aware conditional layer if using conditional weighting
    if hasattr(model, 'conditional_layer') and config.use_conditional:
        # Get dimensions from existing model
        hidden_size = model.qwen3_model.config.hidden_size
        num_labels = len(model.labels)
        num_chars = len(model.chars)
        num_pos = len(model.pos_tags) if hasattr(model, 'pos_tags') else 10
        
        # Replace conditional layer with RoPE-aware version
        model.rope_conditional_layer = RoPEAwareConditionalLayer(
            hidden_size, num_labels, num_chars, num_pos
        )
        
        print("✓ RoPE-aware conditional layer integrated")
    
    return model
