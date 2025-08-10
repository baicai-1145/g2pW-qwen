"""
Pretrained weight loader for Qwen3 models.
This module handles loading and adapting Qwen3 pretrained weights for g2pW.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PretrainedWeightLoader:
    """
    Handles loading and adapting pretrained weights from Qwen3 models.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.pretrained_model = None
        self.pretrained_config = None
        
    def load_pretrained_model(self):
        """Load the pretrained Qwen3 model with modern transformers support."""
        try:
            print(f"Loading pretrained Qwen3 model from {self.model_path}...")

            # With modern transformers, we can load Qwen3 directly
            self.pretrained_config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            print(f"✓ Config loaded: {self.pretrained_config.model_type}")

            # Load the full model
            self.pretrained_model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map=None,  # Load to CPU first
                trust_remote_code=True
            )

            total_params = sum(p.numel() for p in self.pretrained_model.parameters())
            print(f"✓ Qwen3 model loaded successfully: {total_params:,} parameters")

            # Print model structure info
            print(f"  - Model type: {type(self.pretrained_model)}")
            print(f"  - Hidden size: {self.pretrained_config.hidden_size}")
            print(f"  - Vocab size: {self.pretrained_config.vocab_size}")
            print(f"  - Num layers: {self.pretrained_config.num_hidden_layers}")

            return True

        except Exception as e:
            print(f"✗ Failed to load pretrained Qwen3 model: {e}")
            logger.error(f"Pretrained model loading failed: {e}")

            # Fallback to state dict loading if direct loading fails
            print("Attempting fallback to state dict loading...")
            return self._load_state_dict_fallback()

    def _load_state_dict_fallback(self):
        """Fallback method to load state dict directly."""
        try:
            import json

            # Load config manually
            config_path = os.path.join(self.model_path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)

                # Create a simple config object
                class SimpleConfig:
                    def __init__(self, **kwargs):
                        for k, v in kwargs.items():
                            setattr(self, k, v)

                self.pretrained_config = SimpleConfig(**config_dict)

                # Try to load state dict
                model_files = ['model.safetensors', 'pytorch_model.bin']
                for model_file in model_files:
                    model_path_file = os.path.join(self.model_path, model_file)
                    if os.path.exists(model_path_file):
                        if model_file.endswith('.safetensors'):
                            from safetensors import safe_open
                            self.pretrained_state_dict = {}
                            with safe_open(model_path_file, framework="pt", device="cpu") as f:
                                for key in f.keys():
                                    self.pretrained_state_dict[key] = f.get_tensor(key)
                            print(f"✓ Fallback: Loaded state dict from {model_file}")
                            return True
                        elif model_file.endswith('.bin'):
                            self.pretrained_state_dict = torch.load(model_path_file, map_location='cpu')
                            print(f"✓ Fallback: Loaded state dict from {model_file}")
                            return True

            return False

        except Exception as e:
            print(f"✗ Fallback loading also failed: {e}")
            return False
    
    def extract_embeddings(self) -> Optional[torch.Tensor]:
        """Extract word embeddings from pretrained Qwen3 model."""
        try:
            # Method 1: Extract from loaded Qwen3 model
            if self.pretrained_model is not None:
                # For Qwen3, embeddings are typically at model.embed_tokens
                if hasattr(self.pretrained_model, 'embed_tokens'):
                    embeddings = self.pretrained_model.embed_tokens.weight
                    print(f"✓ Found Qwen3 embeddings: embed_tokens.weight {embeddings.shape}")
                    return embeddings.clone().detach()

                # Try other common locations
                embedding_paths = [
                    ('model.embed_tokens', 'weight'),
                    ('embeddings', 'weight'),
                    ('word_embeddings', 'weight')
                ]

                for path, attr in embedding_paths:
                    try:
                        obj = self.pretrained_model
                        for part in path.split('.'):
                            obj = getattr(obj, part)
                        if hasattr(obj, attr):
                            embeddings = getattr(obj, attr)
                            print(f"✓ Found embeddings: {path}.{attr} {embeddings.shape}")
                            return embeddings.clone().detach()
                    except AttributeError:
                        continue

            # Method 2: Extract from state dict (fallback)
            elif hasattr(self, 'pretrained_state_dict') and self.pretrained_state_dict:
                # Qwen3 state dict keys
                embedding_keys = [
                    'model.embed_tokens.weight',
                    'embed_tokens.weight',
                    'embeddings.weight',
                    'word_embeddings.weight'
                ]

                for key in embedding_keys:
                    if key in self.pretrained_state_dict:
                        embeddings = self.pretrained_state_dict[key]
                        print(f"✓ Found embeddings in state dict: {key} {embeddings.shape}")
                        return embeddings.clone()

                # Search for embedding-like tensors
                for key, value in self.pretrained_state_dict.items():
                    if ('embed' in key.lower() and 'weight' in key and
                        isinstance(value, torch.Tensor) and len(value.shape) == 2):
                        print(f"✓ Found embeddings in state dict: {key} {value.shape}")
                        return value.clone()

            print("✗ Could not find embeddings in Qwen3 model")
            return None

        except Exception as e:
            print(f"✗ Error extracting embeddings: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_transformer_layers(self) -> Optional[Dict[str, torch.Tensor]]:
        """Extract transformer layer weights from pretrained Qwen3 model."""
        if self.pretrained_model is None:
            return None

        try:
            layer_weights = {}

            # For Qwen3, layers are typically at model.layers
            layers = None
            if hasattr(self.pretrained_model, 'layers'):
                layers = self.pretrained_model.layers
            elif hasattr(self.pretrained_model, 'model') and hasattr(self.pretrained_model.model, 'layers'):
                layers = self.pretrained_model.model.layers

            if layers is None or not isinstance(layers, nn.ModuleList):
                print("✗ Could not find Qwen3 transformer layers")
                return None

            print(f"✓ Found Qwen3 transformer layers: {len(layers)} layers")

            # Extract weights from first few layers (we use fewer layers in our simplified model)
            max_layers = min(6, len(layers))  # Use first 6 layers to match our model

            for i in range(max_layers):
                layer = layers[i]
                layer_prefix = f"transformer.layers.{i}"

                # Extract self-attention weights
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    for name, param in attn.named_parameters():
                        key = f"{layer_prefix}.self_attn.{name}"
                        layer_weights[key] = param.clone().detach()

                # Extract MLP weights
                if hasattr(layer, 'mlp'):
                    mlp = layer.mlp
                    for name, param in mlp.named_parameters():
                        key = f"{layer_prefix}.mlp.{name}"
                        layer_weights[key] = param.clone().detach()

                # Extract layer norms
                if hasattr(layer, 'input_layernorm'):
                    for name, param in layer.input_layernorm.named_parameters():
                        key = f"{layer_prefix}.input_layernorm.{name}"
                        layer_weights[key] = param.clone().detach()

                if hasattr(layer, 'post_attention_layernorm'):
                    for name, param in layer.post_attention_layernorm.named_parameters():
                        key = f"{layer_prefix}.post_attention_layernorm.{name}"
                        layer_weights[key] = param.clone().detach()

            print(f"✓ Extracted {len(layer_weights)} transformer parameters from {max_layers} layers")

            # Print some example keys for debugging
            if layer_weights:
                example_keys = list(layer_weights.keys())[:3]
                print(f"  - Example keys: {example_keys}")

            return layer_weights

        except Exception as e:
            print(f"✗ Error extracting transformer layers: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the pretrained model."""
        info = {}

        if self.pretrained_config is not None:
            info.update({
                'model_type': getattr(self.pretrained_config, 'model_type', 'unknown'),
                'hidden_size': getattr(self.pretrained_config, 'hidden_size', 0),
                'num_layers': getattr(self.pretrained_config, 'num_hidden_layers', 0),
                'num_attention_heads': getattr(self.pretrained_config, 'num_attention_heads', 0),
                'vocab_size': getattr(self.pretrained_config, 'vocab_size', 0),
                'max_position_embeddings': getattr(self.pretrained_config, 'max_position_embeddings', 0),
            })

        # Calculate total parameters
        if self.pretrained_model is not None:
            info['total_parameters'] = sum(p.numel() for p in self.pretrained_model.parameters())
        elif hasattr(self, 'pretrained_state_dict') and self.pretrained_state_dict:
            info['total_parameters'] = sum(p.numel() for p in self.pretrained_state_dict.values() if isinstance(p, torch.Tensor))
            info['state_dict_keys'] = len(self.pretrained_state_dict)

        return info


def load_pretrained_weights_for_g2pw(g2pw_model, pretrained_model_path: str, 
                                    freeze_backbone: bool = False,
                                    load_embeddings: bool = True,
                                    load_transformer: bool = True) -> bool:
    """
    Load pretrained weights into G2PW model.
    
    Args:
        g2pw_model: The G2PW model instance
        pretrained_model_path: Path to pretrained Qwen3 model
        freeze_backbone: Whether to freeze the backbone parameters
        load_embeddings: Whether to load embedding weights
        load_transformer: Whether to load transformer weights
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print("=" * 60)
        print("LOADING PRETRAINED WEIGHTS")
        print("=" * 60)
        
        # Initialize loader
        loader = PretrainedWeightLoader(pretrained_model_path)
        
        # Load pretrained model
        if not loader.load_pretrained_model():
            return False
        
        # Get model info
        info = loader.get_model_info()
        print(f"Pretrained model info:")
        for key, value in info.items():
            print(f"  - {key}: {value}")
        
        # Check if G2PW model uses Qwen3 backbone
        if not hasattr(g2pw_model, 'use_qwen3') or not g2pw_model.use_qwen3:
            print("✗ G2PW model is not configured to use Qwen3 backbone")
            return False
        
        backbone = g2pw_model.backbone
        weights_loaded = 0
        
        # Load embeddings
        if load_embeddings:
            print("\nLoading embeddings...")
            pretrained_embeddings = loader.extract_embeddings()

            if pretrained_embeddings is not None and hasattr(backbone, 'embeddings'):
                try:
                    # Check size compatibility
                    target_shape = backbone.embeddings.weight.shape
                    source_shape = pretrained_embeddings.shape

                    print(f"Embedding shapes: target={target_shape}, source={source_shape}")

                    if target_shape == source_shape:
                        backbone.embeddings.weight.data.copy_(pretrained_embeddings)
                        print(f"✓ Embeddings loaded: {source_shape}")
                        weights_loaded += 1
                    else:
                        print(f"⚠️ Embedding size mismatch: target={target_shape}, source={source_shape}")
                        # Try to load partial embeddings
                        min_vocab = min(target_shape[0], source_shape[0])
                        min_dim = min(target_shape[1], source_shape[1])
                        backbone.embeddings.weight.data[:min_vocab, :min_dim].copy_(
                            pretrained_embeddings[:min_vocab, :min_dim]
                        )
                        print(f"✓ Partial embeddings loaded: [{min_vocab}, {min_dim}]")
                        weights_loaded += 1

                except Exception as e:
                    print(f"✗ Failed to load embeddings: {e}")
            else:
                print(f"⚠️ No embeddings found or backbone has no embeddings layer")
        
        # Load transformer layers
        if load_transformer:
            print("\nLoading transformer layers...")
            layer_weights = loader.extract_transformer_layers()
            
            if layer_weights:
                # Try to load compatible weights
                model_state = backbone.state_dict()
                loaded_keys = []
                
                for key, weight in layer_weights.items():
                    # Try to find matching keys in backbone
                    for model_key in model_state.keys():
                        if _keys_compatible(key, model_key) and model_state[model_key].shape == weight.shape:
                            model_state[model_key].copy_(weight)
                            loaded_keys.append(model_key)
                            break
                
                if loaded_keys:
                    backbone.load_state_dict(model_state)
                    print(f"✓ Loaded {len(loaded_keys)} transformer parameters")
                    weights_loaded += len(loaded_keys)
                else:
                    print("⚠️ No compatible transformer weights found")
        
        # Freeze backbone if requested
        if freeze_backbone:
            print(f"\nFreezing backbone parameters...")
            frozen_params = 0
            total_params = 0

            for name, param in g2pw_model.named_parameters():
                total_params += param.numel()
                if 'backbone' in name:
                    param.requires_grad = False
                    frozen_params += param.numel()

            print(f"✓ Backbone frozen: {frozen_params:,} parameters")
            print(f"✓ Trainable parameters: {total_params - frozen_params:,} parameters")
        
        print("\n" + "=" * 60)
        print(f"PRETRAINED WEIGHT LOADING COMPLETED")
        print(f"✓ Successfully loaded {weights_loaded} weight groups")
        print("=" * 60)
        
        return weights_loaded > 0
        
    except Exception as e:
        print(f"✗ Pretrained weight loading failed: {e}")
        logger.error(f"Pretrained weight loading error: {e}")
        return False


def _keys_compatible(pretrained_key: str, model_key: str) -> bool:
    """Check if two parameter keys are compatible for weight transfer."""
    # Simple heuristic: check if key names contain similar components
    pretrained_parts = pretrained_key.lower().split('.')
    model_parts = model_key.lower().split('.')
    
    # Look for common components
    common_parts = set(pretrained_parts) & set(model_parts)
    return len(common_parts) >= 2  # At least 2 common parts
