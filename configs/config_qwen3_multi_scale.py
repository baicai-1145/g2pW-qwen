"""
Multi-scale Qwen3 Model Configuration
Support for different Qwen3 model sizes and variants.
"""

import os

class MultiScaleConfig:
    """Configuration for multi-scale Qwen3 experiments."""
    
    # Model Paths - All available models
    MODEL_PATHS = {
        # Base models (available)
        'qwen3-0.6b-base': './Qwen3-0.6B-Base',
        'qwen3-1.7b-base': './Qwen3-1.7B-Base',  # In g2pW directory
        'qwen3-4b-base': './Qwen3-4B-Base',      # In g2pW directory (if exists)
    }
    
    # Model-specific configurations
    MODEL_CONFIGS = {
        'qwen3-0.6b-base': {
            'batch_size': 40,
            'gradient_accumulation_steps': 8,
            'effective_batch_size': 320,
            'lr': 5e-5,
            'epochs': 100,
            'memory_efficient': False,
            'recommended_for': ['baseline', 'fast_experiment']
        },
        'qwen3-1.7b-base': {
            'batch_size': 30,
            'gradient_accumulation_steps': 10,
            'effective_batch_size': 300,
            'lr': 3e-5,
            'epochs': 100,
            'memory_efficient': True,
            'recommended_for': ['performance_boost', 'main_experiment']
        },
        'qwen3-4b-base': {
            'batch_size': 16,
            'gradient_accumulation_steps': 16,
            'effective_batch_size': 256,
            'lr': 2e-5,
            'epochs': 60,
            'memory_efficient': True,
            'recommended_for': ['maximum_performance', 'final_model']
        },
        'qwen3-4b-thinking': {
            'batch_size': 12,
            'gradient_accumulation_steps': 20,
            'effective_batch_size': 240,
            'lr': 1e-5,  # Lower LR for thinking model
            'epochs': 40,
            'memory_efficient': True,
            'recommended_for': ['chain_of_thought', 'explainable_g2p']
        }
    }
    
    def __init__(self, model_name='qwen3-1.7b-base'):
        """Initialize with specific model configuration."""
        self.model_name = model_name
        self.model_source = self.MODEL_PATHS[model_name]
        
        # Load model-specific config
        model_config = self.MODEL_CONFIGS[model_name]
        
        # Training Parameters
        self.lr = model_config['lr']
        self.batch_size = model_config['batch_size']
        self.gradient_accumulation_steps = model_config['gradient_accumulation_steps']
        self.effective_batch_size = model_config['effective_batch_size']
        self.epochs = model_config['epochs']
        
        # Memory optimization
        self.memory_efficient = model_config['memory_efficient']
        self.fp16 = True
        self.gradient_checkpointing = model_config['memory_efficient']

        # Optimizer selection
        self.use_muon_optimizer = True  # Use Muon+AdamW combination
        self.muon_momentum = 0.95
        self.muon_ns_steps = 5
        self.adamw_betas = (0.9, 0.999)
        self.adamw_eps = 1e-8
        
        # Model-specific optimizations
        if 'thinking' in model_name:
            self.use_thinking_mode = True
            self.thinking_temperature = 0.7
        else:
            self.use_thinking_mode = False
            
        # Common settings
        self.max_len = 128
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        
        # POS Joint Training
        self.use_pos = True
        self.param_pos = {
            'pos_joint_training': True,
            'weight': 0.1,  # Higher weight for larger models
            'train_pos_path': 'cpp_dataset/train.pos',
            'valid_pos_path': 'cpp_dataset/dev.pos',
            'test_pos_path': 'cpp_dataset/test.pos'
        }
        
        # Conditional Mechanism (paper optimal)
        self.use_conditional = True
        self.param_conditional = {
            'affect_location': 'softmax',
            'bias': True,
            'char-linear': True,      # α_char = 1
            'pos-linear': False,      # α_pos = 0 (paper optimal)
            'char+pos-second': True,  # α_cross = 1
            'char+pos-second_lowrank': False,
            'lowrank_size': 0,
            'char+pos-second_fm': False,
            'fm_size': 0,
            'fix_mode': None,
            'count_json': 'train.count.json'
        }
        
        # Focal Loss
        self.use_focal = True
        self.param_focal = {
            'alpha': 1.0,
            'gamma': 2.0,
            'reduction': 'mean'
        }
        
        # Model loading
        self.load_pretrained = True
        self.freeze_backbone = False
        
        # Output directories (model-specific)
        self.output_dir = f'teacher_model_output_{model_name.replace("-", "_")}'
        self.log_dir = f'teacher_model_logs_{model_name.replace("-", "_")}'
        self.tensorboard_dir = f'runs/teacher_model_{model_name.replace("-", "_")}'
        
        # Training monitoring
        self.val_interval = 2473 * 2  # Every 2 epochs
        self.save_interval = 2473 * 5  # Every 5 epochs
        self.log_interval = 100

        # DataLoader settings
        memory_settings = self.get_memory_optimization_settings()
        self.dataloader_num_workers = memory_settings['dataloader_num_workers']
        self.pin_memory = memory_settings['pin_memory']
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
        # Validation
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration."""
        assert os.path.exists(self.model_source), f"Model path not found: {self.model_source}"
        assert self.effective_batch_size == self.batch_size * self.gradient_accumulation_steps
        assert self.use_pos and self.param_pos['pos_joint_training']
        assert not self.param_conditional['pos-linear']  # Paper optimal
        assert self.param_conditional['char-linear']
        assert self.param_conditional['char+pos-second']
    
    def get_optimizer_params(self, model):
        """Get optimizer parameters with differential learning rates."""
        optimizer_params = []
        
        # Backbone parameters (lower LR)
        backbone_params = []
        for name, param in model.named_parameters():
            if any(backbone_name in name for backbone_name in ['qwen3_model', 'backbone', 'transformer']):
                backbone_params.append(param)
        
        # Task-specific parameters (higher LR)
        task_params = []
        for name, param in model.named_parameters():
            if not any(backbone_name in name for backbone_name in ['qwen3_model', 'backbone', 'transformer']):
                task_params.append(param)
        
        # Different learning rates based on model size
        if '4b' in self.model_name:
            backbone_lr_ratio = 0.1  # Very low LR for large model
        elif '1.7b' in self.model_name:
            backbone_lr_ratio = 0.2  # Medium LR
        else:
            backbone_lr_ratio = 0.5  # Higher LR for small model
        
        optimizer_params = [
            {'params': backbone_params, 'lr': self.lr * backbone_lr_ratio},
            {'params': task_params, 'lr': self.lr}
        ]
        
        return optimizer_params
    
    def get_memory_optimization_settings(self):
        """Get memory optimization settings."""
        return {
            'gradient_checkpointing': self.gradient_checkpointing,
            'fp16': self.fp16,
            'dataloader_num_workers': 2 if self.memory_efficient else 4,
            'pin_memory': not self.memory_efficient,
            'prefetch_factor': 2 if self.memory_efficient else 4
        }

    def get_loss_weights(self, epoch):
        """Get dynamic loss weights for current epoch."""
        # Simple implementation - can be made more sophisticated
        # Could add epoch-based scheduling here if needed
        _ = epoch  # Suppress unused parameter warning
        return {
            'phoneme_weight': 1.0,
            'pos_weight': self.param_pos['weight'],
            'focal_weight': 1.0 if self.use_focal else 0.0
        }
    
    @classmethod
    def get_recommended_model(cls, purpose='performance'):
        """Get recommended model for specific purpose."""
        recommendations = {
            'baseline': 'qwen3-0.6b-base',
            'performance': 'qwen3-1.7b-base', 
            'maximum': 'qwen3-4b-base',
            'thinking': 'qwen3-4b-thinking',
            'fast': 'qwen3-0.6b-base',
            'production': 'qwen3-1.7b-base'
        }
        return recommendations.get(purpose, 'qwen3-1.7b-base')
    
    def print_config_summary(self):
        """Print configuration summary."""
        print(f"{'='*60}")
        print(f"MULTI-SCALE QWEN3 CONFIGURATION")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Path: {self.model_source}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Effective Batch Size: {self.effective_batch_size}")
        print(f"Learning Rate: {self.lr}")
        print(f"Epochs: {self.epochs}")
        print(f"Memory Efficient: {self.memory_efficient}")
        print(f"Thinking Mode: {getattr(self, 'use_thinking_mode', False)}")
        print(f"Output Dir: {self.output_dir}")
        print(f"{'='*60}")

# Create different model configurations
def create_config(model_name):
    """Factory function to create model configuration."""
    return MultiScaleConfig(model_name)

# Pre-defined configurations (only for available models)
config_0_6b = MultiScaleConfig('qwen3-0.6b-base')
config_1_7b = MultiScaleConfig('qwen3-1.7b-base')  # Recommended

# Check if 4B model exists before creating config
import os
if os.path.exists('Qwen3-4B-Base'):
    config_4b = MultiScaleConfig('qwen3-4b-base')
else:
    config_4b = None

# Default configuration (recommended)
config = config_1_7b
