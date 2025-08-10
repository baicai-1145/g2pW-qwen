"""
Qwen3 Teacher Model Configuration for Knowledge Distillation
Optimized configuration based on g2pW paper and BERT comparison results.
"""

import os

class Config(object):
    """Teacher model configuration for optimal performance."""
    
    # Model Architecture
    model_source = './Qwen3-0.6B-Base'
    model_type = 'qwen3'
    
    # Training Parameters - Optimized for teacher model
    lr = 5e-5                    # Paper's learning rate (increased from 3e-5)
    batch_size = 100              # Increased from 32 for better training
    gradient_accumulation_steps = 2  # Simulate batch_size=320 like BERT
    effective_batch_size = 200   # batch_size * gradient_accumulation_steps
    
    max_len = 128
    epochs = 150                  # More epochs for teacher model convergence
    weight_decay = 0.01
    max_grad_norm = 1.0
    
    # Learning Rate Scheduling
    warmup_ratio = 0.1           # 10% warmup steps
    scheduler_type = 'linear'    # Linear decay after warmup
    
    # POS Joint Training - Key for performance improvement
    use_pos = True
    param_pos = {
        'pos_joint_training': True,      # Enable joint training
        'weight': 0.1,                   # β=0.3, increased for better POS accuracy
        'train_pos_path': 'cpp_dataset/train.pos',
        'valid_pos_path': 'cpp_dataset/dev.pos',
        'test_pos_path': 'cpp_dataset/test.pos'
    }
    
    # Conditional Mechanism - Based on paper's optimal settings
    use_conditional = True
    param_conditional = {
        'affect_location': 'softmax',    # Apply to softmax layer
        'bias': True,                    # Enable bias term
        'char-linear': True,             # α_char = 1 (paper optimal)
        'pos-linear': False,             # α_pos = 0 (paper optimal!)
        'char+pos-second': True,         # α_cross = 1 (char⊗pos interaction)
        'char+pos-second_lowrank': False, # Disable for teacher (full rank)
        'lowrank_size': 0,
        'char+pos-second_fm': False,     # Disable factorization machine
        'fm_size': 0,
        'fix_mode': None,
        'count_json': 'train.count.json'
    }
    
    # Focal Loss - For handling difficult samples
    use_focal = True
    param_focal = {
        'alpha': 1,                      # Class weighting
        'gamma': 2                       # Focusing parameter
    }
    
    # Pretrained Weights
    load_pretrained = True
    freeze_backbone = False              # Allow fine-tuning for teacher
    
    # Training Monitoring (in steps)
    val_interval = 792                  # Validate every epoch (~2473 steps)
    save_interval = 1584                 # Save checkpoint every 2 epochs (~4946 steps)
    log_interval = 10                   # Log every 100 steps (reasonable frequency)
    
    # Mixed Precision Training
    fp16 = True                          # Enable for memory efficiency

    # Qwen3 Specific Optimizations
    use_rope_scaling = True              # Utilize RoPE position encoding
    attention_dropout = 0.1              # Enable attention dropout for regularization
    hidden_dropout = 0.1                 # Enable hidden dropout for regularization
    
    # Data Processing
    dataloader_num_workers = 4
    pin_memory = True
    
    # Output Directories
    output_dir = 'teacher_model_output_v2'
    log_dir = 'teacher_model_logs_v2'
    tensorboard_dir = 'runs/teacher_model_v2'  # TensorBoard logs
    
    # Early Stopping
    early_stopping_patience = 5         # Stop if no improvement for 5 validations
    early_stopping_threshold = 0.001    # Minimum improvement threshold
    
    # Target Performance
    target_accuracy = 0.90               # Target 90%+ accuracy for teacher
    
    def __init__(self):
        """Initialize configuration and validate settings."""
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        # Validate configuration
        assert self.effective_batch_size == self.batch_size * self.gradient_accumulation_steps
        assert self.use_pos and self.param_pos['pos_joint_training'], "POS joint training must be enabled for teacher"
        assert not self.param_conditional['pos-linear'], "POS linear transformation must be disabled (paper optimal: α_pos=0)"
        assert self.param_conditional['char-linear'], "Character linear transformation must be enabled (paper optimal: α_char=1)"
        assert self.param_conditional['char+pos-second'], "Character-POS interaction must be enabled (paper optimal: α_cross=1)"
        
    def get_optimizer_params(self, model):
        """Get optimized parameters for different model components."""
        # Different learning rates for different components
        backbone_params = []
        classifier_params = []
        pos_params = []
        
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            elif 'pos' in name:
                pos_params.append(param)
            else:
                classifier_params.append(param)
        
        return [
            {'params': backbone_params, 'lr': self.lr * 0.1},      # Lower LR for backbone
            {'params': classifier_params, 'lr': self.lr},          # Standard LR for classifier
            {'params': pos_params, 'lr': self.lr * 0.5}            # Medium LR for POS
        ]
    
    def get_loss_weights(self, epoch):
        """Dynamic loss weighting during training."""
        # Gradually increase POS weight
        pos_weight = min(self.param_pos['weight'], self.param_pos['weight'] * epoch / 5)
        
        # Gradually increase focal loss weight
        focal_weight = 1.0 if epoch > 3 else 0.5
        
        return {
            'task_weight': 1.0,
            'pos_weight': pos_weight,
            'focal_weight': focal_weight
        }

# Create default config instance
config = Config()

# Validation
if __name__ == "__main__":
    print("Teacher Model Configuration:")
    print(f"  - Model: {config.model_source}")
    print(f"  - Effective batch size: {config.effective_batch_size}")
    print(f"  - Learning rate: {config.lr}")
    print(f"  - POS joint training: {config.param_pos['pos_joint_training']}")
    print(f"  - POS weight: {config.param_pos['weight']}")
    print(f"  - Focal loss: {config.use_focal}")
    print(f"  - Target accuracy: {config.target_accuracy}")
    print("Configuration validated successfully!")
