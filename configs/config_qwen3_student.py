"""
Qwen3 Student Model Configuration for Knowledge Distillation
Lightweight configuration optimized for efficiency and performance.
"""

import os

class Config:
    """Student model configuration for knowledge distillation."""
    
    # Model Architecture - Compressed
    model_source = './Qwen3-0.6B-Base'  # Use same tokenizer
    model_type = 'qwen3_lite'
    
    # Compressed Architecture Parameters
    hidden_size = 768                    # Reduced from 1024
    num_attention_heads = 12             # Reduced from 16
    num_hidden_layers = 12               # Reduced from 28
    intermediate_size = 2048             # Reduced from 3072
    
    # Training Parameters - Optimized for student
    lr = 3e-5                           # Slightly higher for student
    batch_size = 64                     # Larger batch size possible
    gradient_accumulation_steps = 4     # Simulate batch_size=256
    effective_batch_size = 256
    
    max_len = 128
    epochs = 20                         # More epochs for distillation
    weight_decay = 0.01
    max_grad_norm = 1.0
    
    # Learning Rate Scheduling
    warmup_ratio = 0.1
    scheduler_type = 'linear'
    
    # POS Joint Training - Inherited from teacher
    use_pos = True
    param_pos = {
        'pos_joint_training': True,
        'weight': 0.1,                  # Same as teacher
        'train_pos_path': 'cpp_dataset/train.pos',
        'valid_pos_path': 'cpp_dataset/dev.pos',
        'test_pos_path': 'cpp_dataset/test.pos'
    }
    
    # Conditional Mechanism - Simplified for student
    use_conditional = True
    param_conditional = {
        'affect_location': 'softmax',
        'bias': True,
        'char-linear': True,
        'pos-linear': True,
        'char+pos-second': True,
        'char+pos-second_lowrank': True,  # Use low-rank for efficiency
        'lowrank_size': 64,               # Compressed interaction
        'char+pos-second_fm': False,
        'fm_size': 0,
        'fix_mode': None,
        'count_json': 'train.count.json'
    }
    
    # Focal Loss
    use_focal = True
    param_focal = {
        'alpha': 1,
        'gamma': 2
    }
    
    # Pretrained Weights - Initialize from teacher
    load_pretrained = False              # Will be initialized from teacher
    freeze_backbone = False
    
    # Distillation Parameters
    distillation = {
        'temperature': 4.0,              # Softmax temperature
        'alpha': 0.7,                    # Weight for distillation loss
        'beta': 0.3,                     # Weight for task loss
        'feature_distill_weight': 0.5,   # Feature distillation weight
        'pos_distill_weight': 0.1,       # POS distillation weight
    }
    
    # Training Monitoring
    val_interval = 200
    save_interval = 1000
    log_interval = 50
    
    # Mixed Precision Training
    fp16 = True
    
    # Data Processing
    dataloader_num_workers = 4
    pin_memory = True
    
    # Output Directories
    output_dir = 'student_model_output'
    log_dir = 'student_model_logs'
    
    # Early Stopping
    early_stopping_patience = 8         # More patience for distillation
    early_stopping_threshold = 0.001
    
    # Target Performance
    target_accuracy = 0.95               # Target 95%+ accuracy (close to BERT)
    compression_ratio = 0.5              # 50% parameter reduction
    
    def __post_init__(self):
        """Post-initialization setup."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Validate distillation parameters
        assert abs(self.distillation['alpha'] + self.distillation['beta'] - 1.0) < 1e-6
        assert self.batch_size > 32, "Student model should support larger batch size"
        
    def get_model_config(self):
        """Get compressed model configuration."""
        return {
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'num_hidden_layers': self.num_hidden_layers,
            'intermediate_size': self.intermediate_size,
            'max_position_embeddings': 512,
            'vocab_size': 151936,  # Qwen3 vocab size
            'layer_norm_eps': 1e-6,
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
        }
    
    def get_distillation_loss_weights(self, epoch):
        """Dynamic loss weighting for distillation."""
        # Gradually shift from distillation to task loss
        progress = min(epoch / 10, 1.0)  # 10 epochs to full task focus
        
        distill_weight = self.distillation['alpha'] * (1 - progress * 0.3)
        task_weight = self.distillation['beta'] + progress * 0.3
        
        return {
            'distillation_weight': distill_weight,
            'task_weight': task_weight,
            'feature_weight': self.distillation['feature_distill_weight'],
            'pos_weight': self.distillation['pos_distill_weight']
        }
    
    def estimate_parameters(self):
        """Estimate student model parameters."""
        # Rough estimation based on architecture
        embedding_params = 151936 * self.hidden_size  # Vocab * hidden
        transformer_params = (
            self.num_hidden_layers * (
                # Self-attention
                4 * self.hidden_size * self.hidden_size +
                # Feed-forward
                2 * self.hidden_size * self.intermediate_size +
                # Layer norms
                2 * self.hidden_size
            )
        )
        classifier_params = self.hidden_size * 1000  # Rough estimate
        
        total_params = embedding_params + transformer_params + classifier_params
        return total_params
    
    def get_memory_estimate(self):
        """Estimate memory usage."""
        params = self.estimate_parameters()
        # Rough memory estimation (parameters + gradients + optimizer states)
        memory_gb = params * 4 * 3 / (1024**3)  # 4 bytes per param, 3x for training
        return memory_gb

# Create default config instance
config = Config()

# Validation and info
if __name__ == "__main__":
    print("Student Model Configuration:")
    print(f"  - Architecture: {config.hidden_size}H-{config.num_attention_heads}A-{config.num_hidden_layers}L")
    print(f"  - Estimated parameters: {config.estimate_parameters()/1e6:.1f}M")
    print(f"  - Compression ratio: {config.compression_ratio:.1%}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Effective batch size: {config.effective_batch_size}")
    print(f"  - Distillation temperature: {config.distillation['temperature']}")
    print(f"  - Target accuracy: {config.target_accuracy}")
    print(f"  - Estimated memory: {config.get_memory_estimate():.1f}GB")
    print("Configuration validated successfully!")
