"""
Configuration for g2pW with Qwen3-0.6B-Base backbone.
Based on the original config_cpp.py but adapted for Qwen3.
"""

import os

# Model configuration
model_source = './Qwen3-0.6B-Base'  # Path to Qwen3 model
use_qwen3 = True  # Enable Qwen3 backbone

# Training configuration
lr = 1e-5  # Lower learning rate for larger model
batch_size = 8  # Smaller batch size due to larger model
max_len = 512  # Can use longer sequences with Qwen3
epochs = 10
warmup_steps = 1000

# Model architecture
hidden_size = 1024  # Qwen3 hidden size
num_attention_heads = 16  # Qwen3 attention heads
num_hidden_layers = 28  # Qwen3 layers (though we use fewer in practice)

# Conditional mechanism configuration
use_conditional = True
param_conditional = {
    'affect_location': 'softmax',
    'bias': True,
    'char-linear': True,
    'pos-linear': True,
    'char+pos-second': True,
    'char+pos-second_lowrank': False,
    'lowrank_size': 0,
    'char+pos-second_fm': False,
    'fm_size': 0,
    'fix_mode': None,
    'count_json': 'train.count.json'
}

# Focal loss configuration
use_focal = False
param_focal = {
    'alpha': 1,
    'gamma': 2
}

# POS tagging configuration
use_pos = True
param_pos = {
    'pos_joint_training': True,
    'train_pos_path': 'cpp_dataset/train.pos',
    'valid_pos_path': 'cpp_dataset/dev.pos',
    'test_pos_path': 'cpp_dataset/test.pos'
}

# Dataset configuration
dataset_name = 'cpp'
train_sent_path = 'cpp_dataset/train.sent'
train_lb_path = 'cpp_dataset/train.lb'
valid_sent_path = 'cpp_dataset/dev.sent'
valid_lb_path = 'cpp_dataset/dev.lb'
test_sent_path = 'cpp_dataset/test.sent'
test_lb_path = 'cpp_dataset/test.lb'
polyphonic_chars_path = 'cpp_dataset/POLYPHONIC_CHARS.txt'

# Training settings
window_size = 0  # No windowing for now
max_window_size = 32
stride = 16
seed = 42
device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'

# Optimization settings
weight_decay = 0.01
adam_epsilon = 1e-8
max_grad_norm = 1.0
gradient_accumulation_steps = 1

# Scheduler settings
scheduler_type = 'linear'
num_warmup_steps = warmup_steps

# Logging and saving
logging_steps = 100
save_steps = 1000
eval_steps = 500
save_total_limit = 3
load_best_model_at_end = True
metric_for_best_model = 'eval_accuracy'
greater_is_better = True

# Output directories
output_dir = 'outputs/qwen3_cpp_ops'
logging_dir = f'{output_dir}/logs'
cache_dir = f'{output_dir}/cache'

# Model saving
save_model = True
model_save_path = f'{output_dir}/best_model'

# Evaluation settings
do_eval = True
evaluation_strategy = 'steps'
eval_accumulation_steps = None

# Prediction settings
do_predict = True
predict_with_generate = False

# Mixed precision training
fp16 = True  # Enable for faster training
fp16_opt_level = 'O1'

# Data loading
dataloader_num_workers = 4
dataloader_pin_memory = True
remove_unused_columns = False

# Tokenizer settings
tokenizer_name = model_source
use_fast_tokenizer = True
tokenizer_cache_dir = cache_dir

# Special tokens (Qwen3 specific)
pad_token = '<|endoftext|>'  # Qwen3 uses this as pad token
unk_token = '<|endoftext|>'
bos_token = '<|endoftext|>'
eos_token = '<|endoftext|>'

# Model specific settings
model_type = 'qwen3'
problem_type = 'single_label_classification'

# Qwen3 specific configurations
qwen3_config = {
    'vocab_size': 151936,
    'hidden_size': 1024,
    'num_hidden_layers': 28,
    'num_attention_heads': 16,
    'intermediate_size': 3072,
    'max_position_embeddings': 32768,
    'pad_token_id': 151643,
    'use_cache': False,  # Disable for training
    'output_attentions': False,
    'output_hidden_states': False,
    'return_dict': True
}

# Training stability settings
gradient_checkpointing = True  # Save memory for large model
dataloader_drop_last = True
ignore_data_skip = False

# Reproducibility
set_seed = True
seed_value = seed

# Debugging
debug = False
debug_samples = 100

# Performance monitoring
log_level = 'info'
disable_tqdm = False
report_to = ['tensorboard']

# Early stopping
early_stopping_patience = 3
early_stopping_threshold = 0.001

print(f"Qwen3 configuration loaded:")
print(f"  - Model source: {model_source}")
print(f"  - Hidden size: {hidden_size}")
print(f"  - Batch size: {batch_size}")
print(f"  - Learning rate: {lr}")
print(f"  - Max length: {max_len}")
print(f"  - Output dir: {output_dir}")
print(f"  - Use Qwen3: {use_qwen3}")
