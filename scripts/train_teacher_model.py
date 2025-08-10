"""
Teacher Model Training Script with POS Joint Training
Based on g2pW paper implementation with Qwen3 backbone.
"""

import sys
import os
import torch
import time
import json
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Add paths
sys.path.insert(0, '.')
sys.path.insert(0, 'g2pw')
sys.path.insert(0, 'configs')

def load_complete_dataset():
    """Load the complete CPP dataset."""
    print("Loading complete CPP dataset...")
    
    try:
        # Load training data
        train_texts = []
        train_labels = []
        train_pos_tags = []
        
        with open('cpp_dataset/train.sent', 'r', encoding='utf-8') as f:
            train_texts = [line.strip() for line in f]
        
        with open('cpp_dataset/train.lb', 'r', encoding='utf-8') as f:
            train_labels = [line.strip() for line in f]
        
        with open('cpp_dataset/train.pos', 'r', encoding='utf-8') as f:
            train_pos_tags = [line.strip() for line in f]
        
        # Load validation data
        valid_texts = []
        valid_labels = []
        valid_pos_tags = []
        
        with open('cpp_dataset/dev.sent', 'r', encoding='utf-8') as f:
            valid_texts = [line.strip() for line in f]
        
        with open('cpp_dataset/dev.lb', 'r', encoding='utf-8') as f:
            valid_labels = [line.strip() for line in f]
        
        with open('cpp_dataset/dev.pos', 'r', encoding='utf-8') as f:
            valid_pos_tags = [line.strip() for line in f]
        
        # Find query positions
        def get_query_ids(texts):
            query_ids = []
            for text in texts:
                anchor_pos = text.find('▁')
                if anchor_pos != -1 and anchor_pos + 1 < len(text):
                    query_ids.append(anchor_pos + 1)
                else:
                    query_ids.append(0)
            return query_ids
        
        train_query_ids = get_query_ids(train_texts)
        valid_query_ids = get_query_ids(valid_texts)
        
        print(f"✓ Dataset loaded:")
        print(f"  - Training samples: {len(train_texts)}")
        print(f"  - Validation samples: {len(valid_texts)}")
        
        return {
            'train_texts': train_texts,
            'train_labels': train_labels,
            'train_pos_tags': train_pos_tags,
            'train_query_ids': train_query_ids,
            'valid_texts': valid_texts,
            'valid_labels': valid_labels,
            'valid_pos_tags': valid_pos_tags,
            'valid_query_ids': valid_query_ids
        }
        
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return None

def prepare_labels_and_chars():
    """Prepare labels and characters from polyphonic chars file."""
    print("Preparing labels and characters...")
    
    polyphonic_chars = []
    with open('cpp_dataset/POLYPHONIC_CHARS.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                polyphonic_chars.append((parts[0], parts[1]))
    
    labels = sorted(list(set([phoneme for char, phoneme in polyphonic_chars])))
    chars = sorted(list(set([char for char, phoneme in polyphonic_chars])))
    
    char2phonemes = {}
    for char, phoneme in polyphonic_chars:
        if char not in char2phonemes:
            char2phonemes[char] = []
        if phoneme in labels:
            char2phonemes[char].append(labels.index(phoneme))
    
    print(f"✓ Labels and characters prepared:")
    print(f"  - Labels: {len(labels)}")
    print(f"  - Characters: {len(chars)}")
    
    return labels, chars, char2phonemes

def create_teacher_model(labels, chars):
    """Create optimized teacher model with POS joint training."""
    print("Creating Qwen3 teacher model with POS joint training...")
    
    from g2pw.module import G2PW
    from transformers import AutoTokenizer
    from configs.config_qwen3_multi_scale import config_1_7b as config
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_source, trust_remote_code=True)
    
    # Real POS tags from dataset
    pos_tags = ['A', 'C', 'D', 'DE', 'I', 'N', 'P', 'T', 'UNK', 'V']
    
    # Create teacher model with optimized configuration
    model = G2PW.create_qwen3_model(
        qwen3_model_path=config.model_source,
        labels=labels,
        chars=chars,
        pos_tags=pos_tags,
        use_conditional=config.use_conditional,
        param_conditional=config.param_conditional,
        use_focal=config.use_focal,
        param_focal=config.param_focal,
        use_pos=config.use_pos,
        param_pos=config.param_pos,
        load_pretrained=config.load_pretrained,
        freeze_backbone=config.freeze_backbone
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Teacher model created:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - POS joint training: {config.param_pos['pos_joint_training']}")
    print(f"  - POS weight: {config.param_pos['weight']}")
    print(f"  - Focal loss: {config.use_focal}")
    
    return model, tokenizer, config

def create_datasets_and_loaders(data, labels, chars, char2phonemes, tokenizer, config):
    """Create datasets and data loaders."""
    print("Creating datasets and data loaders...")
    
    from g2pw.dataset_enhanced import TextDatasetEnhanced
    from g2pw.batch_processor import create_mini_batch_optimized
    from torch.utils.data import DataLoader
    
    # Create datasets with POS support
    train_dataset = TextDatasetEnhanced(
        tokenizer=tokenizer,
        labels=labels,
        char2phonemes=char2phonemes,
        chars=chars,
        texts=data['train_texts'],
        query_ids=data['train_query_ids'],
        phonemes=data['train_labels'],
        pos_tags=data['train_pos_tags'],
        use_mask=True,  # 🔧 Enable phoneme mask!
        use_pos=config.use_pos,
        max_len=config.max_len,
        for_train=True
    )
    
    valid_dataset = TextDatasetEnhanced(
        tokenizer=tokenizer,
        labels=labels,
        char2phonemes=char2phonemes,
        chars=chars,
        texts=data['valid_texts'],
        query_ids=data['valid_query_ids'],
        phonemes=data['valid_labels'],
        pos_tags=data['valid_pos_tags'],
        use_mask=True,  # 🔧 Enable phoneme mask!
        use_pos=config.use_pos,
        max_len=config.max_len,
        for_train=True
    )
    
    def collate_fn(samples):
        return create_mini_batch_optimized(samples, tokenizer, max_len=config.max_len)
    
    # Create data loaders with gradient accumulation support
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.pin_memory
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.pin_memory
    )
    
    print(f"✓ Datasets and loaders created:")
    print(f"  - Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  - Validation: {len(valid_dataset)} samples, {len(valid_loader)} batches")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Effective batch size: {config.effective_batch_size}")
    
    return train_loader, valid_loader

def train_teacher_model(model, train_loader, valid_loader, config):
    """Train the teacher model with POS joint training."""
    print(f"\nStarting teacher model training...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Training on: {device}")
    
    # Setup optimizer - use Muon+AdamW combination
    if hasattr(config, 'use_muon_optimizer') and config.use_muon_optimizer:
        from g2pw.optimizers import G2PMuonAdamW
        optimizer = G2PMuonAdamW(
            model,
            lr=config.lr,
            weight_decay=config.weight_decay,
            muon_momentum=config.muon_momentum,
            muon_ns_steps=config.muon_ns_steps,
            adamw_betas=config.adamw_betas,
            adamw_eps=config.adamw_eps,
            verbose=True
        )
        print("✓ Using Muon+AdamW combined optimizer")
    else:
        # Fallback to standard AdamW with different learning rates
        optimizer_params = config.get_optimizer_params(model)
        optimizer = torch.optim.AdamW(
            optimizer_params,
            weight_decay=config.weight_decay,
            eps=1e-8
        )
        print("✓ Using standard AdamW optimizer")
    
    # Learning rate scheduler
    from transformers import get_linear_schedule_with_warmup
    num_training_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    num_warmup_steps = int(config.warmup_ratio * num_training_steps)

    # Handle different optimizer types
    if hasattr(optimizer, 'muon_optimizer'):
        # For G2PMuonAdamW, we need to create a custom scheduler
        class CombinedScheduler:
            def __init__(self, optimizer, num_warmup_steps, num_training_steps):
                self.optimizer = optimizer
                self.muon_scheduler = get_linear_schedule_with_warmup(
                    optimizer.muon_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )
                self.adamw_scheduler = get_linear_schedule_with_warmup(
                    optimizer.adamw_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )

            def step(self):
                self.muon_scheduler.step()
                self.adamw_scheduler.step()

            def get_last_lr(self):
                return self.muon_scheduler.get_last_lr()

        scheduler = CombinedScheduler(optimizer, num_warmup_steps, num_training_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    print(f"✓ Training setup:")
    print(f"  - Total steps: {num_training_steps}")
    print(f"  - Warmup steps: {num_warmup_steps}")
    print(f"  - Gradient accumulation: {config.gradient_accumulation_steps}")
    
    # Training tracking
    best_accuracy = 0.0
    training_history = []

    # Create output directory and TensorBoard writer
    os.makedirs(config.output_dir, exist_ok=True)
    writer = SummaryWriter(config.tensorboard_dir)

    print(f"✓ TensorBoard logging to: {config.tensorboard_dir}")
    print(f"  Run: tensorboard --logdir={config.tensorboard_dir}")
    
    # Training loop
    for epoch in range(config.epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{config.epochs}")
        print(f"{'='*60}")
        
        # Get dynamic loss weights
        loss_weights = config.get_loss_weights(epoch)
        
        # Training phase
        model.train()
        epoch_loss = 0
        epoch_pos_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                phoneme_mask = batch['phoneme_mask'].to(device)
                char_ids = batch['char_ids'].to(device)
                position_ids = batch['position_ids'].to(device)
                label_ids = batch['label_ids'].to(device)
                pos_ids = batch.get('pos_ids')
                if pos_ids is not None:
                    pos_ids = pos_ids.to(device)
                
                # Forward pass
                probs, loss, pos_logits = model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    phoneme_mask=phoneme_mask,
                    char_ids=char_ids,
                    position_ids=position_ids,
                    label_ids=label_ids,
                    pos_ids=pos_ids
                )
                
                # Scale loss for gradient accumulation
                loss = loss / config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * config.gradient_accumulation_steps
                num_batches += 1

                # Log to TensorBoard every 50 batches
                if batch_idx % 50 == 0:
                    global_step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar('Training/BatchLoss', loss.item() * config.gradient_accumulation_steps, global_step)
                    writer.add_scalar('Training/LearningRate', optimizer.param_groups[0]['lr'], global_step)

                # Update progress bar
                if hasattr(optimizer, 'get_lr'):
                    current_lr = optimizer.get_lr()
                elif hasattr(optimizer, 'param_groups'):
                    current_lr = optimizer.param_groups[0]['lr']
                else:
                    current_lr = config.lr

                progress_bar.set_postfix({
                    'loss': f'{loss.item() * config.gradient_accumulation_steps:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue
        
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        
        # Log epoch metrics to TensorBoard
        writer.add_scalar('Training/EpochLoss', avg_epoch_loss, epoch)
        writer.add_scalar('Training/EpochLR', optimizer.param_groups[0]['lr'], epoch)

        # Get current learning rate
        if hasattr(optimizer, 'get_lr'):
            current_lr = optimizer.get_lr()
        elif hasattr(optimizer, 'param_groups'):
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = config.lr

        print(f"\nEpoch {epoch + 1} Training Summary:")
        print(f"  - Average Loss: {avg_epoch_loss:.4f}")
        print(f"  - Learning Rate: {current_lr:.2e}")

        # Save checkpoint
        if (epoch + 1) % 2 == 0:  # Save every 2 epochs
            checkpoint_path = os.path.join(config.output_dir, f'teacher_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy,
                'training_history': training_history
            }, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    # Close TensorBoard writer
    writer.close()

    print(f"\n{'='*60}")
    print("TEACHER MODEL TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"✓ Training completed successfully")
    print(f"✓ Model saved to: {config.output_dir}")
    print(f"✓ TensorBoard logs saved to: {config.tensorboard_dir}")

    return model

def main():
    """Main training function."""
    print("=" * 60)
    print("QWEN3 TEACHER MODEL TRAINING WITH POS JOINT TRAINING")
    print("=" * 60)
    print(f"Training started at: {datetime.now()}")
    
    # Load dataset
    data = load_complete_dataset()
    if data is None:
        return
    
    # Prepare labels and characters
    labels, chars, char2phonemes = prepare_labels_and_chars()
    
    # Create teacher model
    model, tokenizer, config = create_teacher_model(labels, chars)
    
    # Create datasets and loaders
    train_loader, valid_loader = create_datasets_and_loaders(
        data, labels, chars, char2phonemes, tokenizer, config
    )
    
    # Train teacher model
    trained_model = train_teacher_model(model, train_loader, valid_loader, config)
    
    print(f"\n🎉 TEACHER MODEL TRAINING COMPLETED!")
    print(f"Training completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
