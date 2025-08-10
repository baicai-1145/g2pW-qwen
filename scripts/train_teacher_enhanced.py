"""
Enhanced Teacher Model Training with Qwen3 Optimizations
Based on paper analysis and Qwen3 specific improvements.
"""

import sys
import os
import torch
import time
import json
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# Add paths
sys.path.insert(0, '.')
sys.path.insert(0, 'g2pw')
sys.path.insert(0, 'configs')

def create_enhanced_teacher_model(labels, chars):
    """Create enhanced teacher model with Qwen3 optimizations."""
    print("Creating Enhanced Qwen3 teacher model...")
    
    from g2pw.module import G2PW
    from transformers import AutoTokenizer
    from configs.config_qwen3_teacher import config
    
    # Load tokenizer with Qwen3 optimizations
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_source, 
        trust_remote_code=True,
        use_fast=True  # Use fast tokenizer for Qwen3
    )
    
    # Real POS tags from dataset
    pos_tags = ['A', 'C', 'D', 'DE', 'I', 'N', 'P', 'T', 'UNK', 'V']
    
    # Create enhanced model
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
    
    # Apply Qwen3 specific optimizations
    if hasattr(config, 'attention_dropout'):
        for module in model.modules():
            if hasattr(module, 'attention_dropout'):
                module.attention_dropout = config.attention_dropout
            if hasattr(module, 'hidden_dropout'):
                module.hidden_dropout = config.hidden_dropout

    # Integrate RoPE enhancements
    if hasattr(config, 'use_rope_scaling') and config.use_rope_scaling:
        from g2pw.rope_enhanced import integrate_rope_into_g2pw
        model = integrate_rope_into_g2pw(model, config)
        print("✓ RoPE position encoding enhancements integrated")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Enhanced teacher model created:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Learning rate: {config.lr}")
    print(f"  - Batch size: {config.effective_batch_size}")
    print(f"  - POS joint training: {config.param_pos['pos_joint_training']}")
    print(f"  - POS weight: {config.param_pos['weight']}")
    
    return model, tokenizer, config

def enhanced_training_loop(model, train_loader, valid_loader, config):
    """Enhanced training loop with Qwen3 optimizations."""
    print(f"\nStarting enhanced teacher model training...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Training on: {device}")
    
    # Enhanced optimizer with different learning rates
    optimizer_params = []
    
    # Backbone (Qwen3) with lower learning rate
    backbone_params = []
    for name, param in model.named_parameters():
        if 'qwen3_model' in name or 'backbone' in name:
            backbone_params.append(param)
    
    # Task-specific layers with higher learning rate
    task_params = []
    for name, param in model.named_parameters():
        if 'qwen3_model' not in name and 'backbone' not in name:
            task_params.append(param)
    
    optimizer_params = [
        {'params': backbone_params, 'lr': config.lr * 0.1},  # Lower LR for backbone
        {'params': task_params, 'lr': config.lr}  # Higher LR for task layers
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_params,
        weight_decay=config.weight_decay,
        eps=1e-8,
        betas=(0.9, 0.999)  # Optimized for Qwen3
    )
    
    # Enhanced learning rate scheduler
    from transformers import get_cosine_schedule_with_warmup
    num_training_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if config.fp16 else None
    
    # TensorBoard
    writer = SummaryWriter(config.tensorboard_dir)
    
    print(f"✓ Enhanced training setup:")
    print(f"  - Total steps: {num_training_steps}")
    print(f"  - Warmup steps: {num_warmup_steps}")
    print(f"  - Cosine annealing scheduler")
    print(f"  - Mixed precision: {config.fp16}")
    print(f"  - Differential learning rates")
    
    # Training tracking
    best_accuracy = 0.0
    patience = 10
    patience_counter = 0
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(config.epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{config.epochs}")
        print(f"{'='*60}")
        
        # Training phase
        model.train()
        epoch_loss = 0
        epoch_phoneme_loss = 0
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
                
                # Mixed precision forward pass
                with torch.cuda.amp.autocast(enabled=config.fp16):
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
                
                # Backward pass with mixed precision
                if config.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    if config.fp16:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * config.gradient_accumulation_steps
                num_batches += 1
                
                # Log to TensorBoard
                if batch_idx % config.log_interval == 0:
                    global_step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar('Training/BatchLoss', loss.item() * config.gradient_accumulation_steps, global_step)
                    writer.add_scalar('Training/LearningRate', scheduler.get_last_lr()[0], global_step)
                
                # Update progress bar
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * config.gradient_accumulation_steps:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue
        
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        
        # Log epoch metrics
        writer.add_scalar('Training/EpochLoss', avg_epoch_loss, epoch)
        writer.add_scalar('Training/EpochLR', scheduler.get_last_lr()[0], epoch)
        
        print(f"\nEpoch {epoch + 1} Training Summary:")
        print(f"  - Average Loss: {avg_epoch_loss:.4f}")
        print(f"  - Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        
        # Validation every few epochs
        if (epoch + 1) % 5 == 0:
            print("Running validation...")
            val_accuracy = evaluate_model_quick(model, valid_loader, device)
            writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
            
            print(f"  - Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
            
            # Early stopping
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
                
                # Save best model
                best_model_path = os.path.join(config.output_dir, 'best_teacher_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_accuracy': best_accuracy,
                    'config': config
                }, best_model_path)
                print(f"  ✓ New best model saved: {val_accuracy*100:.2f}%")
            else:
                patience_counter += 1
                print(f"  - No improvement ({patience_counter}/{patience})")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(config.output_dir, f'teacher_enhanced_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy
            }, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            break
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"\n{'='*60}")
    print("ENHANCED TEACHER MODEL TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"✓ Training completed successfully")
    print(f"✓ Best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"✓ Model saved to: {config.output_dir}")
    
    return model

def evaluate_model_quick(model, valid_loader, device):
    """Quick validation during training."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in valid_loader:
            if total >= 1000:  # Quick validation on 1000 samples
                break
                
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
            
            probs, _, _ = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                phoneme_mask=phoneme_mask,
                char_ids=char_ids,
                position_ids=position_ids,
                label_ids=label_ids,
                pos_ids=pos_ids
            )
            
            predictions = torch.argmax(probs, dim=-1)
            correct += (predictions == label_ids).sum().item()
            total += label_ids.size(0)
    
    model.train()
    return correct / total if total > 0 else 0.0

def main():
    """Main enhanced training function."""
    print("=" * 60)
    print("ENHANCED QWEN3 TEACHER MODEL TRAINING")
    print("=" * 60)
    print(f"Training started at: {datetime.now()}")
    
    # Load dataset (reuse existing functions)
    from scripts.train_teacher_model import load_complete_dataset, prepare_labels_and_chars, create_datasets_and_loaders
    
    # Load dataset
    data = load_complete_dataset()
    if data is None:
        return
    
    # Prepare labels and characters
    labels, chars, char2phonemes = prepare_labels_and_chars()
    
    # Create enhanced teacher model
    model, tokenizer, config = create_enhanced_teacher_model(labels, chars)
    
    # Create datasets and loaders
    train_loader, valid_loader = create_datasets_and_loaders(
        data, labels, chars, char2phonemes, tokenizer, config
    )
    
    # Enhanced training
    trained_model = enhanced_training_loop(model, train_loader, valid_loader, config)
    
    print(f"\n🎉 ENHANCED TEACHER MODEL TRAINING COMPLETED!")
    print(f"Training completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
