"""
Training script for g2pW with Qwen3 support.
Enhanced version of train_g2p_bert.py with Qwen3 optimizations.
"""

import sys
sys.path.insert(0, './')

import os
import argparse
from datetime import datetime
import random
from shutil import copyfile
from collections import defaultdict
import itertools
import statistics
import warnings

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import enhanced modules
from g2pw.dataset import prepare_data, prepare_pos, get_phoneme_labels, get_char_phoneme_labels
from g2pw.dataset_enhanced import TextDatasetEnhanced
from g2pw.batch_processor import BatchProcessor, create_mini_batch_optimized
from g2pw.module import G2PW
from g2pw.utils import get_model_config, RunningAverage, get_logger


def setup_training_environment(config):
    """Setup training environment with proper device and memory management."""
    # Set device
    if torch.cuda.is_available() and config.device != 'cpu':
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Set random seeds for reproducibility
    if hasattr(config, 'seed'):
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
    
    # Enable mixed precision if configured
    scaler = None
    if hasattr(config, 'fp16') and config.fp16 and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("Mixed precision training enabled")
    
    return device, scaler


def create_qwen3_model(config, labels, chars):
    """Create G2PW model with Qwen3 backbone."""
    print("Creating G2PW model with Qwen3 backbone...")
    
    model = G2PW.create_qwen3_model(
        qwen3_model_path=config.model_source,
        labels=labels,
        chars=chars,
        pos_tags=TextDatasetEnhanced.POS_TAGS,
        use_conditional=config.use_conditional,
        param_conditional=config.param_conditional,
        use_focal=config.use_focal,
        param_focal=config.param_focal if hasattr(config, 'param_focal') else None,
        use_pos=config.use_pos,
        param_pos=config.param_pos if hasattr(config, 'param_pos') else None
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def create_optimized_dataloader(dataset, config, tokenizer, is_training=True):
    """Create optimized DataLoader with batch processor."""
    # Create batch processor
    batch_processor = BatchProcessor(tokenizer, max_len=config.max_len)
    
    # Optimize batch size if not specified
    if hasattr(config, 'auto_batch_size') and config.auto_batch_size:
        optimal_batch_size = batch_processor.optimize_batch_size(
            len(dataset), 
            available_memory_gb=8.0  # Default assumption
        )
        print(f"Auto-optimized batch size: {optimal_batch_size}")
        batch_size = optimal_batch_size
    else:
        batch_size = config.batch_size
    
    # Create collate function
    def collate_fn(samples):
        batch = create_mini_batch_optimized(samples, tokenizer, config.max_len)
        return batch
    
    # Create DataLoader
    # Use num_workers=0 to avoid multiprocessing pickle issues on Windows
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        collate_fn=collate_fn,
        num_workers=0,  # Disable multiprocessing for compatibility
        pin_memory=getattr(config, 'dataloader_pin_memory', False) and torch.cuda.is_available(),
        drop_last=getattr(config, 'dataloader_drop_last', is_training)
    )
    
    return dataloader


def train_batch(model, data, optimizer, device, scaler=None, gradient_accumulation_steps=1):
    """Enhanced training batch function with mixed precision support."""
    model.train()
    
    # Move data to device
    input_ids = data['input_ids'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    phoneme_mask = data['phoneme_mask'].to(device)
    char_ids = data['char_ids'].to(device)
    position_ids = data['position_ids'].to(device)
    label_ids = data['label_ids'].to(device)
    pos_ids = data.get('pos_ids')
    if pos_ids is not None:
        pos_ids = pos_ids.to(device)
    
    # Forward pass with mixed precision
    if scaler is not None:
        with torch.cuda.amp.autocast():
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
            loss = loss / gradient_accumulation_steps
    else:
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
        loss = loss / gradient_accumulation_steps
    
    # Backward pass
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    
    return loss.item() * gradient_accumulation_steps


def evaluate(model, valid_loader, device, scaler=None):
    """Enhanced evaluation function."""
    model.eval()
    
    total_loss = 0
    total_acc = 0
    total_acc_by_char = defaultdict(list)
    total_acc_by_char_bopomofo = defaultdict(list)
    total_pos_acc = 0
    total_samples = 0
    
    with torch.no_grad():
        for data in tqdm(valid_loader, desc="Evaluating"):
            # Move data to device
            input_ids = data['input_ids'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            phoneme_mask = data['phoneme_mask'].to(device)
            char_ids = data['char_ids'].to(device)
            position_ids = data['position_ids'].to(device)
            label_ids = data['label_ids'].to(device)
            pos_ids = data.get('pos_ids')
            if pos_ids is not None:
                pos_ids = pos_ids.to(device)
            
            # Forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
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
            else:
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
            
            # Calculate metrics
            batch_size = input_ids.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            
            # Accuracy calculation
            predictions = torch.argmax(probs, dim=-1)
            correct = (predictions == label_ids).float()
            total_acc += correct.sum().item()
            
            # POS accuracy if available
            if pos_logits is not None and pos_ids is not None:
                pos_predictions = torch.argmax(pos_logits, dim=-1)
                pos_correct = (pos_predictions == pos_ids).float()
                total_pos_acc += pos_correct.sum().item()
    
    # Calculate final metrics
    avg_loss = total_loss / total_samples
    acc = total_acc / total_samples
    pos_acc = total_pos_acc / total_samples if model.use_pos else None
    
    # For compatibility with original evaluation format
    avg_acc_by_char = acc  # Simplified for now
    avg_acc_by_char_bopomofo = acc  # Simplified for now
    
    return {
        'avg_loss': avg_loss,
        'acc': acc,
        'avg_acc_by_char': avg_acc_by_char,
        'avg_acc_by_char_bopomofo': avg_acc_by_char_bopomofo,
        'pos_acc': pos_acc
    }


def create_optimizer_and_scheduler(model, config, num_training_steps):
    """Create optimizer and learning rate scheduler."""
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=getattr(config, 'weight_decay', 0.01),
        eps=getattr(config, 'adam_epsilon', 1e-8)
    )
    
    # Create scheduler
    scheduler = None
    if hasattr(config, 'scheduler_type') and config.scheduler_type == 'linear':
        num_warmup_steps = getattr(config, 'num_warmup_steps', 0)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    return optimizer, scheduler


def main(config_path):
    """Main training function."""
    print("=" * 60)
    print("G2PW Training with Qwen3")
    print("=" * 60)
    
    # Load configuration
    config = get_model_config('qwen3') if config_path == 'qwen3' else get_model_config(config_path.replace('.py', '').split('/')[-1])
    
    # Setup environment
    device, scaler = setup_training_environment(config)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    save_checkpoint_dir = config.output_dir
    
    # Setup logging
    logger = get_logger(os.path.join(save_checkpoint_dir, 'train.log'))
    logger.info(f'Training started at {datetime.now()}')
    logger.info(f'Configuration: {config_path}')
    logger.info(f'Device: {device}')
    logger.info(f'Mixed precision: {scaler is not None}')
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_source)
    
    # Prepare data
    print("Preparing data...")
    train_texts, train_labels, train_query_ids = prepare_data(
        config.train_sent_path, config.train_lb_path, config.polyphonic_chars_path
    )
    valid_texts, valid_labels, valid_query_ids = prepare_data(
        config.valid_sent_path, config.valid_lb_path, config.polyphonic_chars_path
    )
    
    # Get labels and characters
    if config.use_char_phoneme:
        labels = get_char_phoneme_labels(config.polyphonic_chars_path)
    else:
        labels = get_phoneme_labels(config.polyphonic_chars_path)
    
    chars = []
    with open(config.polyphonic_chars_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                chars.append(parts[0])
    chars = sorted(list(set(chars)))
    
    print(f"Data loaded: {len(train_texts)} train, {len(valid_texts)} valid")
    print(f"Labels: {len(labels)}, Characters: {len(chars)}")
    
    # Create model
    model = create_qwen3_model(config, labels, chars)
    model.to(device)
    
    # Create datasets
    print("Creating datasets...")

    # Prepare character to phonemes mapping
    char2phonemes = {}
    with open(config.polyphonic_chars_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                char, phoneme = parts[0], parts[1]
                if char not in char2phonemes:
                    char2phonemes[char] = []
                if phoneme in labels:
                    char2phonemes[char].append(labels.index(phoneme))

    # Create enhanced datasets
    train_dataset = TextDatasetEnhanced(
        tokenizer=tokenizer,
        labels=labels,
        char2phonemes=char2phonemes,
        chars=chars,
        texts=train_texts,
        query_ids=train_query_ids,
        phonemes=train_labels,
        use_mask=True,
        use_char_phoneme=config.use_char_phoneme if hasattr(config, 'use_char_phoneme') else False,
        use_pos=config.use_pos,
        max_len=config.max_len,
        for_train=True
    )

    valid_dataset = TextDatasetEnhanced(
        tokenizer=tokenizer,
        labels=labels,
        char2phonemes=char2phonemes,
        chars=chars,
        texts=valid_texts,
        query_ids=valid_query_ids,
        phonemes=valid_labels,
        use_mask=True,
        use_char_phoneme=config.use_char_phoneme if hasattr(config, 'use_char_phoneme') else False,
        use_pos=config.use_pos,
        max_len=config.max_len,
        for_train=True
    )

    # Create dataloaders
    train_loader = create_optimized_dataloader(train_dataset, config, tokenizer, is_training=True)
    valid_loader = create_optimized_dataloader(valid_dataset, config, tokenizer, is_training=False)

    print(f"Datasets created: train={len(train_dataset)}, valid={len(valid_dataset)}")
    print(f"Batch size: {train_loader.batch_size}")

    # Calculate training steps
    num_training_steps = len(train_loader) * config.epochs
    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, num_training_steps)

    print("Starting training...")
    logger.info(f"Training parameters:")
    logger.info(f"  - Epochs: {config.epochs}")
    logger.info(f"  - Batch size: {train_loader.batch_size}")
    logger.info(f"  - Learning rate: {config.lr}")
    logger.info(f"  - Training steps: {num_training_steps}")
    logger.info(f"  - Gradient accumulation: {gradient_accumulation_steps}")

    # Training loop
    best_accuracy = 0.0
    checkpoint_path = os.path.join(save_checkpoint_dir, 'best_model.pth')
    train_loss_averager = RunningAverage()

    step = 0
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

        # Training phase
        model.train()
        epoch_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch_idx, batch_data in enumerate(progress_bar):
            try:
                # Training step
                loss = train_batch(
                    model, batch_data, optimizer, device, scaler, gradient_accumulation_steps
                )

                train_loss_averager.add(loss)
                epoch_loss += loss
                num_batches += 1

                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if hasattr(config, 'max_grad_norm') and config.max_grad_norm > 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                            optimizer.step()
                    else:
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                    optimizer.zero_grad()

                    if scheduler is not None:
                        scheduler.step()

                    step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'avg_loss': f'{train_loss_averager.get():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })

                # Validation and logging
                if hasattr(config, 'eval_steps') and step % config.eval_steps == 0:
                    print(f"\nValidation at step {step}...")

                    # Validation
                    metrics = evaluate(model, valid_loader, device, scaler)

                    # Save best model
                    if metrics['acc'] > best_accuracy:
                        best_accuracy = metrics['acc']
                        torch.save({
                            'epoch': epoch,
                            'step': step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'best_accuracy': best_accuracy,
                            'config': config
                        }, checkpoint_path)
                        print(f"New best model saved! Accuracy: {best_accuracy:.4f}")

                    # Logging
                    train_loss = train_loss_averager.get()
                    train_loss_averager.flush()

                    pos_acc = 'none' if metrics['pos_acc'] is None else f"{metrics['pos_acc']:.4f}"
                    logger.info(
                        f'[Step {step}] train_loss={train_loss:.6f} '
                        f'valid_loss={metrics["avg_loss"]:.6f} '
                        f'valid_pos_acc={pos_acc} '
                        f'valid_acc={metrics["acc"]:.6f} '
                        f'best_acc={best_accuracy:.6f}'
                    )

                    print(f"Step {step}: train_loss={train_loss:.4f}, valid_acc={metrics['acc']:.4f}, best_acc={best_accuracy:.4f}")

            except Exception as e:
                print(f"Error in training step: {e}")
                logger.error(f"Training step error: {e}")
                continue

        # End of epoch logging
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

    print("\nTraining completed!")
    logger.info(f"Training completed at {datetime.now()}")
    logger.info(f"Best accuracy achieved: {best_accuracy:.6f}")
    print(f"Best model saved to: {checkpoint_path}")
    print(f"Best accuracy: {best_accuracy:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='qwen3', help='config name or path')
    opt = parser.parse_args()
    
    main(opt.config)
