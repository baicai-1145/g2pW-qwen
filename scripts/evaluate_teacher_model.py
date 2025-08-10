"""
Evaluate Teacher Model Performance
Test the trained teacher model and compare with previous results.
"""

import sys
import os
import torch
import json
from datetime import datetime
from tqdm import tqdm

# Add paths
sys.path.insert(0, '.')
sys.path.insert(0, 'g2pw')
sys.path.insert(0, 'configs')

def load_teacher_model():
    """Load the trained teacher model."""
    print("Loading trained teacher model...")
    
    try:
        from g2pw.module import G2PW
        from transformers import AutoTokenizer
        from configs.config_qwen3_teacher import config
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_source, trust_remote_code=True)
        
        # Prepare labels and characters
        polyphonic_chars = []
        with open('cpp_dataset/POLYPHONIC_CHARS.txt', 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    polyphonic_chars.append((parts[0], parts[1]))
        
        labels = sorted(list(set([phoneme for char, phoneme in polyphonic_chars])))
        chars = sorted(list(set([char for char, phoneme in polyphonic_chars])))
        pos_tags = ['A', 'C', 'D', 'DE', 'I', 'N', 'P', 'T', 'UNK', 'V']
        
        # Create model
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
        
        # Load trained weights
        checkpoint_files = [f for f in os.listdir(config.output_dir) if f.endswith('.pth')]
        if checkpoint_files:
            # Load the latest checkpoint by epoch number
            def extract_epoch(filename):
                try:
                    # Extract epoch number from filename like "teacher_epoch_45.pth"
                    return int(filename.split('_epoch_')[1].split('.pth')[0])
                except:
                    return 0

            latest_checkpoint = max(checkpoint_files, key=extract_epoch)
            checkpoint_path = os.path.join(config.output_dir, latest_checkpoint)
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"✓ Loaded checkpoint: {latest_checkpoint}")
            print(f"  - Epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"  - Best accuracy: {checkpoint.get('best_accuracy', 'unknown')}")
        else:
            print("⚠️ No checkpoint found, using untrained model")
        
        return model, tokenizer, labels, chars, config
        
    except Exception as e:
        print(f"✗ Failed to load teacher model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def load_validation_data():
    """Load validation dataset."""
    print("Loading validation data...")
    
    try:
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
        valid_query_ids = []
        for text in valid_texts:
            anchor_pos = text.find('▁')
            if anchor_pos != -1 and anchor_pos + 1 < len(text):
                valid_query_ids.append(anchor_pos + 1)
            else:
                valid_query_ids.append(0)
        
        print(f"✓ Validation data loaded: {len(valid_texts)} samples")
        
        return {
            'texts': valid_texts,
            'labels': valid_labels,
            'pos_tags': valid_pos_tags,
            'query_ids': valid_query_ids
        }
        
    except Exception as e:
        print(f"✗ Failed to load validation data: {e}")
        return None

def evaluate_model(model, tokenizer, labels, chars, data, config):
    """Evaluate model performance."""
    print("Evaluating teacher model performance...")
    
    try:
        from g2pw.dataset_enhanced import TextDatasetEnhanced
        from g2pw.batch_processor import create_mini_batch_optimized
        from torch.utils.data import DataLoader
        
        # Prepare char2phonemes mapping
        polyphonic_chars = []
        with open('cpp_dataset/POLYPHONIC_CHARS.txt', 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    polyphonic_chars.append((parts[0], parts[1]))
        
        char2phonemes = {}
        for char, phoneme in polyphonic_chars:
            if char not in char2phonemes:
                char2phonemes[char] = []
            if phoneme in labels:
                char2phonemes[char].append(labels.index(phoneme))
        
        # Create validation dataset
        valid_dataset = TextDatasetEnhanced(
            tokenizer=tokenizer,
            labels=labels,
            char2phonemes=char2phonemes,
            chars=chars,
            texts=data['texts'],
            query_ids=data['query_ids'],
            phonemes=data['labels'],
            pos_tags=data['pos_tags'],
            use_mask=True,  # 🔧 Enable phoneme mask!
            use_pos=config.use_pos,
            max_len=config.max_len,
            for_train=True
        )
        
        def collate_fn(samples):
            return create_mini_batch_optimized(samples, tokenizer, max_len=config.max_len)
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False
        )
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        print(f"✓ Evaluation setup complete")
        print(f"  - Device: {device}")
        print(f"  - Validation batches: {len(valid_loader)}")
        
        # Evaluation
        total_loss = 0
        total_correct = 0
        total_samples = 0
        pos_correct = 0
        pos_total = 0
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Evaluating"):
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
                    
                    total_loss += loss.item()
                    
                    # Calculate phoneme accuracy
                    predictions = torch.argmax(probs, dim=-1)
                    total_correct += (predictions == label_ids).sum().item()
                    total_samples += label_ids.size(0)
                    
                    # Calculate POS accuracy if available
                    if pos_logits is not None and pos_ids is not None:
                        pos_predictions = torch.argmax(pos_logits, dim=-1)
                        pos_correct += (pos_predictions == pos_ids).sum().item()
                        pos_total += pos_ids.size(0)
                    
                except Exception as e:
                    print(f"Error in batch: {e}")
                    continue
        
        # Calculate final metrics
        avg_loss = total_loss / len(valid_loader)
        phoneme_accuracy = total_correct / total_samples if total_samples > 0 else 0
        pos_accuracy = pos_correct / pos_total if pos_total > 0 else 0
        
        results = {
            'validation_loss': avg_loss,
            'phoneme_accuracy': phoneme_accuracy,
            'pos_accuracy': pos_accuracy,
            'total_samples': total_samples,
            'evaluation_time': datetime.now().isoformat()
        }
        
        print(f"\n{'='*60}")
        print("TEACHER MODEL EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"✓ Validation Loss: {avg_loss:.4f}")
        print(f"✓ Phoneme Accuracy: {phoneme_accuracy:.4f} ({phoneme_accuracy*100:.2f}%)")
        if pos_total > 0:
            print(f"✓ POS Accuracy: {pos_accuracy:.4f} ({pos_accuracy*100:.2f}%)")
        print(f"✓ Total Samples: {total_samples:,}")
        
        # Save results
        results_file = os.path.join(config.output_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_previous_results(results):
    """Compare with previous training results."""
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    # Previous results
    previous_qwen3 = 0.8191  # 81.91% from previous training
    bert_baseline = 0.9871   # 98.71% from BERT
    
    current_accuracy = results['phoneme_accuracy']
    
    print(f"Previous Qwen3 (no POS): {previous_qwen3:.4f} ({previous_qwen3*100:.2f}%)")
    print(f"BERT Baseline:           {bert_baseline:.4f} ({bert_baseline*100:.2f}%)")
    print(f"Teacher Model (with POS): {current_accuracy:.4f} ({current_accuracy*100:.2f}%)")
    
    # Calculate improvements
    improvement_vs_previous = current_accuracy - previous_qwen3
    improvement_vs_bert = current_accuracy - bert_baseline
    
    print(f"\nImprovements:")
    print(f"  vs Previous Qwen3: {improvement_vs_previous:+.4f} ({improvement_vs_previous/previous_qwen3*100:+.2f}%)")
    print(f"  vs BERT Baseline:  {improvement_vs_bert:+.4f} ({improvement_vs_bert/bert_baseline*100:+.2f}%)")
    
    if improvement_vs_previous > 0.03:  # 3% improvement
        print(f"\n🎉 SIGNIFICANT IMPROVEMENT ACHIEVED!")
        print(f"✓ POS joint training shows clear benefits")
    elif improvement_vs_previous > 0.01:  # 1% improvement
        print(f"\n✅ MODERATE IMPROVEMENT ACHIEVED!")
        print(f"✓ POS joint training provides measurable benefits")
    else:
        print(f"\n⚠️ Limited improvement observed")
        print(f"Consider further optimization")

def main():
    """Main evaluation function."""
    print("=" * 60)
    print("TEACHER MODEL EVALUATION")
    print("=" * 60)
    print(f"Evaluation started at: {datetime.now()}")
    
    # Load teacher model
    model, tokenizer, labels, chars, config = load_teacher_model()
    if model is None:
        return
    
    # Load validation data
    data = load_validation_data()
    if data is None:
        return
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, labels, chars, data, config)
    if results is None:
        return
    
    # Compare with previous results
    compare_with_previous_results(results)
    
    print(f"\n🎉 EVALUATION COMPLETED!")
    print(f"Evaluation completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
