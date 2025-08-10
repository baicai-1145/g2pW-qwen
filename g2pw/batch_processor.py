"""
Optimized batch processing for g2pW with Qwen3 support.
This module provides efficient batch creation and processing for both BERT and Qwen3 models.
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Optional, Union
import numpy as np


class BatchProcessor:
    """
    Optimized batch processor for g2pW.
    
    Features:
    - Memory-efficient padding
    - Automatic tokenizer detection
    - Dynamic batch size adjustment
    - GPU memory optimization
    """
    
    def __init__(self, tokenizer, max_len: int = 512, device: str = 'cpu'):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        
        # Detect tokenizer type
        self.is_qwen3 = 'qwen' in tokenizer.__class__.__name__.lower()
        
        # Set padding token ID
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            self.pad_token_id = tokenizer.pad_token_id
        else:
            # Fallback for tokenizers without pad_token_id
            self.pad_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 0
    
    def create_mini_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Create an optimized mini-batch from samples.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            Dictionary containing batched tensors
        """
        if not samples:
            return {}
        
        batch_size = len(samples)
        
        # Helper function to aggregate values
        def _agg(name: str, default=None):
            values = []
            for sample in samples:
                if name in sample:
                    values.append(sample[name])
                elif default is not None:
                    values.append(default)
            return values
        
        # Create batch dictionary
        batch_output = {}
        
        # Process input sequences with optimized padding
        input_ids_list = _agg('input_ids')
        if input_ids_list:
            # Find optimal sequence length (avoid excessive padding)
            seq_lengths = [len(ids) for ids in input_ids_list]
            max_seq_len = min(max(seq_lengths), self.max_len)
            
            # Truncate sequences that are too long
            input_ids_truncated = []
            for ids in input_ids_list:
                if len(ids) > max_seq_len:
                    input_ids_truncated.append(ids[:max_seq_len])
                else:
                    input_ids_truncated.append(ids)
            
            # Pad sequences
            batch_output['input_ids'] = pad_sequence(
                input_ids_truncated, 
                batch_first=True, 
                padding_value=self.pad_token_id
            )
        
        # Process token type IDs
        token_type_ids_list = _agg('token_type_ids')
        if token_type_ids_list:
            # For Qwen3, token_type_ids might be ignored, but we still process them for compatibility
            token_type_ids_truncated = []
            for ids in token_type_ids_list:
                if len(ids) > max_seq_len:
                    token_type_ids_truncated.append(ids[:max_seq_len])
                else:
                    token_type_ids_truncated.append(ids)
            
            batch_output['token_type_ids'] = pad_sequence(
                token_type_ids_truncated, 
                batch_first=True, 
                padding_value=0
            )
        
        # Process attention masks
        attention_mask_list = _agg('attention_mask')
        if attention_mask_list:
            attention_mask_truncated = []
            for mask in attention_mask_list:
                if len(mask) > max_seq_len:
                    attention_mask_truncated.append(mask[:max_seq_len])
                else:
                    attention_mask_truncated.append(mask)
            
            batch_output['attention_mask'] = pad_sequence(
                attention_mask_truncated, 
                batch_first=True, 
                padding_value=0
            )
        
        # Process phoneme masks (no padding needed, fixed size)
        phoneme_mask_list = _agg('phoneme_mask')
        if phoneme_mask_list:
            batch_output['phoneme_mask'] = torch.tensor(phoneme_mask_list, dtype=torch.float)
        
        # Process character IDs
        char_id_list = _agg('char_id')
        if char_id_list:
            batch_output['char_ids'] = torch.tensor(char_id_list, dtype=torch.long)
        
        # Process position IDs with bounds checking
        position_id_list = _agg('position_id')
        if position_id_list:
            # Ensure position IDs are within sequence bounds
            adjusted_position_ids = []
            for i, pos_id in enumerate(position_id_list):
                if i < len(seq_lengths):
                    # Clamp position ID to valid range
                    max_pos = min(seq_lengths[i] - 1, max_seq_len - 1)
                    adjusted_pos_id = max(0, min(pos_id, max_pos))
                    adjusted_position_ids.append(adjusted_pos_id)
                else:
                    adjusted_position_ids.append(0)
            
            batch_output['position_ids'] = torch.tensor(adjusted_position_ids, dtype=torch.long)
        
        # Process POS IDs if present
        pos_id_list = _agg('pos_id')
        if pos_id_list:
            batch_output['pos_ids'] = torch.tensor(pos_id_list, dtype=torch.long)
        
        # Process labels for training
        label_id_list = _agg('label_id')
        if label_id_list:
            batch_output['label_ids'] = torch.tensor(label_id_list, dtype=torch.long)
        
        # Process info for inference
        info_list = _agg('info')
        if info_list:
            batch_output['infos'] = info_list
        
        return batch_output
    
    def optimize_batch_size(self, dataset_size: int, available_memory_gb: float = 8.0) -> int:
        """
        Calculate optimal batch size based on available memory and model size.
        
        Args:
            dataset_size: Size of the dataset
            available_memory_gb: Available GPU memory in GB
            
        Returns:
            Recommended batch size
        """
        # Estimate memory usage per sample
        if self.is_qwen3:
            # Qwen3 is larger, needs more memory
            memory_per_sample_mb = 50  # Rough estimate
            base_batch_size = 8
        else:
            # BERT is smaller
            memory_per_sample_mb = 20  # Rough estimate
            base_batch_size = 16
        
        # Calculate based on available memory
        available_memory_mb = available_memory_gb * 1024
        max_batch_size = int(available_memory_mb / memory_per_sample_mb)
        
        # Use reasonable bounds
        optimal_batch_size = min(max_batch_size, max(base_batch_size, 4))
        optimal_batch_size = min(optimal_batch_size, dataset_size)
        
        return optimal_batch_size
    
    def validate_batch(self, batch: Dict[str, torch.Tensor]) -> bool:
        """
        Validate batch consistency and detect potential issues.
        
        Args:
            batch: Batch dictionary
            
        Returns:
            True if batch is valid, False otherwise
        """
        try:
            # Check if batch is empty
            if not batch:
                return False
            
            # Get batch size from first tensor
            batch_size = None
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    if batch_size is None:
                        batch_size = value.size(0)
                    elif value.size(0) != batch_size:
                        print(f"Batch size mismatch: {key} has size {value.size(0)}, expected {batch_size}")
                        return False
            
            # Check sequence length consistency
            if 'input_ids' in batch and 'attention_mask' in batch:
                if batch['input_ids'].shape != batch['attention_mask'].shape:
                    print("Input IDs and attention mask shape mismatch")
                    return False
            
            # Check position IDs are within bounds
            if 'position_ids' in batch and 'input_ids' in batch:
                max_seq_len = batch['input_ids'].size(1)
                if torch.any(batch['position_ids'] >= max_seq_len):
                    print("Position IDs exceed sequence length")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Batch validation error: {e}")
            return False
    
    def to_device(self, batch: Dict[str, torch.Tensor], device: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Move batch to specified device.
        
        Args:
            batch: Batch dictionary
            device: Target device (if None, use self.device)
            
        Returns:
            Batch with tensors moved to device
        """
        target_device = device or self.device
        
        if target_device == 'cpu':
            return batch
        
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(target_device)
            else:
                device_batch[key] = value
        
        return device_batch


# Standalone function for backward compatibility
def create_mini_batch_optimized(samples: List[Dict[str, Any]], 
                               tokenizer=None, 
                               max_len: int = 512,
                               device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    Optimized mini-batch creation function.
    
    Args:
        samples: List of sample dictionaries
        tokenizer: Tokenizer instance (for pad_token_id)
        max_len: Maximum sequence length
        device: Target device
        
    Returns:
        Batch dictionary
    """
    if not tokenizer:
        # Fallback to simple implementation
        return _create_mini_batch_simple(samples)
    
    processor = BatchProcessor(tokenizer, max_len, device)
    batch = processor.create_mini_batch(samples)
    
    # Validate batch
    if not processor.validate_batch(batch):
        print("Warning: Batch validation failed, using fallback")
        return _create_mini_batch_simple(samples)
    
    return batch


def _create_mini_batch_simple(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Simple fallback batch creation function.
    """
    if not samples:
        return {}
    
    def _agg(name: str):
        return [sample[name] for sample in samples if name in sample]
    
    batch_output = {}
    
    # Basic tensor aggregation
    for key in ['input_ids', 'token_type_ids', 'attention_mask']:
        values = _agg(key)
        if values:
            batch_output[key] = pad_sequence(values, batch_first=True)
    
    for key in ['phoneme_mask']:
        values = _agg(key)
        if values:
            batch_output[key] = torch.tensor(values, dtype=torch.float)
    
    for key in ['char_id', 'position_id', 'label_id', 'pos_id']:
        values = _agg(key)
        if values:
            tensor_key = key + 's' if not key.endswith('_id') else key.replace('_id', '_ids')
            batch_output[tensor_key] = torch.tensor(values, dtype=torch.long)
    
    # Handle info
    infos = _agg('info')
    if infos:
        batch_output['infos'] = infos
    
    return batch_output
