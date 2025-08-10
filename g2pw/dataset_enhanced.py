"""
Enhanced TextDataset with Qwen3 support.
This module provides an improved TextDataset that works with both BERT and Qwen3 tokenizers.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

try:
    from .tokenizer_adapter import tokenize_and_map_unified as tokenize_and_map, is_qwen3_tokenizer
except ImportError:
    try:
        from .utils import tokenize_and_map
        def is_qwen3_tokenizer(tokenizer):
            return 'qwen' in tokenizer.__class__.__name__.lower()
    except ImportError:
        from tokenizer_adapter import tokenize_and_map_unified as tokenize_and_map, is_qwen3_tokenizer

ANCHOR_CHAR = '▁'


class TextDatasetEnhanced(Dataset):
    """
    Enhanced TextDataset that supports both BERT and Qwen3 tokenizers.
    
    Key improvements:
    - Automatic tokenizer detection and adaptation
    - Proper handling of Qwen3's BPE tokenization
    - Flexible special token handling
    - Better error handling and validation
    """
    
    POS_TAGS = ['UNK', 'A', 'C', 'D', 'I', 'N', 'P', 'T', 'V', 'DE', 'SHI']

    def __init__(self, tokenizer, labels, char2phonemes, chars, texts, query_ids, phonemes=None, pos_tags=None,
                 use_mask=False, use_char_phoneme=False, use_pos=False, window_size=None, max_len=512, for_train=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.window_size = window_size
        self.for_train = for_train

        self.labels = labels
        self.char2phonemes = char2phonemes
        self.chars = chars
        self.texts = texts
        self.query_ids = query_ids
        self.phonemes = phonemes
        self.pos_tags = pos_tags

        self.use_mask = use_mask
        self.use_char_phoneme = use_char_phoneme
        self.use_pos = use_pos
        
        # Detect tokenizer type
        self.is_qwen3 = is_qwen3_tokenizer(tokenizer)
        
        # Set up special tokens based on tokenizer type
        self._setup_special_tokens()

        if window_size is not None:
            self.truncated_texts, self.truncated_query_ids = self._truncate_texts(self.window_size, texts, query_ids)

    def _setup_special_tokens(self):
        """Setup special tokens based on tokenizer type."""
        if self.is_qwen3:
            # Qwen3 uses different special tokens
            self.cls_token = '<|endoftext|>'  # Qwen3 doesn't have CLS, use endoftext
            self.sep_token = '<|endoftext|>'  # Qwen3 doesn't have SEP, use endoftext
            self.pad_token = '<|endoftext|>'
            self.unk_token = '<|endoftext|>'
            
            # For Qwen3, we'll use a simpler approach without CLS/SEP
            self.use_cls_sep = False
        else:
            # BERT tokens
            self.cls_token = '[CLS]'
            self.sep_token = '[SEP]'
            self.pad_token = '[PAD]'
            self.unk_token = '[UNK]'
            self.use_cls_sep = True

    def _truncate_texts(self, window_size, texts, query_ids):
        """Truncate texts to window size around query position."""
        truncated_texts = []
        truncated_query_ids = []
        for text, query_id in zip(texts, query_ids):
            start = max(0, query_id - window_size // 2)
            end = min(len(text), query_id + window_size // 2)
            truncated_text = text[start:end]
            truncated_texts.append(truncated_text)

            truncated_query_id = query_id - start
            truncated_query_ids.append(truncated_query_id)
        return truncated_texts, truncated_query_ids

    def _truncate(self, max_len, text, query_id, tokens, text2token, token2text):
        """Truncate tokens to max_len while keeping query position."""
        # Adjust for special tokens
        special_tokens_count = 2 if self.use_cls_sep else 0
        truncate_len = max_len - special_tokens_count
        
        if len(tokens) <= truncate_len:
            return (text, query_id, tokens, text2token, token2text)

        # Ensure query_id is within bounds
        if query_id >= len(text2token):
            query_id = len(text2token) - 1
            
        token_position = text2token[query_id]

        token_start = token_position - truncate_len // 2
        token_end = token_start + truncate_len
        font_exceed_dist = -token_start
        back_exceed_dist = token_end - len(tokens)
        if font_exceed_dist > 0:
            token_start += font_exceed_dist
            token_end += font_exceed_dist
        elif back_exceed_dist > 0:
            token_start -= back_exceed_dist
            token_end -= back_exceed_dist

        # Ensure indices are within bounds
        token_start = max(0, token_start)
        token_end = min(len(tokens), token_end)
        
        if token_start >= len(token2text) or token_end > len(token2text):
            # Fallback: use simple truncation
            return (text, query_id, tokens[:truncate_len], text2token[:len(text)], token2text[:truncate_len])

        start = token2text[token_start][0]
        end = token2text[token_end - 1][1] if token_end > 0 else len(text)

        return (
            text[start:end],
            query_id - start,
            tokens[token_start:token_end],
            [i - token_start if i is not None and i >= token_start else None for i in text2token[start:end]],
            [(s - start, e - start) for s, e in token2text[token_start:token_end]]
        )

    def __getitem__(self, idx):
        """Get a single sample."""
        text = (self.truncated_texts if self.window_size else self.texts)[idx].lower()
        query_id = (self.truncated_query_ids if self.window_size else self.query_ids)[idx]

        try:
            tokens, text2token, token2text = tokenize_and_map(self.tokenizer, text)
        except Exception as e:
            print(f'Warning: text "{text}" tokenization failed: {e}')
            # Return next sample to avoid crash
            return self[(idx + 1) % len(self)]

        # Validate tokenization results
        if len(text2token) != len(text):
            print(f'Warning: tokenization length mismatch for text "{text[:50]}..."')
            # Try to fix by padding or truncating
            if len(text2token) < len(text):
                text2token.extend([len(tokens) - 1] * (len(text) - len(text2token)))
            else:
                text2token = text2token[:len(text)]

        text, query_id, tokens, text2token, token2text = self._truncate(self.max_len, text, query_id, tokens, text2token, token2text)

        # Prepare tokens with special tokens
        if self.use_cls_sep:
            processed_tokens = [self.cls_token] + tokens + [self.sep_token]
            position_offset = 1  # Account for CLS token
        else:
            processed_tokens = tokens
            position_offset = 0

        # Convert tokens to IDs
        try:
            input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(processed_tokens))
        except Exception as e:
            print(f'Warning: token conversion failed: {e}')
            # Fallback: encode the text directly
            encoded = self.tokenizer.encode(text, add_special_tokens=self.use_cls_sep, return_tensors="pt")
            input_ids = encoded.squeeze(0)
            if len(input_ids) > self.max_len:
                input_ids = input_ids[:self.max_len]

        # Create attention mask and token type IDs
        token_type_ids = torch.tensor([0] * len(input_ids))
        attention_mask = torch.tensor([1] * len(input_ids))

        # Get query character and create phoneme mask
        if query_id < len(text):
            query_char = text[query_id]
        else:
            query_char = text[-1] if text else '了'  # Fallback
            
        phoneme_mask = [1 if i in self.char2phonemes.get(query_char, []) else 0 for i in range(len(self.labels))] \
            if self.use_mask else [1] * len(self.labels)
            
        # Get character ID
        char_id = self.chars.index(query_char) if query_char in self.chars else 0
        
        # Calculate position ID
        if query_id < len(text2token) and text2token[query_id] is not None:
            position_id = text2token[query_id] + position_offset
        else:
            position_id = position_offset  # Fallback to first position

        outputs = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'phoneme_mask': phoneme_mask,
            'char_id': char_id,
            'position_id': position_id,
        }

        if self.use_pos and self.pos_tags is not None:
            pos_id = self.POS_TAGS.index(self.pos_tags[idx]) if self.pos_tags[idx] in self.POS_TAGS else 0
            outputs['pos_id'] = pos_id

        if self.for_train and self.phonemes is not None:
            phoneme = self.phonemes[idx]
            label_key = f'{query_char} {phoneme}' if self.use_char_phoneme else phoneme
            label_id = self.labels.index(label_key) if label_key in self.labels else 0
            outputs['label_id'] = label_id

        # Add debug info
        info = {
            'text': text,
            'tokens': tokens,
            'text2token': text2token,
            'token2text': token2text,
            'query_char': query_char,
            'is_qwen3': self.is_qwen3
        }
        outputs['info'] = info
        return outputs

    def __len__(self):
        return len(self.texts)

    def create_mini_batch(self, samples):
        """Create a mini-batch from samples."""
        def _agg(name):
            return [sample[name] for sample in samples]

        # Pad sequences to same length
        input_ids = pad_sequence(_agg('input_ids'), batch_first=True, padding_value=self.tokenizer.pad_token_id or 0)
        token_type_ids = pad_sequence(_agg('token_type_ids'), batch_first=True)
        attention_mask = pad_sequence(_agg('attention_mask'), batch_first=True)
        phoneme_mask = torch.tensor(_agg('phoneme_mask'), dtype=torch.float)
        char_ids = torch.tensor(_agg('char_id'), dtype=torch.long)
        position_ids = torch.tensor(_agg('position_id'), dtype=torch.long)

        batch_output = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'phoneme_mask': phoneme_mask,
            'char_ids': char_ids,
            'position_ids': position_ids
        }

        if self.use_pos and any('pos_id' in sample for sample in samples):
            pos_ids = torch.tensor(_agg('pos_id'), dtype=torch.long)
            batch_output['pos_ids'] = pos_ids

        if self.for_train and any('label_id' in sample for sample in samples):
            label_ids = torch.tensor(_agg('label_id'), dtype=torch.long)
            batch_output['label_ids'] = label_ids
        else:
            infos = _agg('info')
            batch_output['infos'] = infos

        return batch_output


# Create a function to use the enhanced dataset
def create_mini_batch(samples):
    """Standalone function for creating mini-batches."""
    if not samples:
        return {}
    
    # Use the first sample to determine the dataset type
    first_sample = samples[0]
    if 'info' in first_sample and 'is_qwen3' in first_sample['info']:
        # This is from our enhanced dataset, use its method
        # For now, we'll implement a simple version here
        pass
    
    # Fallback to simple implementation
    def _agg(name):
        return [sample[name] for sample in samples if name in sample]

    batch_output = {}
    
    if _agg('input_ids'):
        batch_output['input_ids'] = pad_sequence(_agg('input_ids'), batch_first=True)
    if _agg('token_type_ids'):
        batch_output['token_type_ids'] = pad_sequence(_agg('token_type_ids'), batch_first=True)
    if _agg('attention_mask'):
        batch_output['attention_mask'] = pad_sequence(_agg('attention_mask'), batch_first=True)
    if _agg('phoneme_mask'):
        batch_output['phoneme_mask'] = torch.tensor(_agg('phoneme_mask'), dtype=torch.float)
    if _agg('char_id'):
        batch_output['char_ids'] = torch.tensor(_agg('char_id'), dtype=torch.long)
    if _agg('position_id'):
        batch_output['position_ids'] = torch.tensor(_agg('position_id'), dtype=torch.long)
    if _agg('label_id'):
        batch_output['label_ids'] = torch.tensor(_agg('label_id'), dtype=torch.long)
    if _agg('pos_id'):
        batch_output['pos_ids'] = torch.tensor(_agg('pos_id'), dtype=torch.long)
    if _agg('info'):
        batch_output['infos'] = _agg('info')

    return batch_output
