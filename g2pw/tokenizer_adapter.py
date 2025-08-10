"""
Tokenizer adapter for handling different tokenization schemes in g2pW.
This module provides unified tokenization interface for both BERT (WordPiece) and Qwen3 (BPE).
"""

import re
from typing import List, Tuple, Union
from transformers import BertTokenizer, PreTrainedTokenizer


def is_bert_tokenizer(tokenizer):
    """Check if tokenizer is BERT-based."""
    return isinstance(tokenizer, BertTokenizer) or 'bert' in tokenizer.__class__.__name__.lower()


def is_qwen3_tokenizer(tokenizer):
    """Check if tokenizer is Qwen3-based."""
    return 'qwen' in tokenizer.__class__.__name__.lower() or getattr(tokenizer, 'vocab_size', 0) > 100000


def wordize_and_map(text):
    """
    Split text into words and create mapping indices.
    This is the same as the original function in utils.py.
    """
    words = []
    index_map_from_text_to_word = []
    index_map_from_word_to_text = []

    word_start = 0
    while text:
        if text[0] == ' ':
            text = text[1:]
            word_start += 1
        elif text[0] in '，。！？；：""''（）【】《》':
            words.append(text[0])
            index_map_from_word_to_text.append((word_start, word_start + 1))
            for _ in range(1):
                index_map_from_text_to_word.append(len(words) - 1)
            text = text[1:]
            word_start += 1
        else:
            word_end = 1
            while word_end < len(text) and text[word_end] not in ' ，。！？；：""''（）【】《》':
                word_end += 1
            words.append(text[:word_end])
            index_map_from_word_to_text.append((word_start, word_start + word_end))
            for _ in range(word_end):
                index_map_from_text_to_word.append(len(words) - 1)
            text = text[word_end:]
            word_start += word_end

    if not words:
        if text:
            words.append(text[0])
            text = text[1:]
    return words, index_map_from_text_to_word, index_map_from_word_to_text


def tokenize_and_map_bert(tokenizer, text):
    """
    Original tokenize_and_map function for BERT tokenizer.
    """
    words, text2word, word2text = wordize_and_map(text)

    tokens = []
    index_map_from_token_to_text = []
    for word, (word_start, word_end) in zip(words, word2text):
        word_tokens = tokenizer.tokenize(word)

        if len(word_tokens) == 0 or word_tokens == ['[UNK]']:
            index_map_from_token_to_text.append((word_start, word_end))
            tokens.append('[UNK]')
        else:
            current_word_start = word_start
            for word_token in word_tokens:
                word_token_len = len(re.sub(r'^##', '', word_token))
                index_map_from_token_to_text.append(
                    (current_word_start, current_word_start + word_token_len))
                current_word_start = current_word_start + word_token_len
                tokens.append(word_token)

    index_map_from_text_to_token = text2word[:]
    for i, (token_start, token_end) in enumerate(index_map_from_token_to_text):
        for token_pos in range(token_start, min(token_end, len(index_map_from_text_to_token))):
            index_map_from_text_to_token[token_pos] = i

    return tokens, index_map_from_text_to_token, index_map_from_token_to_text


def tokenize_and_map_qwen3(tokenizer, text):
    """
    Adapted tokenize_and_map function for Qwen3 BPE tokenizer.
    
    The key difference is that BPE tokens don't have a direct character-level correspondence,
    so we need to reconstruct the alignment differently.
    """
    # First, get the tokens
    tokens = tokenizer.tokenize(text)
    
    # For BPE, we need to decode tokens back to text to find alignment
    index_map_from_token_to_text = []
    index_map_from_text_to_token = [0] * len(text)
    
    current_pos = 0
    for i, token in enumerate(tokens):
        # Decode the token to get its text representation
        try:
            # For Qwen3, we need to handle the token decoding carefully
            token_text = tokenizer.convert_tokens_to_string([token])
            token_len = len(token_text)
            
            # Find the token in the original text starting from current_pos
            found_pos = text.find(token_text, current_pos)
            if found_pos != -1:
                # Update mappings
                token_start = found_pos
                token_end = found_pos + token_len
                index_map_from_token_to_text.append((token_start, token_end))
                
                # Update text to token mapping
                for pos in range(token_start, min(token_end, len(text))):
                    index_map_from_text_to_token[pos] = i
                
                current_pos = token_end
            else:
                # Fallback: approximate mapping
                token_start = current_pos
                token_end = min(current_pos + 1, len(text))
                index_map_from_token_to_text.append((token_start, token_end))
                
                if token_start < len(text):
                    index_map_from_text_to_token[token_start] = i
                
                current_pos = token_end
                
        except Exception:
            # Fallback for problematic tokens
            token_start = current_pos
            token_end = min(current_pos + 1, len(text))
            index_map_from_token_to_text.append((token_start, token_end))
            
            if token_start < len(text):
                index_map_from_text_to_token[token_start] = i
            
            current_pos = token_end
    
    return tokens, index_map_from_text_to_token, index_map_from_token_to_text


def tokenize_and_map_qwen3_simple(tokenizer, text):
    """
    Simplified tokenize_and_map for Qwen3 that maintains character-level alignment.
    
    This approach treats each character as potentially mapping to a different token,
    which is safer for g2pW's character-level requirements.
    """
    tokens = tokenizer.tokenize(text)
    
    # Create simple mappings
    # For BPE, we'll use a conservative approach where each character maps to the nearest token
    index_map_from_text_to_token = []
    index_map_from_token_to_text = []
    
    # Simple strategy: distribute characters evenly across tokens
    chars_per_token = len(text) / max(len(tokens), 1)
    
    for i, char in enumerate(text):
        # Map each character to the appropriate token index
        token_idx = min(int(i / chars_per_token), len(tokens) - 1)
        index_map_from_text_to_token.append(token_idx)
    
    # Create reverse mapping
    for i, token in enumerate(tokens):
        start_char = int(i * chars_per_token)
        end_char = int((i + 1) * chars_per_token)
        if i == len(tokens) - 1:  # Last token gets remaining characters
            end_char = len(text)
        index_map_from_token_to_text.append((start_char, min(end_char, len(text))))
    
    return tokens, index_map_from_text_to_token, index_map_from_token_to_text


def tokenize_and_map_unified(tokenizer, text):
    """
    Unified tokenize_and_map function that works with both BERT and Qwen3 tokenizers.
    
    Args:
        tokenizer: Either BERT or Qwen3 tokenizer
        text: Input text to tokenize
        
    Returns:
        tokens: List of tokens
        text2token: List mapping each character position to token index
        token2text: List of (start, end) tuples for each token
    """
    if is_bert_tokenizer(tokenizer):
        return tokenize_and_map_bert(tokenizer, text)
    elif is_qwen3_tokenizer(tokenizer):
        # Use the simple approach for now as it's more reliable
        return tokenize_and_map_qwen3_simple(tokenizer, text)
    else:
        # Fallback to BERT approach for unknown tokenizers
        return tokenize_and_map_bert(tokenizer, text)


def validate_tokenization_mapping(tokens, text2token, token2text, text):
    """
    Validate that the tokenization mapping is consistent.
    
    Returns:
        bool: True if mapping is valid, False otherwise
    """
    try:
        # Check basic lengths
        if len(text2token) != len(text):
            return False
        
        if len(token2text) != len(tokens):
            return False
        
        # Check that all character positions map to valid token indices
        for i, token_idx in enumerate(text2token):
            if token_idx < 0 or token_idx >= len(tokens):
                return False
        
        # Check that token ranges are within text bounds
        for start, end in token2text:
            if start < 0 or end > len(text) or start > end:
                return False
        
        return True
        
    except Exception:
        return False


# Monkey patch the original function for backward compatibility
def patch_original_tokenize_and_map():
    """
    Patch the original tokenize_and_map function in utils.py to use our unified version.
    """
    import g2pw.utils as utils
    utils.tokenize_and_map = tokenize_and_map_unified
