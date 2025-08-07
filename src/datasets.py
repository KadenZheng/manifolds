"""
Dataset module for the manifolds analysis project.

This module contains dataset classes for temporal and control word analysis.
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

from .config import Config


class TemporalDataset(Dataset):
    """PyTorch dataset for temporal word sequences."""
    
    def __init__(self, encodings, dataframe):
        """
        Initialize the temporal dataset.
        
        Args:
            encodings: Tokenized encodings from the tokenizer
            dataframe: DataFrame with metadata about each sample
        """
        self.encodings = encodings
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['day'] = self.dataframe.iloc[idx]['day'] 
        item['offset'] = self.dataframe.iloc[idx]['offset']
        return item


class TemporalDatasetCreator:
    """Creates temporal datasets with various contexts."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def create_dataset(self, contexts: Optional[List[str]] = None):
        """
        Create dataset with add_sing and add_plur examples only.
        For add_sing: offset is always 1 (singular "day")
        For add_plur: offsets range from 2-70 (plural "days"), excluding multiples of 7
        
        Args:
            contexts: List of context names to use (default from config)
            
        Returns:
            list: List of dictionaries with text, day, offset, and context
        """
        if contexts is None:
            contexts = self.config.contexts
            
        data = []
        days = self.config.days_of_week
        
        for ctx in contexts:
            if ctx not in self.config.CONTEXT_TEMPLATES:
                raise ValueError(f"Unknown context: {ctx}")
                
            tpl = self.config.CONTEXT_TEMPLATES[ctx]
            
            if ctx == 'add_sing':
                offsets = [1]  # Only offset 1 for singular
            elif ctx == 'add_plur':
                # Offsets from 2-70, excluding multiples of 7
                offsets = [x for x in range(2, 71) if x % 7 != 0]
            else:
                raise ValueError(f"Unsupported context: {ctx}")
            
            for day_idx, day in enumerate(days):
                for offset in offsets:
                    result_idx = (day_idx + offset) % 7
                    result_day = days[result_idx]
                    
                    # Format the text according to the template
                    if ctx == 'add_sing':
                        text = tpl.format(day, offset, result_day)
                    else:  # add_plur
                        text = tpl.format(day, offset, result_day)
                    
                    data.append({
                        'text': text,
                        'day': day_idx,
                        'offset': offset,
                        'result_day': result_idx,
                        'context': ctx
                    })
        
        return data

class ControlDataset(Dataset):
    """PyTorch dataset for control words."""
    
    def __init__(self, encodings, labels):
        """
        Initialize the control dataset.
        
        Args:
            encodings: Tokenized encodings from the tokenizer
            labels: List of labels for each sample
        """
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = self.labels[idx]
        return item


def create_incremental_datasets(config: Config, max_offsets: List[int] = None):
    """
    Create incremental datasets with varying maximum offsets.
    
    Args:
        config: Configuration object
        max_offsets: List of maximum offset values to use
        
    Returns:
        dict: Dictionary mapping max_offset to dataset
    """
    if max_offsets is None:
        max_offsets = [7, 14, 21, 28, 35, 42, 49, 56, 63, 70]
    
    datasets = {}
    days = config.days_of_week
    
    for max_offset in max_offsets:
        data = []
        for day_idx, day in enumerate(days):
            # Only add_plur context for incremental analysis
            for offset in range(2, max_offset + 1):
                if offset % 7 == 0:  # Skip multiples of 7
                    continue
                    
                result_idx = (day_idx + offset) % 7
                result_day = days[result_idx]
                
                text = config.CONTEXT_TEMPLATES['add_plur'].format(day, offset, result_day)
                
                data.append({
                    'text': text,
                    'day': day_idx,
                    'offset': offset,
                    'result_day': result_idx,
                    'context': 'add_plur',
                    'max_offset_range': max_offset
                })
        
        datasets[max_offset] = data
    
    return datasets


def get_control_words():
    """
    Get list of control words for comparison.
    
    Returns:
        list: List of control words
    """
    # Based on the notebook, control words are a simple list, not categorized
    control_words = [
        'apple', 'banana', 'orange', 'grape', 'strawberry',
        'carrot', 'broccoli', 'potato', 'tomato', 'lettuce',
        'chicken', 'beef', 'pork', 'fish', 'turkey',
        'water', 'milk', 'juice', 'coffee', 'tea',
        'bread', 'rice', 'pasta', 'cheese', 'yogurt',
        'happy', 'sad', 'angry', 'excited', 'tired',
        'big', 'small', 'hot', 'cold', 'fast',
        'run', 'walk', 'jump', 'sit', 'stand',
        'read', 'write', 'speak', 'listen', 'think'
    ]
    return control_words


def prepare_control_data(control_words: List[str], tokenizer: PreTrainedTokenizer):
    """
    Prepare control words for analysis.
    
    Args:
        control_words: List of control words
        tokenizer: Tokenizer to use for encoding
        
    Returns:
        tuple: (encodings, labels) ready for dataset creation
    """
    # Tokenize all control words
    encodings = tokenizer(
        control_words,
        truncation=True,
        padding=True,
        return_tensors='pt',
        max_length=128
    )
    
    # Create simple labels (just the word itself)
    labels = control_words
    
    return encodings, labels