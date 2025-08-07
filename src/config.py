"""
Configuration module for the manifolds analysis project.

This module contains configuration classes and model setup utilities.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Config:
    """Configuration class for model and analysis settings."""
    
    model_name: str = "meta-llama/Llama-3.2-1B"
    batch_size: int = 16
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    cache_dir: Optional[str] = None
    
    # Analysis settings
    random_state: int = 42
    n_components: int = 2  # For PCA
    
    # Temporal data settings
    contexts: list = field(default_factory=lambda: ['add_sing', 'add_plur'])
    
    # Context templates for add_sing and add_plur only
    CONTEXT_TEMPLATES: Dict[str, str] = field(default_factory=lambda: {
        'add_sing': "{} plus {} day equals {}",   # e.g. "Monday plus 1 day equals Tuesday"
        'add_plur': "{} plus {} days equals {}"   # e.g. "Monday plus 3 days equals Thursday"
    })
    
    # Days of the week
    days_of_week: list = field(default_factory=lambda: [
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"Device must be 'cuda' or 'cpu', got {self.device}")
        
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")


def load_model_and_tokenizer(config: Config):
    """
    Load the model and tokenizer based on configuration.
    
    Args:
        config: Configuration object with model settings
        
    Returns:
        tuple: (model, tokenizer) loaded and ready to use
    """
    print(f"Loading model: {config.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir
    )
    
    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir,
        torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
        device_map="auto" if config.device == "cuda" else None
    )
    
    model.eval()  # Set to evaluation mode
    
    return model, tokenizer


def get_default_config():
    """Get default configuration object."""
    return Config()