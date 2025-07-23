#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025/07/23

🚀 Welcome to the Awesome Python Script 🚀

User: Messou Franck Junior Aboya
Email: messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University - IIST - (Tokyo, Japan)
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

# argument_parser.py

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Run imputation models via PyPOTS pipeline")

    # General arguments
    parser.add_argument("--model", type=str, default="llm4imp", help="Model name: llm4imp | saits | timellm")
    parser.add_argument("--missing_rate", type=float, default=0.1, help="Artificial missing rate in dataset")
    parser.add_argument("--saving_path", type=str, default="output/imputation", help="Directory to save results")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cpu', 'cuda', or None (auto)")

    # Shared model hyperparameters
    parser.add_argument("--n_steps", type=int, default=48, help="Number of time steps")
    parser.add_argument("--n_features", type=int, default=35, help="Number of input features")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--d_model", type=int, default=64, help="Model hidden size")
    parser.add_argument("--d_ffn", type=int, default=128, help="Feed-forward network size")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # LLM4IMP & TimeLLM-specific
    parser.add_argument("--patch_size", type=int, default=12, help="Patch length for patch embedding")
    parser.add_argument("--patch_stride", type=int, default=6, help="Patch stride")
    parser.add_argument("--d_llm", type=int, default=768, help="Dimension of LLM model (GPT2: 768, LLaMA: 4096)")
    parser.add_argument("--prompt_template", type=str, default="Impute missing values at time steps where mask=0", help="Instruction prompt template")
    parser.add_argument("--llm_model_type", type=str, default="GPT2", help="LLM model type: GPT2 | LLaMA | BERT")

    # SAITS-specific
    parser.add_argument("--d_k", type=int, default=16, help="Key/query dimension for SAITS")
    parser.add_argument("--d_v", type=int, default=16, help="Value dimension for SAITS")
    parser.add_argument("--attn_dropout", type=float, default=0.1, help="Attention dropout for SAITS")
    parser.add_argument("--diagonal_attention_mask", action="store_true", help="Use diagonal attention mask in SAITS")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")

    return parser.parse_args()