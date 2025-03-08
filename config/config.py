#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#
#
"""
Setup script.
@File:  config.py
Authors: shajiu
Date:    2025/3/9 17:27
@Software:  PyCharm
功能为：基础设置和导入: 包含所有配置变量，如路径、模型参数、训练参数等。
"""

import argparse

# Import necessary libraries
# Basic Python libraries for various operations
import random
import copy
import re
import os
import numpy as np
import wandb

# PyTorch and related libraries for deep learning
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# Hugging Face libraries for transformer models
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def set_random_seed(seed: int = 42):
    """
    通过为 Python 的随机模块、NumPy 和 PyTorch 设置种子，确保可复现性；
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None

    Explanation:
        1. Sets seed for Python's built-in random module for basic random operations.
        2. Sets seed for NumPy, ensuring consistent random number generation in array operations.
        3. Sets seed for PyTorch CPU operations.
        4. If CUDA is available, sets seed for all GPU devices.
        5. Configures cuDNN to ensure deterministic behavior:
           - Sets deterministic flag to True, ensuring reproducible results.
           - Disables benchmarking to prevent algorithm selection based on hardware.

    Note:
        Setting deterministic behavior may impact performance but ensures consistent results
        across multiple runs, which is crucial for debugging and research.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model_name_or_path", type=str, help="",default="Qwen/Qwen2.5-1.5B-Instruct")
    # DataSet
    parser.add_argument("--train_path", default="openai/gsm8k", type=str, help="")
    parser.add_argument("--max_len", type=int, default=1024, help="")
    parser.add_argument("--max_src_len", type=int, default=256, help="")
    parser.add_argument("--is_skip", action='store_true', help="")
    # Train
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="")
    parser.add_argument("--num_gpus", type=int, default=4, help="")
    parser.add_argument("--num_steps", type=int, default=10, help="")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="")
    parser.add_argument("--output_dir", type=str, default="grpo_finetuned_model", help="")
    parser.add_argument("--mode", type=str, default="glm2", help="")
    parser.add_argument("--train_type", type=str, default="lora", help="")
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument("--show_loss_step", default=10, type=int, help="")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="")
    parser.add_argument("--save_model_step", default=None, type=int, help="")
    # deepspeed features
    parser.add_argument("--ds_file", type=str, default="ds_zero2.json", help="")
    # LoRA
    parser.add_argument("--lora_dim", type=int, default=8, help="")
    parser.add_argument("--lora_alpha", type=int, default=30, help="")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="")
    parser.add_argument("--lora_module_name", type=str, default="query_key_value", help="")
    # Freeze
    parser.add_argument("--freeze_module_name", type=str, default="layers.27.", help="")
    parser.add_argument("--wandb_api_key", type=str, default="80220a70381b6e059a35692371e2a5679f751465", help="访问：https://wandb.ai/authorize 即可看到key")
    parser.add_argument("--wandb_project", type=str, default="grpo_finetuned_model", help="")
    # P-tuning
    parser.add_argument('--pre_seq_len', type=int, default=16, help='')
    parser.add_argument('--prefix_projection', type=bool, default=True, help='')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("基本参数设置:",args.model_name_or_path)
