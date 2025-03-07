#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#
#
"""
Setup script.
@File:   train.py
Authors: shajiu
Date:    2025/3/9 17:27
@Software:  PyCharm
功能为： 包括模型训练过程的脚本，如模型初始化、训练循环、保存检查点等。
训练设置和执行：
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import wandb
import os

from dataset.dataset_loader import prepare_dataset
from scripts.evaluate import evaluate_model
from models.grpo import train_with_grpo
from models.reward import combined_reward
from config.config import parse_args,set_random_seed

def optimize_model_memory(model):
    """
    Optimizes the model to use less memory during training.

    Args:
        model: The language model to optimize.

    Returns:
        The optimized model.

    Explanation:
        1. Sets the model to training mode.
        2. Disables KV caching to save memory.
        3. Enables gradient checkpointing to trade computation for memory.
        4. Ensures that input embeddings require gradients:
           - Either uses the built-in method if available.
           - Or adds a forward hook to the input embeddings layer.
        5. Returns the optimized model ready for memory-efficient training.
    """
    model.train()
    model.config.use_cache = False

    # First ensure inputs will require gradients
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Then enable gradient checkpointing
    model.gradient_checkpointing_enable()

    return model

def main(args):
    # Call the function to set random seed for reproducibility
    set_random_seed(args.seed)
    # Set environment variables for Weights & Biases (wandb) logging
    # 访问：https://wandb.ai/authorize 即可看到key
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    os.environ["WANDB_PROJECT"] = args.wandb_project
    # Main execution
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using primary device: {device}")
    # 设置模型存储文件路径
    output_dir = "math_solver_model"

    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("Model downloaded")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    num_gpus =args.num_gpus   # torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")
    device_ids = list(range(num_gpus)) if num_gpus > 1 else None

    # 从数据加载部分调用函数并获取训练部分的数据集
    all_data = prepare_dataset(args,"train")
    # 随机打乱
    random.shuffle(all_data)
    # 从训练数据中分割部分为验证集
    size_of_eval_data = 30  # change to a smaller value to save time or to a larger number for a more reliable estimate
    eval_data = all_data[:size_of_eval_data]
    train_data = all_data[size_of_eval_data:]

    # (1) 未训练前进行预测获取指标
    print("\nInitial model evaluation before finetuning:")
    pre_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
    print(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")

    # 优化模型使用使用更少的内存
    model = optimize_model_memory(model)

    print("\nStarting RL fine-tuning using GRPO...")
    # 配置训练配置参数
    # This config was tested on a 8xA100 node, where each A100 is has 80GB of VRAM
    training_config = {
        'num_iterations': args.num_train_epochs,  # 从当前策略模型创建新参考模型的外部迭代次数。一次迭代是指在整个数据集上执行一次通过。
        'num_steps': args.num_steps,  # 训练循环将执行最多 500 个步骤，每个步骤处理一批样本。
        'batch_size': args.per_device_train_batch_size,  # 在 8 台 GPU 的情况下，每个步骤每批处理 7 个样本，每台 GPU 上放置 1 个样本。使用一个 GPU (0) 被 DataParallel 用作主节点来聚合梯度并收集输出。
        'num_generations': 12,
        # 对于训练数据中的每个提示词，训练器将生成 12 个不同的完成结果。这些生成结果将被用于计算指导强化学习更新的相对优势（或奖励信号）。如果你的 GPU 的 VRAM 较少，请减少此数字。
        'max_completion_length': args.max_len,
        # 在生成完成结果（序列的 response 部分）时，生成上限为 400 个 token。这限制了模型在 RL 阶段生成的输出的长度。如果你的 GPU 的 VRAM 较少，请减少此数字。
        'beta': 0.04,  # GRPO 损失函数中 KL 散度惩罚的系数。这控制的是模型与参考模型的偏差程度。
        'learning_rate': args.learning_rate,  # RL 微调的学习率。为了实现稳定的策略更新，这里使用了相对较低的学习率。
        'mu': 1,  # 对每批 rollout 数据执行的策略更新次数。在这里，我们每批只执行一次更新。
        'epsilon': args.warmup_ratio  # GRPO 的 PPO 组件的 clipping 参数。这可以防止策略在单次更新中发生太大的变化。
    }
    wandb.init(project=os.environ["WANDB_PROJECT"], reinit=True)
    print("Weights & Biases initialized.")

    # 训练过程
    model = train_with_grpo(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        reward_function=combined_reward,
        device_ids=device_ids,
        **training_config
    )

    wandb.finish()
    print("Training completed and wandb run finished.")

    print("\nFinal model evaluation after GRPO RL fine-tuning:")
    # (2) 训练后再次预测获取指标
    post_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
    print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")

    print("\nSaving GRPO fine-tuned model...")
    # 模型存储在此文件夹下
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)


