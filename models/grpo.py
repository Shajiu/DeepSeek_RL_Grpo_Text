#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#
#
"""
Setup script.
@File:  model.py
Authors: shajiu
Date:    2025/3/9 17:27
@Software:  PyCharm
功能为：包含基于llm的模型架构，可能包括加载预训练模型和定义新的层或变化。
从头实现 GRPO 算法的所有构建模块。首先，这里假设运行代码的机器至少有2台GPU。为此，这里要使用PyTorch的 DataParallelAPI来。将策略模型放在多个GPU核心上，每个GPU核心都有该模型的一个副本。然后将批量数据分散在这些GPU核心上完成处理。
"""
import torch
import torch.nn as nn
import copy
import random
import wandb


def selective_log_softmax(logits, input_ids):
    """
    Computes log probabilities for specific tokens in the vocabulary.

    Args:
        logits (torch.Tensor): The raw logits output from the model.
        input_ids (torch.Tensor): The token IDs for which we want the log probabilities.

    Returns:
        torch.Tensor: Log probabilities of the selected tokens.

    Explanation:
        1. Applies log softmax to convert logits to log probabilities over the vocabulary.
        2. Uses gather to extract only the log probabilities corresponding to the input_ids.
        3. Removes the extra dimension to match the original shape of input_ids.
    """
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


def compute_log_probs(model, input_ids, attention_mask, logits_to_keep):
    """
    Computes the log probabilities for a batch of tokens.

    Args:
        model: The language model.
        input_ids (torch.Tensor): Token IDs for input sequences.
        attention_mask (torch.Tensor): Attention mask for input sequences.
        logits_to_keep (int): Number of tokens to keep from the end of the sequence.

    Returns:
        torch.Tensor: Log probabilities of the selected tokens.

    Explanation:
        1. Gets logits from the model for the input sequence.
        2. Selects logits for all tokens except the last one (as we predict next tokens).
        3. Selects only the last 'logits_to_keep' tokens from both logits and input_ids.
        4. Computes log probabilities for these tokens using selective_log_softmax.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, input_ids)


def create_completion_mask(completion_ids, eos_token_id):
    """
    Creates a mask for completion tokens that excludes tokens after the EOS token.

    Args:
        completion_ids (torch.Tensor): Token IDs of the generated completions.
        eos_token_id (int): The ID of the end-of-sequence token.

    Returns:
        torch.Tensor: A binary mask with 1s for valid tokens and 0s after the EOS token.

    Explanation:
        1. Identifies positions where EOS tokens occur in each sequence.
        2. Finds the index of the first EOS token in each sequence.
        3. Creates a mask where positions before and including the first EOS are 1, others are 0.
        4. If no EOS token is found in a sequence, all positions are set to 1.
    """
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (sequence_indices <= eos_idx.unsqueeze(1)).int()


def generate_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=32):
    """
    Generates multiple completions for each prompt.

    Args:
        model: The language model.
        tokenizer: The tokenizer for encoding and decoding text.
        prompts (list): List of text prompts.
        num_generations (int): Number of completions to generate per prompt.
        max_completion_length (int): Maximum number of tokens to generate.

    Returns:
        tuple: Containing prompt IDs, prompt mask, completion IDs, and completion mask.

    Explanation:
        1. Encodes the prompts and moves them to the appropriate device.
        2. Repeats each prompt num_generations times to generate multiple completions.
        3. Generates completions using the model with specified parameters.
        4. Extracts the completion IDs (excluding the prompt tokens).
        5. Creates a mask for the completions using create_completion_mask.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    print(f"Input batch size: {prompt_ids.size(0)}, Device before model: {prompt_ids.device}")
    prompt_length = prompt_ids.size(1)
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    outputs = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=False
    )
    print(f"Output batch size: {outputs.size(0)}, Device after model: {outputs.device}")
    completion_ids = outputs[:, prompt_length:]
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    return prompt_ids, prompt_mask, completion_ids, completion_mask


def generate_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length):
    """
    Generates data for GRPO rollouts including completions and log probabilities.

    Args:
        model: The policy model being trained.
        ref_model: The reference model for KL divergence calculation.
        tokenizer: The tokenizer for encoding and decoding text.
        batch_samples (list): Batch of training samples.
        num_generations (int): Number of completions to generate per sample.
        max_completion_length (int): Maximum completion length.

    Returns:
        dict: Dictionary containing all data needed for GRPO updates.

    Explanation:
        1. Extracts prompts and expected answers from the batch samples.
        2. Generates completions using the current policy model.
        3. Combines prompt and completion tokens.
        4. Computes log probabilities from both the policy model and reference model.
        5. Formats completions for reward calculation.
        6. Repeats prompts and answers to match the number of generated completions.
        7. Returns all data needed for GRPO loss calculation.
    """
    # 具体单个case可以为字典{"prompt":"xxxxxx","answer":"xxxxxx"},也可以为数组方式["prompt","answer"]
    prompts = [sample["prompt"] if isinstance(sample, dict) else sample[0] for sample in batch_samples]
    answers = [sample["answer"] if isinstance(sample, dict) else sample[1] for sample in batch_samples]

    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            model, tokenizer, prompts, num_generations, max_completion_length
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        old_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
        ref_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep)
    formatted_completions = [[{'content': tokenizer.decode(ids, skip_special_tokens=True)}] for ids in completion_ids]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted_completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": logits_to_keep,
        "batch_size": len(prompts),
        "num_generations": num_generations
    }

def grpo_loss(model, ref_model, rollout_data, tokenizer, reward_function, beta=0.01, epsilon=0.2):
    """
    Computes the GRPO loss for updating the policy model.

    Args:
        model: The policy model being trained.
        ref_model: The reference model for KL divergence calculation.
        rollout_data (dict): Data generated by generate_rollout_data.
        tokenizer: The tokenizer for encoding and decoding text.
        reward_function: Function that calculates rewards for completions.
        beta (float): KL penalty coefficient.
        epsilon (float): Clipping parameter for PPO.

    Returns:
        torch.Tensor: The GRPO loss to be minimized.

    Explanation:
        1. Computes current token log probabilities using the policy model.
        2. Calculates the probability ratio between current and old policies.
        3. Computes rewards using the provided reward_function.
        4. Calculates advantages by standardizing rewards within each prompt.
        5. Computes the PPO surrogate objective with clipping.
        6. Calculates the KL divergence between reference and policy models.
        7. Combines surrogate loss and KL penalty.
        8. Averages the loss across all tokens and batches.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    token_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
    ratio = torch.exp(token_log_probs - old_log_probs)
    rewards = torch.tensor(
        reward_function(prompts=rollout_data["repeated_prompts"], completions=rollout_data["formatted_completions"], answer=rollout_data["repeated_answers"]),
        dtype=torch.float32,
        device=device
    )
    #print(f"Rewards: {rewards}")  # Debug rewards
    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]
    rewards = rewards.view(batch_size, num_generations)
    avg_reward = rewards.mean().item()
    print("Average Reward:", avg_reward)
    mean_rewards = rewards.mean(dim=1).repeat_interleave(num_generations)
    std_rewards = rewards.std(dim=1).repeat_interleave(num_generations)
    advantages = ((rewards.view(-1) - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate_loss = torch.min(surr1, surr2)
    kl = torch.exp(ref_log_probs - token_log_probs) - (ref_log_probs - token_log_probs) - 1
    per_token_loss = surrogate_loss - beta * kl
    loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss, avg_reward
    
def train_with_grpo(model, tokenizer, train_data, num_iterations=1, num_steps=500, batch_size=4,
                    num_generations=4, max_completion_length=128, beta=0.1,
                    learning_rate=5e-6, mu=3, epsilon=0.2, reward_function=None, device_ids=None):
    """
    进行强化学习微调:为从头开始实现 GRPO 的训练函数 train_with_grpo 配置适当的训练参数和奖励函数。然后，在剩余的训练数据上执行强化学习训练。
    This function is your original working code (train_with_grpo_static)
    with an added outer loop for iterative GRPO updates per the pseudocode.

    Args:
        model: The language model to train.
        tokenizer: The tokenizer for encoding and decoding text.
        train_data (list): Training dataset.
        num_iterations (int): Number of outer iterations (reference model updates).
        num_steps (int): Number of batch updates per iteration.
        batch_size (int): Number of prompts per batch.
        num_generations (int): Number of completions per prompt.
        max_completion_length (int): Maximum token length for completions.
        beta (float): KL penalty coefficient.
        learning_rate (float): Learning rate for optimizer.
        mu (int): Number of policy updates per batch.
        epsilon (float): PPO clipping parameter.
        reward_function: Function that calculates rewards for completions.
        device_ids (list): List of GPU device IDs for DataParallel.

    Returns:
        The trained model.

    Explanation:
        1. For each outer iteration:
           - Creates a reference model as a deep copy of the current policy model.
           - Reinitializes the optimizer for the policy model.
           - For each training step:
             a. Samples a batch of examples from the training data.
             b. Generates rollout data including completions and log probabilities.
             c. For mu iterations:
                i. Computes the GRPO loss.
                ii. Updates the policy model using gradient descent.
           - Monitors GPU memory usage and prints progress information.
    """
    assert device_ids is not None and len(device_ids) > 1, "This code needs at least 2 GPU cores to run!"

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Wrap model with DataParallel if multiple GPUs are available.

    model = nn.DataParallel(model, device_ids=device_ids)
    print(f"Model wrapped with DataParallel across GPUs: {device_ids}")

    # Outer loop: iterative GRPO updates.
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")

        # Create a reference model (deep copy) and set it to eval mode.
        ref_model = copy.deepcopy(model.module)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        print("Reference model created.")

        # Reinitialize the optimizer for this iteration.
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model.train()

        # Inner loop: your original training steps.
        for step in range(num_steps):
            # 训练集（train_data）中随机选择batch_size大小数据进行训练
            batch_samples = random.sample(train_data, batch_size)
            with torch.no_grad():
                rollout_data = generate_rollout_data(
                    model.module,
                    ref_model,
                    tokenizer,
                    batch_samples,
                    num_generations,
                    max_completion_length
                )

            for grpo_iter in range(mu):
                loss, avg_reward = grpo_loss(
                    model.module,
                    ref_model,
                    rollout_data,
                    tokenizer,
                    reward_function,
                    beta=beta,
                    epsilon=epsilon
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                # Log to wandb
                wandb.log({
                    "loss": loss.item(),
                    "average_reward": avg_reward,
                    "iteration": iteration + 1,
                    "step": step + 1,
                    "grpo_iter": grpo_iter + 1
                })
                print(f"Iteration {iteration + 1}/{num_iterations}, Step {step + 1}/{num_steps}, "
                      f"GRPO iter {grpo_iter + 1}/{mu}, loss: {loss.item():.4f}")
                # for i in range(torch.cuda.device_count()):
                #    print(f"GPU {i} Usage: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MiB, "
                #          f"Utilization: {torch.cuda.utilization(i)}%")
                # Uncomment to see the GPU utilization stats
    return model.module
