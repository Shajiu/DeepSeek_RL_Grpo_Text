#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#
#
"""
Setup script.
@File:  dataset_loader.py
Authors: shajiu
Date:    2025/3/9 17:27
@Software:  PyCharm
功能为：用于读取数据集，并将数据转换为模型训练、验证或测试所需的格式。
  项目使用 GSM8K(https://huggingface.co/datasets/openai/gsm8k) 数据集进行训练。项目使用了此数据集中的示例来训练模型，基于强化学习（RL）训练范式，让模型生成多个问题解答样本，之后我们将这些解答与GSM8K示例中的标准答案进行对比，如果匹配，就为 RL 算法（GRPO）提供高奖励，然后更新模型权重，以增加模型下次获得高奖励的可能性。详细过程为： 首先从 Hugging Face 加载数据集，然后格式化每个示例，包括系统提示和用户提示。通过定义两个辅助函数去完成：prepare_dataset 以及 build_prompt。
"""
from datasets import load_dataset
from dataset.preprocess import SYSTEM_PROMPT,extract_answer_from_dataset


def prepare_dataset(args,split="train"):
    """
    Load and prepare the GSM8K dataset for training with string prompts.
    若需要加载自己的数据集可以在此自定义加载数据进行更换数据集
    Args:
        split (str): The dataset split to load ("train" or "test"). Defaults to "train".

    Returns:
        list: A list of formatted examples, each containing a prompt string and answer.

    Explanation:
        1. Loads the GSM8K dataset from the Hugging Face datasets hub.
        2. For each example in the dataset:
           - Creates a list of messages with system prompt and the question.
           - Converts this list into a single string prompt using build_prompt().
           - Extracts the answer from the dataset example.
           - Creates a formatted example dictionary with prompt and answer.
        3. Returns the list of formatted examples ready for model training or evaluation.
    """
    data = load_dataset(args.train_path, 'main')[split]
    formatted_data = []
    for example in data:
        # Convert list of messages to a single string prompt.
        prompt_str = build_prompt([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]}
        ])
        formatted_example = {
            "prompt": prompt_str,  # Now a string rather than a list.
            "answer": extract_answer_from_dataset(example["answer"])
        }
        formatted_data.append(formatted_example)
    return formatted_data


def build_prompt(messages):
    """
    Build a single prompt string from a list of messages.

    Args:
        messages (list): A list of message dictionaries, each with 'role' and 'content' keys.

    Returns:
        str: A concatenated string of all message contents.

    Explanation:
        1. Takes a list of message dictionaries in the typical chat format.
        2. Extracts the 'content' field from each message and strips whitespace.
        3. Joins all content strings with newlines to create a single prompt.
        4. This preserves the training format while converting from structured messages to a string.
    """
    return "\n".join([msg["content"].strip() for msg in messages])
