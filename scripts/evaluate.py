#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#
#
"""
Setup script.
@File:  evaluate.py
Authors: shajiu
Date:    2025/3/9 17:27
@Software:  PyCharm
功能为： 用于评估模型性能，可能包括加载模型、计算评估指标等
评估函数:
1.评估对于跟踪模型的进展至关重要。因此这里定义了一些函数，从而可以在一组示例上对模型进行评估。项目的评估函数执行以下任务;
2.token 化提示并生成响应：模型的输出是在token化提示的基础上生成的。
3.提取预测答案：从生成的响应中提取答案。将预测答案与预期答案进行比较：这种比较是通过精确匹配以及数值等价检查来完成的。
4.在这段代码中，两个辅助函数 _extract_last_number 和 _extract_single_number 被用来从文本中提取数字。
5.评估函数 evaluate_model 使用这些辅助函数来确定预测答案是否正确：
"""
import torch
import re

from dataset.preprocess import extract_answer_from_model_output

def extract_single_number(text):
    """
    Extracts a single number from text if exactly one number is present.

    Args:
        text (str): The text to extract a number from.

    Returns:
        float or None: The single number in the text, or None if zero or multiple numbers are found.

    Explanation:
        1. Uses regex to find all numbers in the text (including negative numbers and decimals).
        2. If exactly one number is found, returns it as a float.
        3. If zero or multiple numbers are found, returns None.
    """
    numbers = re.findall(r'-?\d*\.?\d+', text)
    return float(numbers[0]) if len(numbers) == 1 else None


def extract_last_number(text):
    """
    Extracts the last number appearing in the text.

    Args:
        text (str): The text to extract a number from.

    Returns:
        float or None: The last number in the text, or None if no number is found.

    Explanation:
        1. Removes dollar signs and percent symbols from the text.
        2. Uses regex to find a number that appears at the end of the text (possibly after whitespace).
        3. The pattern matches numbers that appear at the end of the string, with or without decimal points.
        4. Returns the found number as a float, or None if no match is found.
    """
    text = text.replace('$', '').replace('%', '')
    pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def evaluate_model(model, tokenizer, eval_examples, device):
    """
    Evaluates the model on a set of examples and prints detailed results.

    Args:
        model: The language model to evaluate.
        tokenizer: The tokenizer for encoding inputs and decoding outputs.
        eval_examples (list): List of evaluation examples, each containing "prompt" and "answer".
        device: The device (CPU or GPU) to run evaluation on.

    Returns:
        float: The accuracy percentage (correct predictions / total examples * 100).

    Explanation:
        1. Sets the model to evaluation mode.
        2. For each example in the evaluation set:
           - Encodes the prompt and generates a response using the model.
           - Extracts the predicted answer from the generated response.
           - Compares the predicted answer with the expected answer using multiple methods:
             a. Exact string matching
             b. Single number extraction and comparison
             c. Last number extraction and comparison
           - Prints detailed information about each example.
        3. Calculates and returns the overall accuracy.
        4. Returns the model to training mode.
    """
    model.eval()
    correct = 0
    total = len(eval_examples)
    print("\n" + "=" * 50)
    print("EVALUATION ON", total, "EXAMPLES")
    print("=" * 50)

    for example in eval_examples:
        # Get the prompt and expected answer
        full_prompt = example["prompt"]
        expected = example["answer"]

        # Tokenize and generate response
        inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                forced_eos_token_id=tokenizer.eos_token_id,
                early_stopping=False,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            # Extract answer and check correctness
            predicted = extract_answer_from_model_output(response)

            # Try different matching methods(输出答案完全一致)
            if predicted == expected:  # Exact match
                is_correct = True
            else:
                # Try single number matching(生成的答案数字部分一致)
                pred_num = extract_single_number(str(predicted))
                exp_num = extract_single_number(str(expected))
                if pred_num is not None and exp_num is not None and pred_num == exp_num:
                    is_correct = True
                else:
                    # Try last number matching(生成的最终的答案是否一致)
                    pred_num = extract_last_number(str(predicted))
                    exp_num = extract_last_number(str(expected))
                    is_correct = (pred_num is not None and exp_num is not None and
                                  pred_num == exp_num)

            # Update counter for correct answers
            if is_correct:
                correct += 1

            # Print evaluation details
            print("\nPrompt:")
            print(full_prompt)
            print("\nExpected Answer:")
            print(expected)
            print("\nExtracted Answer:")
            print(predicted)
            print("\nFull Generated Response:")
            print(response)
            print("\nCorrect:", "✓" if is_correct else "✗")
            print("-" * 50)

        except Exception as e:
            print("\nFailed to parse model output for prompt:")
            print(full_prompt)
            print("Error:", e)
            print("-" * 50)

    # Calculate and print final accuracy
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    print("=" * 50)

    # Return model to training mode
    model.train()
    return accuracy

