#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#
#
"""
Setup script.
@File:   predict.py
Authors: shajiu
Date:    2025/3/9 17:27
@Software:  PyCharm
功能为： 实现模型预测功能，输入新数据，输出预测结果。

"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from dataset.dataset_loader import build_prompt
from dataset.preprocess import SYSTEM_PROMPT, extract_answer_from_model_output
from config.config import parse_args

def inference(saved_model_path, prompts_to_test):
    """
    :param saved_model_path: string型；训练好的模型路径
    :param prompts_to_test:  list型；测试数据集
    :return:
    Main function to load the fine-tuned model and test it on example math problems.
    Explanation:
    1. Determines the device (GPU if available, otherwise CPU).
    2. Loads the fine-tuned model and tokenizer from the saved path.
    3. Tests the model on predefined math problems.
    4. Formats the prompt using the same SYSTEM_PROMPT and build_prompt function as training.
    5. Generates and displays responses for each test prompt.
    """

    # Determine the device: use GPU if available, else fallback to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    loaded_model = AutoModelForCausalLM.from_pretrained(
        saved_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    loaded_tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
    loaded_tokenizer.pad_token = loaded_tokenizer.eos_token

    # Test each prompt
    for prompt in prompts_to_test:
        # Prepare the prompt using the same format as during training
        test_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        # content中的内容转换为字符串格式待输入
        test_prompt = build_prompt(test_messages)

        # Tokenize the prompt and generate a response
        test_input_ids = loaded_tokenizer.encode(test_prompt, return_tensors="pt").to(device)

        # Generate response with similar parameters to those used in training
        with torch.no_grad():
            test_output_ids = loaded_model.generate(
                test_input_ids,
                max_new_tokens=400,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=loaded_tokenizer.pad_token_id,
                eos_token_id=loaded_tokenizer.eos_token_id,
                do_sample=True,
                early_stopping=False
            )

        test_response = loaded_tokenizer.decode(test_output_ids[0], skip_special_tokens=True)

        # Print the test prompt and the model's response
        print("\nTest Prompt:")
        print(test_prompt)
        print("\nModel Response:")
        print(test_response)
        # Extract and display the answer part for easier evaluation
        try:
            # 最终输出仅仅为答案的部分(</answer>标签后的内容)
            extracted_answer = extract_answer_from_model_output(test_response)
            print("\nExtracted Answer:")
            print(extracted_answer)
            print("-" * 50)
        except Exception as e:
            print(f"\nFailed to extract answer: {e}")
            print("-" * 50)

def main(args):
    # Load the saved model and tokenizer
    saved_model_path =args.output_dir
    # Define test prompts
    prompts_to_test = [
        "How much is 1+1?",
        "I have 3 apples, my friend eats one and I give 2 to my sister, how many apples do I have now?",
        "Solve the equation 6x + 4 = 40"
    ]
    inference(saved_model_path, prompts_to_test)

if __name__ == "__main__":
    args = parse_args()
    main(args)