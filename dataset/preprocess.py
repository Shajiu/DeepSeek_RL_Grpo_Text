#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#
#
"""
Setup script.
@File:  preprocess.py
Authors: shajiu
Date:    2025/3/9 17:27
@Software:  PyCharm
功能为：包括所有前处理步骤，例如数据清洗、标准化、编码等。
数据格式以及答案提取
    项目定义了数据格式，以及模型如何从输出和数据集中提取答案段落。为了确保模型输出格式一致，项目还定义了一个系统提示。
    该提示指示模型生成包含 < reasoning > 和 < answer > 标签的输出。这一步通过两个函数完成：
"""

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
"""
Setup script.
@File:  data_formatting_and_answer_extraction.py
Authors: shajiu(shajiu@baidu.com)
Date:    2025/3/3 16:56
@Software:  PyCharm
功能为：数据格式以及答案提取
       项目定义了数据格式，以及模型如何从输出和数据集中提取答案段落。为了确保模型输出格式一致，项目还定义了一个系统提示。
       该提示指示模型生成包含 < reasoning > 和 < answer > 标签的输出。这一步通过两个函数完成：
"""

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_answer_from_model_output(text):
    """
    Extracts the value from the last <answer> tag in the text.
    (此函数获取模型的输出文本，并提取 < answer > 标签内的内容；)
    Args:
        text (str): The model-generated text containing XML-style <answer> tags.

    Returns:
        str or None: The content inside the <answer> tags, or None if no valid answer is found.

    Explanation:
        1. Splits the text on the <answer> tag to isolate content after the tag.
        2. Checks if at least one <answer> tag exists in the text.
        3. For the last <answer> segment:
           - Verifies it contains a closing </answer> tag.
           - Extracts only the content between the tags.
        4. Returns None if the answer is empty (just "...") or if tags are missing.
    """
    # Split on <answer> and take everything after the last occurrence
    parts = text.split("<answer>")
    if len(parts) < 2:  # No <answer> tag found
        return None
    last_part = parts[-1]

    # Extract content up to </answer>
    if "</answer>" not in last_part:
        return None
    answer = last_part.split("</answer>")[0].strip()
    return None if answer == "..." else answer


def extract_answer_from_dataset(text):
    """
    Extracts the answer from the GSM8K dataset examples.
    (此函数从GSM8K数据集中提取预期答案，该数据集使用 “####” 分隔符来分隔答案：)
    Args:
        text (str): The dataset example text containing a question and answer.

    Returns:
        str or None: The extracted answer part after the '####' delimiter, or None if not found.

    Explanation:
        1. Checks if the text contains the '####' delimiter that separates question from answer.
        2. If found, splits the text at this delimiter and returns the second part (the answer).
        3. The answer is stripped of leading/trailing whitespace.
        4. Returns None if no delimiter is present.
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


