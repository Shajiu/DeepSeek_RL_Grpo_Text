from datasets import load_dataset
import json

data = load_dataset("openai/gsm8k", 'main')
for tpye in ["train","test"]:
    file_output=open(f"/root/paddlejob/workspace/env_run/output/DeepSeek_RL_Grpo/data/{tpye}.jsonl",'w',encoding="utf-8")
    for example in data[tpye]:
        print("*"*100)
        file_output.write(json.dumps(example,ensure_ascii=False)+"\n")