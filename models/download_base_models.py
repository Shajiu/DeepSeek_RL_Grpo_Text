from huggingface_hub import snapshot_download

snapshot_download(repo_id="Qwen/Qwen2.5-1.5B-Instruct",    # 模型ID
                  local_dir="Qwen2_5_1_5B_Instruct") # 指定本地地址保存模型
