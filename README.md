# Setup


0. git repo
git clone 
cd loopaware-agent


1. If you have not download uv: 
curl -LsSf https://astral.sh/uv/install.sh | sh

2. update uv env based on pyproject.toml & uv.lock
uv sync

3. how to run script (test if model works)
uv run python try.py

4. how to run vllm + ouro (tes if compatible)
vllm serve ByteDance/Ouro-2.6B-Thinking   --dtype bfloat16   --max-model-len 8192   --trust-remote-code

use following to test:

curl http://localhost:8000/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "ByteDance/Ouro-2.6B-Thinking",
    "prompt": "Explain what a recurrent transformer is in one sentence.",
    "max_tokens": 32,
    "temperature": 0
  }'


# Bugs

1. When doing try.py
For missing key_cahche, value_cache using following command:
sed -i \
's/self\.key_cache/self.k_cache/g; s/self\.value_cache/self.v_cache/g' \
/workspace/.cache/huggingface/modules/transformers_modules/Bytedance/Ouro-2.6B-Thinking/f1edd81e7ac41355db670500ceaf204e0f73af68/modeling_ouro.py

2. When doing vllm+ouro, vllm=0.11.2 works, if the version goes up, it does not support ouro


