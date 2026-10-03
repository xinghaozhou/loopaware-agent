# Setup


0. git repo
git clone 
cd loopaware-agent


1. If you have not download uv: 
curl -LsSf https://astral.sh/uv/install.sh | sh

2. update uv env based on pyproject.toml & uv.lock
uv sync

3. how to run script
uv run python try.py

# Bugs

For missing key_cahche, value_cache using following command:
sed -i \
's/self\.key_cache/self.k_cache/g; s/self\.value_cache/self.v_cache/g' \
/workspace/.cache/huggingface/modules/transformers_modules/Bytedance/Ouro-2.6B-Thinking/f1edd81e7ac41355db670500ceaf204e0f73af68/modeling_ouro.py


