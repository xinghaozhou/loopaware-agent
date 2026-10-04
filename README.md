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

5. Any modification to vllm 
cd vendor/vllm
git add vllm/model_executor/models/ouro.py
git commit -m <message>
cd /root/loopaware-agent
git status
git add vendor/vllm

# Bugs

1. When doing try.py
For missing key_cahche, value_cache using following command:
sed -i \
's/self\.key_cache/self.k_cache/g; s/self\.value_cache/self.v_cache/g' \
/workspace/.cache/huggingface/modules/transformers_modules/Bytedance/Ouro-2.6B-Thinking/f1edd81e7ac41355db670500ceaf204e0f73af68/modeling_ouro.py

2. When doing vllm+ouro, vllm=0.11.2 works, if the version goes up, it does not support ouro


# working notes
1. To see the output structure: 
print(out.outputs[0].__dict__.keys())

It looks like: 
dict_keys(['index', 'text', 'token_ids', 'cumulative_logprob', 'logprobs', 'finish_reason', 'stop_reason', 'lora_request'])

'test': the genereated text

2. git clone branch v0.11.2 and git switch to this branch for our implementation (scratch)
git clone --branch v0.11.2   https://github.com/vllm-project/vllm.git   vendor/vllm
git switch -c loopaware-vllm

Sanity check: /root/loopaware-agent/vendor/vllm/vllm/model_executor/models/ouro.py

uv run python - <<'PY'
import inspect
from vllm.model_executor.models.ouro import OuroModel

print(inspect.getfile(OuroModel))
PY

3. vllm 0.11.2 in vendor, and ouro's exeuction code, now add the recurrence tracer
vendor/vllm/vllm/model_executor/models/ouro.py

4. ouro structure:
  - OuroMLP: MLP after attention
  - OuroAttention: Attention block reside in recurrent block
  - OuroDecoderLayer: Recurrent Block (important)
  - OuroModel: Ouro Structure
  - OuroForCausalLM: quick start use abtract wrapper

5. There is no early stop applied in vllm 0.11.2!!!

6. But there is one in ouro-2.6B-reasoning, called early_exit_threshold

7. Do a trajectory characterization:
  - Count the actual recurrence (expect it to hit the same recurrence config)
  - Count the predicted recurrence (If we apply a threshold to it, does it hit the same recurrence?)
  - vendor/vllm/vllm/model_executor/models/ouro.py

  - 7.1
    - Add trace collector in model_init
    - Shown that it hits the same recurrences

  - 7.2
    - Adding early exit gate to in model forward()
    - Identify the gap between actual & expected recurrence