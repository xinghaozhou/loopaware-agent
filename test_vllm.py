from vllm import LLM, SamplingParams

llm = LLM(
    model="ByteDance/Ouro-2.6B-Thinking",
    dtype="bfloat16",
    max_model_len=8192, #context length
    trust_remote_code=True,
    enforce_eager=True,
    hf_overrides={
        "total_ut_steps": 8,
    },
)

outputs = llm.generate(
    ["Explain recurrent transformers."],
    SamplingParams(
        max_tokens=128, # how many tokens can be generated
        temperature=0,
        logprobs=5,
    ),
)

out = outputs[0]


# print(type(out))
# print(out)

# print("request_id:", out.request_id)
# print("prompt:", out.prompt)
# print("prompt_token_ids:", out.prompt_token_ids)

# gen = out.outputs[0]
# print("text:", gen.text)
# print("token_ids:", gen.token_ids)
# print("logprobs:", gen.logprobs)
# print("finish_reason:", gen.finish_reason)