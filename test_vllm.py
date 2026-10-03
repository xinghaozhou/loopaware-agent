from vllm import LLM, SamplingParams

llm = LLM(
    model="ByteDance/Ouro-1.4B",
    trust_remote_code=True,
    enforce_eager=True,
)

outputs = llm.generate(
    ["Hello, my name is"],
    SamplingParams(
        max_tokens=20,
        temperature=0,
    ),
)

print(outputs[0].outputs[0].text)