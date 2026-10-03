from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Bytedance/Ouro-2.6B-Thinking"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

# Generate with reasoning
messages = [
    {"role": "user", "content": "Solve: If 2x + 3 = 11, what is x?"}
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(inputs, max_new_tokens=512, temperature=1.0, top_p=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
