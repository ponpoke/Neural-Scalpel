from vllm import LLM
try:
    llm = LLM(model="Qwen/Qwen2.5-0.5B", enforce_eager=True)
    print("SUCCESS: Qwen2.5-0.5B loaded")
except Exception as e:
    print(f"FAILED: {e}")
