import os
from vllm import LLM, SamplingParams

def main():
    print("Testing vLLM determinism across batch sizes...")
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B",
        enforce_eager=True,
        disable_log_stats=True,
        dtype="float16",
        gpu_memory_utilization=0.8,
    )
    
    prompt = "Write a short poem about a neural scalpel that swaps intelligence."
    params = SamplingParams(temperature=0.0, max_tokens=32)
    
    print("\nRunning Batch Size 50...")
    out_50 = llm.generate([prompt] * 50, params)
    text_50 = out_50[0].outputs[0].text
    
    print("\nRunning Batch Size 1...")
    out_1 = llm.generate([prompt], params)
    text_1 = out_1[0].outputs[0].text
    
    print("\nText from BS=50:")
    print(repr(text_50))
    print("\nText from BS=1:")
    print(repr(text_1))
    
    if text_50 == text_1:
        print("\nMATCH!")
    else:
        print("\nMISMATCH!")

if __name__ == "__main__":
    main()
