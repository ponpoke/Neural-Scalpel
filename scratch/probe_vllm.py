from vllm import LLM
llm = LLM(model="Qwen/Qwen2.5-0.5B", enforce_eager=True)
print("LLM Engine attributes:", dir(llm.llm_engine))
if hasattr(llm.llm_engine, 'driver_worker'):
    print("Driver worker attributes:", dir(llm.llm_engine.driver_worker))
