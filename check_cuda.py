from llama_cpp import Llama
import time

# Load model fully on GPU
print("Loading model...")

llm = Llama(
    model_path="models/Phi-3.5-mini-instruct-Q4_K_M.gguf",
    n_gpu_layers=-1,           # full GPU offload
    n_ctx=4096,                # longer context
    verbose=False              # set True for debugging CUDA logs
)

print("Model loaded. Running inference...\n")

# --------------- FAST INFERENCE TEST ---------------
prompt = "Explain quantum computing in one sentence."

# Measure inference time
start = time.time()

output = llm(
    prompt,
    max_tokens=60,
    temperature=0.2,
    top_p=0.9,
    repeat_penalty=1.1,
)

end = time.time()

print("\n=== Response ===")
print(output["choices"][0]["text"].strip())

print(f"\n⏱️ Inference Time: {end - start:.2f} seconds")
