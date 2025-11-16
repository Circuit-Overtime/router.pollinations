from llama_cpp import Llama
import json, time

MODEL_PATH = "models/Phi-3.5-mini-instruct-Q4_K_M.gguf"

print("Loading model...")
t0 = time.time()
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_batch=512,
    n_threads=6,
    n_gpu_layers=-1,
    flash_attn=True,
    verbose=False
)
print(f"[READY] load={time.time()-t0:.2f}s")


SYSTEM = (
    "You are a routing engine. "
    "Return VALID JSON ONLY with:\n"
    "{"
    "\"tasks\": {\"text\":\"\",\"image\":\"\",\"audio\":\"\",\"web\":\"\"}, "
    "\"reason\": \"short\""
    "}"
)


def run_inference(q):
    prompt = (
        "<|system|>\n" + SYSTEM + "\n"
        "<|user|>\n" + q + "\n"
        "<|assistant|>\n"
    )

    t1 = time.time()
    out = llm(
        prompt,
        max_tokens=120,
        temperature=0.0,
        stop=["<|user|>", "<|system|>"]
    )
    latency = time.time() - t1

    txt = out["choices"][0]["text"].strip()

    # Basic fixing
    if "}" not in txt:
        txt += "}"

    try:
        data = json.loads(txt)
    except:
        data = {"tasks": {"text": q, "image":"", "audio":"", "web":""}, "reason": "fallback"}

    data["latency"] = latency
    print(json.dumps(data, indent=2))
    print(f"[latency] {latency:.2f}s")


if __name__ == "__main__":
    run_inference("what is the capital of france?")
