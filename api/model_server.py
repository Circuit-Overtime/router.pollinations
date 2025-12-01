from multiprocessing.managers import BaseManager
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

SYSTEM_PROMPT = """You are a task router that analyzes user requests and assigns them to tools.

Tools:
- text: language tasks, explanations, summaries, reasoning
- image: visual generation or analysis
- audio: sound or voice tasks
- web: real-time or external information (only if explicitly requested)
Return ONLY valid JSON:
{
  "tasks": {
    "text": "<prompt or null>",
    "image": "<prompt or null>",
    "audio": "<prompt or null>"
  }
}
Rules:
- Break complex requests into multiple tasks
- No emojis or special formatting
- Give clear actionable prompts
- Use null when a tool isn't needed
- Output JSON only, no extra text
"""
class ModelManager:
        def __init__(self):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.load_model()

        def load_model(self):
                MODEL_PATH = "models/phi-3.5-mini"
                try:
                        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
                        self.model = AutoModelForCausalLM.from_pretrained(
                                MODEL_PATH,
                                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                                trust_remote_code=True,
                                device_map="auto" if self.device == "cuda" else None
                        )
                except Exception as e:
                        print(f"Failed to load local model: {e}. Downloading fallback...")
                        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1", trust_remote_code=True)
                        self.model = AutoModelForCausalLM.from_pretrained(
                                "microsoft/phi-1",
                                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                                trust_remote_code=True,
                                device_map="auto" if self.device == "cuda" else None
                        )
        def fast_inference(self, user_msg, max_tokens=60, temperature=0.7, top_p=0.8):
                prompt = f"{SYSTEM_PROMPT}\nUser: {user_msg}\nAssistant:"
                try:
                        # Set pad token if not set
                        if self.tokenizer.pad_token is None:
                                self.tokenizer.pad_token = self.tokenizer.eos_token

                        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
                        with torch.no_grad():
                                output = self.model.generate(
                                        **inputs,
                                        max_new_tokens=max_tokens,
                                        temperature=temperature,
                                        top_p=top_p,
                                        do_sample=True,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        use_cache=False
                                )
                                # Decode only the new tokens generated
                                return self.tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                except Exception as e:
                        print(f"Inference error: {e}")
                        return json.dumps({"error": str(e)})




model_manager = ModelManager()

class ModelServer(BaseManager): pass
ModelServer.register("get_model", callable=lambda: model_manager)

if __name__ == "__main__":
    port = 7002
    server = ModelServer(address=("localhost", port), authkey=b"moe_model_key")
    print(f"Phi-1 server running on port {port} (device={model_manager.device})")
    server.get_server().serve_forever()
