from multiprocessing.managers import BaseManager
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os

class ModelManager:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        MODEL_PATH = "models/phi1"
        if not os.path.exists(MODEL_PATH):
            from download_model import download_phi1_model
            if not download_phi1_model():
                raise FileNotFoundError(f"Missing model at {MODEL_PATH}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1", trust_remote_code=True)

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=dtype,
            trust_remote_code=True
        ).to(self.device)

    def fast_inference(self, prompt, max_tokens=100, temperature=0.1, top_p=0.8):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            return json.dumps({"error": str(e)})

model_manager = ModelManager()

class ModelServer(BaseManager): pass
ModelServer.register("get_model", callable=lambda: model_manager)

if __name__ == "__main__":
    port = 7002
    server = ModelServer(address=("localhost", port), authkey=b"moe_model_key")
    server.get_server().serve_forever()
