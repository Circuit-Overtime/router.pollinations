from multiprocessing.managers import BaseManager
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import json
import os

SYSTEM_MOE_PROMPT = """You are a router that decides which tools to use and provides prompts for each tool.
- text: Use for answering questions, explanations, summaries, or any task that requires language or general knowledge.
- image: Use for requests involving visual content, such as generating, describing, or editing images, diagrams, or visualizations.
- audio: Use for tasks involving sound, such as generating, transcribing, or analyzing audio, music, or speech.
- web: For real-time/external information (only when user explicitly asks or when current events/real-time data is needed)

Output ONLY valid JSON:
{
        "tasks": {
                "text": "<prompt or null>",
                "image": "<prompt or null>",
                "audio": "<prompt or null>"
        }
}
"""

class ModelManager:
        def __init__(self):
                self.tokenizer = None
                self.model = None
                self.device = None
                self.load_model()
                self.build_system_prefix()

        def load_model(self):
                print("Loading model and tokenizer...")
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Using device: {self.device}")
                
                MODEL_PATH = "models/phi1"
                
                # Check if model path exists
                if not os.path.exists(MODEL_PATH):
                        print(f"❌ Model path not found: {MODEL_PATH}")
                        print("📥 Attempting to download model...")
                        from download_model import download_phi1_model
                        if not download_phi1_model():
                                raise FileNotFoundError(f"Failed to download model to {MODEL_PATH}")
                
                print(f"Loading from: {MODEL_PATH}")
                
                # List files in directory
                files = os.listdir(MODEL_PATH)
                print(f"Files found: {files}")
                
                try:
                        # Load tokenizer
                        print("Loading tokenizer...")
                        self.tokenizer = AutoTokenizer.from_pretrained(
                                MODEL_PATH,
                                trust_remote_code=True
                        )
                        print("✅ Tokenizer loaded successfully")
                except Exception as e:
                        print(f"Error loading tokenizer from local: {e}")
                        print("Downloading tokenizer from HuggingFace...")
                        self.tokenizer = AutoTokenizer.from_pretrained(
                                "microsoft/phi-1",
                                trust_remote_code=True
                        )
                        print("✅ Tokenizer downloaded and loaded")
                
                # Load model
                try:
                        print("Loading model...")
                        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                        
                        self.model = AutoModelForCausalLM.from_pretrained(
                                MODEL_PATH,
                                dtype=dtype,
                                trust_remote_code=True,
                                device_map=None,  # Manual device mapping
                        ).to(self.device)
                        
                        print("✅ Model loaded successfully")
                        
                except Exception as e:
                        print(f"Error loading model: {e}")
                        raise

        def build_system_prefix(self):
                self.system_prefix = (
                        f"<|system|>\n{SYSTEM_MOE_PROMPT}\n"
                        f"<|user|>\n"
                )

        def fast_inference(self, user_msg, max_tokens=60, temperature=0.1, top_p=0.8):
                full_prompt = (
                        self.system_prefix
                        + user_msg
                        + "\n<|assistant|>\n"
                )

                try:
                        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
                        
                        with torch.no_grad():
                                generation_output = self.model.generate(
                                        **inputs,
                                        max_new_tokens=max_tokens,
                                        temperature=temperature,
                                        top_p=top_p,
                                        do_sample=True,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                )
                        
                        response = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)
                        
                        # Extract only the assistant's response
                        if "<|assistant|>" in response:
                                response = response.split("<|assistant|>")[-1].strip()
                        
                        return response
                
                except Exception as e:
                        print(f"Error during inference: {e}")
                        return json.dumps({"error": str(e)})


model_manager = ModelManager()

class ModelServer(BaseManager):
        pass

ModelServer.register("get_model", callable=lambda: model_manager)

if __name__ == "__main__":
        port = 7002
        print(f"Starting model server on port {port}")
        print(f"Using Phi-1 model from models/phi1")
        
        server = ModelServer(address=("localhost", port), authkey=b"moe_model_key")
        server.get_server().serve_forever()