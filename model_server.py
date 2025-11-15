from multiprocessing.managers import BaseManager
from multiprocessing.managers import BaseManager
from llama_cpp import Llama
import json
import sys

class ModelManager:
    def __init__(self):
        self.llm = None
        self.load_model()
    
    def load_model(self):
        print("Loading TinyLlama model...")
        self.llm = Llama(
            model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            n_ctx=1024,
            n_threads=8,
            n_gpu_layers=1,
            verbose=False
        )
        print("Model loaded successfully!")
    
    def inference(self, prompt, max_tokens=250, temperature=0.2, stop=None):
        if self.llm is None:
            return {"error": "Model not loaded"}
        
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or ["###", "\n\n"],
                echo=False,  
                top_p=0.9,
                top_k=40
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            return {"error": str(e)}

# Global model instance
model_manager = ModelManager()

# Register the manager
class ModelServer(BaseManager):
    pass

ModelServer.register('get_model', callable=lambda: model_manager)

if __name__ == "__main__":
    print("Starting Model Server on port 7002...")
    
    server = ModelServer(address=('localhost', 7002), authkey=b'moe_model_key')
    server_obj = server.get_server()
    
    print("Model Server ready! Listening on localhost:7002")
    try:
        server_obj.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Model Server...")
        server_obj.shutdown()