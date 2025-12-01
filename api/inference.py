from multiprocessing.managers import BaseManager
import random
import json
import re

class ModelClient(BaseManager):
    pass

ModelClient.register("get_model")
MODEL_SERVERS = [
    {"address": ("localhost", 7002), "authkey": b"moe_model_key"}
]
models = []


def extract_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {"raw_response": text}



for server_config in MODEL_SERVERS:
    try:
        manager = ModelClient(address=server_config["address"], authkey=server_config["authkey"])
        manager.connect()
        models.append(manager.get_model())
        print(f"Connected to model server at {server_config['address']}")
    except Exception as e:
        print(f"Failed to connect to {server_config['address']}: {e}")

if not models:
    raise Exception("No model servers available")

def get_available_model():
    return random.choice(models)
def inference(prompt):
    model = get_available_model()
    response = model.fast_inference(prompt)    
    result = extract_json(response)
    print(f"Response: {result}")

if __name__ == "__main__":
    test_prompt = "draw me a picture of a bird and explain me aerodynamics"
    inference(test_prompt)