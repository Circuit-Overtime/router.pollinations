from quart import Quart, request, jsonify
from multiprocessing.managers import BaseManager
import asyncio
import json
import random
import re

app = Quart(__name__)

class ModelClient(BaseManager):
    pass

ModelClient.register("get_model")

MODEL_SERVERS = [
    {"address": ("localhost", 7002), "authkey": b"moe_model_key"},
    {"address": ("localhost", 7003), "authkey": b"moe_model_key"},
]

models = []
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
def count_words(text):
    return len(text.split())

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

@app.route("/gen", methods=["GET", "POST"])
async def infer():
    try:
        if request.method == "GET":
            prompt = request.args.get("prompt", "").strip()
        else:
            data = await request.get_json()
            prompt = data.get("prompt", "").strip()
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        word_count = count_words(prompt)
        if word_count > 100:
            return jsonify({"error": f"Prompt exceeds 100 words (current: {word_count})"}), 400
        
        model = get_available_model()
        response = model.fast_inference(prompt)
        
        result = extract_json(response)
        print(f"Response: {result}")
        return jsonify({
            "prompt": prompt,
            "word_count": word_count,
            "result": result
        }), 200
    
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON response from model"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
async def health():
    return jsonify({"status": "ok", "models_connected": len(models)}), 200
    
async def trial_run():
    test_prompt = "Give me a picture of monalisa in the style of van gogh "
    model = get_available_model()
    response = model.fast_inference(test_prompt)
    print("Trial run response:", response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000, workers=10)
    # asyncio.run(trial_run())
