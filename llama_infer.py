from multiprocessing.managers import BaseManager
import json
import sys
import re

# ---------- MODEL CLIENT ----------
class ModelClient:
    def __init__(self):
        self.manager = None
        self.model = None
        self.connect()
    
    def connect(self):
        class ModelServer(BaseManager):
            pass
        
        ModelServer.register('get_model')
        
        try:
            self.manager = ModelServer(address=('localhost', 7002), authkey=b'moe_model_key')
            self.manager.connect()
            self.model = self.manager.get_model()
            print("Connected to model server successfully!")
        except Exception as e:
            print(f"Failed to connect to model server: {e}")
            print("Make sure model_server.py is running!")
            sys.exit(1)
    
    def inference(self, prompt, max_tokens=100, temperature=0.1, stop=None):
        return self.model.inference(prompt, max_tokens, temperature, stop)

# Global client instance
client = ModelClient()

# ---------- SYSTEM PROMPT ----------
SYSTEM_MOE_PROMPT = """You are a JSON router. Reply ONLY with valid JSON.

Format:
{"tasks":{"text":"query or null","image":"prompt or null","audio":"prompt or null","web":"query or null"},"final_decision":"text"}

Rules:
- text: for knowledge questions
- image: for image generation  
- audio: for speech/sound
- web: for current info only
- Use null for unused fields"""

# ---------- JSON FIXING ----------
def fix_json(raw_output):
    """Try to fix common JSON issues"""
    # Add missing opening brace
    if not raw_output.strip().startswith('{'):
        raw_output = '{' + raw_output
    
    # Add missing closing brace
    if not raw_output.strip().endswith('}'):
        raw_output = raw_output + '}'
    
    # Fix null values without quotes
    raw_output = re.sub(r':\s*null\s*([,}])', r': null\1', raw_output)
    
    return raw_output.strip()

# ---------- RUN INFERENCE ----------
def run_moe(query: str):
    prompt = f"""{SYSTEM_MOE_PROMPT}

Query: {query}

JSON:"""

    raw = client.inference(
        prompt,
        max_tokens=80,
        temperature=0.1,
        stop=["\n", "Query:", "JSON:"]
    )

    # Handle error responses
    if isinstance(raw, dict) and "error" in raw:
        return raw

    # Clean and fix JSON
    raw = raw.strip()
    raw = fix_json(raw)
    
    # Try to validate JSON
    try:
        parsed = json.loads(raw)
        return parsed
    except json.JSONDecodeError as e:
        # Return a fallback valid JSON for text queries
        return {
            "tasks": {
                "text": query,
                "image": None,
                "audio": None, 
                "web": None
            },
            "final_decision": "text",
            "error": f"JSON parse error: {str(e)}",
            "raw_output": raw
        }

# ---------- MAIN ENTRY POINT ----------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What's the capital of France?"
    
    print(f"Query: {query}")
    output = run_moe(query)
    print(json.dumps(output, indent=2))