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
        print(f"DEBUG: Sending prompt (length: {len(prompt)})")
        result = self.model.inference(prompt, max_tokens, temperature, stop)
        print(f"DEBUG: Raw result: {repr(result)}")
        return result

# Global client instance
client = ModelClient()

# ---------- SIMPLIFIED SYSTEM PROMPT ----------
SYSTEM_MOE_PROMPT = """Reply with JSON only:
{"tasks":{"text":"query","image":null,"audio":null,"web":null},"final_decision":"text"}"""

# ---------- JSON FIXING ----------
def fix_json(raw_output):
    """Try to fix common JSON issues"""
    if not raw_output:
        return '{"tasks":{"text":"default","image":null,"audio":null,"web":null},"final_decision":"text"}'
    
    # Add missing opening brace
    if not raw_output.strip().startswith('{'):
        raw_output = '{' + raw_output
    
    # Add missing closing brace
    if not raw_output.strip().endswith('}'):
        raw_output = raw_output + '}'
    
    return raw_output.strip()

# ---------- RUN INFERENCE ----------
def run_moe(query: str):
    # Much simpler prompt
    prompt = f"""Question: {query}
Reply with JSON:"""

    print(f"DEBUG: Full prompt:\n{prompt}")
    
    raw = client.inference(
        prompt,
        max_tokens=150,
        temperature=0.2,
        stop=["\n\n", "Question:"]
    )

    print(f"DEBUG: Raw response: {repr(raw)}")

    # Handle error responses
    if isinstance(raw, dict) and "error" in raw:
        print(f"DEBUG: Error in response: {raw}")
        return raw

    # Handle empty response
    if not raw or raw.strip() == "":
        print("DEBUG: Empty response detected")
        return {
            "tasks": {
                "text": query,
                "image": None,
                "audio": None, 
                "web": None
            },
            "final_decision": "text",
            "error": "Empty response from model"
        }

    # Clean and fix JSON
    raw = raw.strip()
    raw = fix_json(raw)
    print(f"DEBUG: Fixed JSON: {raw}")
    
    # Try to validate JSON
    try:
        parsed = json.loads(raw)
        return parsed
    except json.JSONDecodeError as e:
        print(f"DEBUG: JSON parse error: {e}")
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
    query = "What is the capital of France?"
    print(f"Query: {query}")
    output = run_moe(query)
    print("Final output:")
    print(json.dumps(output, indent=2))