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
    
    def inference(self, prompt, max_tokens=200, temperature=0.2, stop=None):
        return self.model.inference(prompt, max_tokens, temperature, stop)

# Global client instance
client = ModelClient()

SYSTEM_MOE_PROMPT = """
You are a Mixture-of-Experts (MOE) decision router.

Your job is NOT to answer the user's question.
Your ONLY job is to decide *which tools must be used* and provide the PROMPT/QUERY for each tool.

Available tools:
- text  → Use when the model can fully answer from internal knowledge. Provide the PROMPT for text generation.
- image → Use when a visual output is required. Provide the PROMPT for image generation.
- audio → Use when spoken or sound output is requested. Provide the PROMPT for audio generation.
- web   → Use only if the question *clearly requires* real-time, updated, unknown, or external information. Provide the SEARCH QUERY for web search.

IMPORTANT: For each tool, you must provide the PROMPT/QUERY that will be sent to that tool, NOT the answer itself.

You must ALWAYS output STRICT VALID JSON in the following format:

{
  "tasks": {
    "text": "<prompt for text generation or null>",
    "image": "<prompt for image generation or null>", 
    "audio": "<prompt for audio generation or null>",
    "web": "<search query for web search or null>"
  },
  "final_decision": "<one of: text | image | audio | web | combination>"
}

Examples:
- User asks "What is AI?" → text: "Explain what artificial intelligence is"
- User asks "Draw a cat" → image: "A cute cat sitting on a windowsill"
- User asks "Latest news about Tesla stock" → web: "Tesla stock news today latest updates"

Rules:
- If multiple tools are needed, set final_decision to "combination".
- Use null when a field is not required.
- NEVER provide answers, explanations, reasoning, disclaimers, or commentary.
- You ONLY output JSON with PROMPTS/QUERIES for tools. No extra text.
- Always provide the prompt that should be sent to each tool, not the final answer.
"""

# ---------- JSON REPAIR ----------
def fix_json(raw_output):
    """Try to fix common JSON issues while keeping structure intact."""
    if not raw_output:
        return '{"tasks":{"text":null,"image":null,"audio":null,"web":null},"final_decision":"text"}'

    raw_output = raw_output.strip()

    # Force starts/ends if missing
    if not raw_output.startswith('{'):
        raw_output = '{' + raw_output
    if not raw_output.endswith('}'):
        raw_output = raw_output + '}'

    return raw_output

# ---------- MAIN MOE ROUTER ----------
def run_moe(query: str):
    """
    query   → user text
    """
    
    # Construct the prompt for the model
    prompt = f"""
### System:
{SYSTEM_MOE_PROMPT}

### User Query:
{query}

### Assistant (JSON only):
"""

    print(f"\nDEBUG: Sending prompt:\n{prompt}\n")

    raw = client.inference(
        prompt,
        max_tokens=200,
        temperature=0.2,
        stop=["###", "\n\n"]
    )

    print(f"DEBUG: Raw response: {repr(raw)}")

    if isinstance(raw, dict) and "error" in raw:
        return raw

    # Empty output → fallback
    if not raw or raw.strip() == "":
        return {
            "tasks": {
                "text": query,
                "image": None,
                "audio": None,
                "web": None
            },
            "final_decision": "text",
            "error": "Empty output from model"
        }

    fixed = fix_json(raw)
    print(f"DEBUG: Fixed JSON: {fixed}")

    try:
        return json.loads(fixed)
    except Exception as e:
        return {
            "tasks": {
                "text": query,
                "image": None,
                "audio": None,
                "web": None
            },
            "final_decision": "text",
            "error": f"JSON parse error: {str(e)}",
            "raw_output": fixed
        }

# ---------- MAIN ENTRY POINT ----------
if __name__ == "__main__":
    query = "What is the capital of France?"
    
    print(f"Query: {query}")
    out = run_moe(query)
    print(json.dumps(out, indent=2))