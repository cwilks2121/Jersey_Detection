import base64
import json
import requests
from pathlib import Path

class colorDetection():
    def __init__(self, model_name: str, ollama_url: str = "http://localhost:11434/api/chat"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.session = requests.Session()
        self.session.trust_env = False
        
    def _b64_image(self, path: str) -> str:
        data = Path(path).read_bytes()
        return base64.b64encode(data).decode("utf-8")
    
    def extract_color_info(self, image_path: str, prompt: str, **kwargs) -> dict:
        
        img_b64 = self._b64_image(image_path)
        
        
        system_prompt = kwargs.get("system_prompt", "Determine the HEX CODE JERSEY COLOR and the raw JERSEY COLOR." \
                                                    "Return JSON only.")
        format = kwargs.get("format", { "type": "object",
                                       "properties": {
                                           "jerseys": {
                                               "items": {
                                                   "type": "object",
                                                   "properties": {
                                                        "hex_code_color": {
                                                            "type": "array",
                                                            "items": {"type": ["string", "null"]}
                                                        },
                                                        "color": {
                                                            "type": "array",
                                                            "items": {"type": ["string", "null"]}
                                                        }
                                                   },
                                                    "required": ["hex_code_color", "color"]
                                               }
                                           }
                                        },
                                        "required": ["colors"]
                                        }
        )
        keep_alive = kwargs.get("keep_alive", "1h")
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt, "images": [img_b64]}
            ],
            "format": format,
            "stream": False,
            "keep_alive": keep_alive,
            # "options": options
        }
       
        r = self.session.post(self.ollama_url, json=payload, timeout=None)

        if r.status_code != 200:
            raise RuntimeError(f"Ollama error {r.status_code}: {r.text}")
        
        out = r.json()

        content = out["message"]["content"]
        print(content)
        return json.loads(content)
