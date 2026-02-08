import base64
import json
import requests
from pathlib import Path

class OllamaModel():
    def __init__(self, model_name: str, ollama_url: str = "http://localhost:11434/api/chat"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.session = requests.Session()
        self.session.trust_env = False

    def _b64_image(self, path: str) -> str:
        data = Path(path).read_bytes()
        return base64.b64encode(data).decode("utf-8")

    def extract_jersey_information(self, image_path: str, prompt: str, **kwargs) -> dict:
        """
        Docstring for extract_jersey_information
        
        Parameters
        ----------
        image_path : str
            Path to the image file.
        prompt : str
            The prompt to send to the model.
        payload : dict
            Additional payload data for the request.
        **kwargs : dict
            system_prompt : str, optional
                System prompt for the model.
            format : dict, optional
                Format specification for the output.
            keep_alive : bool, optional
                Whether to keep the connection alive.
            options : dict, optional
                Additional options for the model.
        
        Returns
        -------
        dict
            The extracted jersey information.
        """
        
        img_b64 = self._b64_image(image_path)

        system_prompt = kwargs.get("system_prompt", "You are a strict OCR engine. Extract JERSEY NUMBER, JERSEY COLOR, and LAST NAME. " \
                                                    "Each list for the key fields must have the same length. Return JSON only.")
        format = kwargs.get("format", { "type": "object",
                                        "properties": {
                                            "number": {
                                                "type": "array",
                                                "items": { "type": ["integer", "null"] }
                                            },
                                            "last_name": { 
                                                "type": "array",
                                                "items": { "type": ["string", "null"] }
                                            },
                                            "color": { 
                                                "type": "array",
                                                "items": { "type": ["string", "null"] }
                                            },
                                            "confidence": { 
                                                "type": "array",
                                                "items": { "type": ["number", "null"] }
                                            },
                                        },
                                        "required": ["number", "last_name", "color", "confidence"]
                                        }
        )
        keep_alive = kwargs.get("keep_alive", "10m")
        options = kwargs.get("options", {"temperature": 0.0})

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt, "images": [img_b64]}
            ],
            "format": format,
            "stream": False,
            "keep_alive": keep_alive,
            "options": options
        }

        r = self.session.post(self.ollama_url, json=payload, timeout=120)

        if r.status_code != 200:
            raise RuntimeError(f"Ollama error {r.status_code}: {r.text}")
        
        out = r.json()

        content = out["message"]["content"]

        return json.loads(content)

