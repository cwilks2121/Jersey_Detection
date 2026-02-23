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

    def extract_jersey_information(self, image_path: str, system_prompt: str, **kwargs) -> dict:
        """
        Docstring for extract_jersey_information
        
        Parameters
        ----------
        image_path : str
            Path to the image file.
        system_prompt : str
            The system prompt to send to the model.
        **kwargs : dict
            payload : dict, optional
                Additional payload data for the request.
            prompt : str, optional
                Prompt for the model.
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

        prompt = kwargs.get("prompt", "Follow the system prompt to extract jersey information from the image.")
        format = kwargs.get("format", {
                                            "type": "object",
                                            "properties": {
                                                "jerseys": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "jersey_number": {
                                                                "type": ["integer", "null"]
                                                            },
                                                            "last_name": {
                                                                "type": ["string", "null"]
                                                            },
                                                            "jersey_color": {
                                                                "type": ["string", "null"]
                                                            },
                                                            "number_color": {
                                                                "type": ["string", "null"]
                                                            },
                                                            "confidence": {
                                                                "type": "number",
                                                                "minimum": 0,
                                                                "maximum": 100
                                                            }
                                                        }
                                                    }    
                                                }
                                            },
                                            "required": ["jerseys"]
                                        }
        )

        options = kwargs.get("options", {
                                            "temperature": 0.0
                                        }
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt, "images": [img_b64]}
            ],
            "format": format,
            "stream": False,
            "options": options
        }

        r = self.session.post(self.ollama_url, json=payload, timeout=300)

        if r.status_code != 200:
            raise RuntimeError(f"Ollama error {r.status_code}: {r.text}")
        
        out = r.json()

        content = out["message"]["content"]

        return json.loads(content)



