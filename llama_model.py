import base64
import json
import requests
import mimetypes
from pathlib import Path

class OllamaModel:
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

        keep_alive = kwargs.get("keep_alive")

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

        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        r = self.session.post(self.ollama_url, json=payload, timeout=300)

        if r.status_code != 200:
            raise RuntimeError(f"Ollama error {r.status_code}: {r.text}")
        
        out = r.json()

        content = out["message"]["content"]

        return json.loads(content)


class LlamaCppModel:
    def __init__(
        self,
        model_name: str,
        server_url: str = "http://localhost:8080/v1/chat/completions",
    ):
        self.model_name = model_name
        self.server_url = server_url
        self.session = requests.Session()
        self.session.trust_env = False

    def _b64_image(self, path: str) -> str:
        data = Path(path).read_bytes()
        return base64.b64encode(data).decode("utf-8")

    def _image_data_url(self, path: str) -> str:
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            mime_type = "image/jpeg"
        img_b64 = self._b64_image(path)
        return f"data:{mime_type};base64,{img_b64}"

    def extract_jersey_information(self, image_path: str, system_prompt: str, **kwargs) -> dict:
        """
        Extract jersey information from an image using a llama.cpp server.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        system_prompt : str
            System prompt to send to the model.
        **kwargs : dict
            prompt : str, optional
                User prompt for the model.
            format : dict, optional
                JSON schema-like format specification for output.
            options : dict, optional
                Sampling options such as temperature, top_p, etc.
            max_tokens : int, optional
                Max number of generated tokens.
            timeout : int | float, optional
                Request timeout in seconds.

        Returns
        -------
        dict
            Parsed JSON output from the model.
        """
        prompt = kwargs.get(
            "prompt",
            "Follow the system prompt to extract jersey information from the image."
        )

        output_format = kwargs.get(
            "format",
            {
                "type": "object",
                "properties": {
                    "jerseys": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "jersey_number": {"type": ["integer", "null"]},
                                "last_name": {"type": ["string", "null"]},
                                "jersey_color": {"type": ["string", "null"]},
                                "number_color": {"type": ["string", "null"]},
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 100,
                                },
                            },
                            "required": [
                                "jersey_number",
                                "last_name",
                                "jersey_color",
                                "number_color",
                                "confidence",
                            ],
                        },
                    }
                },
                "required": ["jerseys"],
            },
        )

        options = kwargs.get("options", {"temperature": 0.0})
        timeout = kwargs.get("timeout", 300)
        max_tokens = kwargs.get("max_tokens", 2000)

        image_data_url = self._image_data_url(image_path)

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                    ],
                },
            ],
            "temperature": options.get("temperature", 0.0),
            "max_tokens": max_tokens,
            "stream": False,
            # Works only if the llama.cpp server/model supports this mode.
            "response_format": {
                "type": "json_object",
                "json_schema": output_format
            },
        }

        # Map a few common optional generation args through.
        for key in ("top_p", "top_k", "min_p", "repeat_penalty", "seed"):
            if key in options:
                payload[key] = options[key]

        response = self.session.post(self.server_url, json=payload, timeout=timeout)

        if response.status_code != 200:
            raise RuntimeError(f"llama.cpp error {response.status_code}: {response.text}")

        out = response.json()

        try:
            content = out["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected llama.cpp response format: {out}") from exc

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = "\n".join(text_parts).strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model did not return valid JSON:\n{content}") from exc