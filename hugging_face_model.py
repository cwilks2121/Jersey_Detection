import json
from pathlib import Path
from PIL import Image
import torch
from transformers import DataProcessor, AutoModelForVision2Seq

class HuggingFaceModel:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.model_name = model_name

        # Device selection
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Dtype selection
        if dtype is None:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.dtype = dtype

        # Load processor + model
        self.processor = DataProcessor.from_pretrained(model_name)

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)

        self.model.eval()

    def extract_jersey_information(self, image_path: str, prompt: str, **kwargs) -> dict:
        """
        Parameters
        ----------
        image_path : str
        prompt : str
        **kwargs:
            system_prompt: str (optional)
            format: dict or "json" (optional)
            max_new_tokens: int (optional)
            temperature: float (optional)
        """
        system_prompt = kwargs.get(
            "system_prompt",
            "You are a strict OCR engine. Extract jersey NUMBER and LAST NAME. Return JSON only."
        )
        fmt = kwargs.get("format", {
            "type": "object",
            "properties": {
                "number": {"type": ["integer", "null"]},
                "last_name": {"type": ["string", "null"]},
                "confidence": {"type": "number"}
            },
            "required": ["number", "last_name", "confidence"]
        })

        max_new_tokens = int(kwargs.get("max_new_tokens", 200))
        temperature = float(kwargs.get("temperature", 0.0))

        # Load image
        image = Image.open(Path(image_path)).convert("RGB")

        # Ask for JSON with a schema “hint” embedded in text (works across most HF VLMs)
        schema_hint = ""
        if isinstance(fmt, dict):
            schema_hint = f"\nReturn JSON that matches this schema:\n{json.dumps(fmt)}\n"
        elif isinstance(fmt, str) and fmt.lower() == "json":
            schema_hint = "\nReturn valid JSON only.\n"

        full_prompt = f"{system_prompt}\n{schema_hint}\nUser:\n{prompt}"

        # Many VLM processors accept text+images like this:
        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt"
        )

        # Move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
        }
        # Remove None values to avoid warnings
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.no_grad():
            out_tokens = self.model.generate(**inputs, **gen_kwargs)

        text = self.processor.batch_decode(out_tokens, skip_special_tokens=True)[0]

        # Try to locate JSON in the model output robustly
        json_str = self._extract_json(text)
        return json.loads(json_str)

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Pull the first top-level JSON object from a model response.
        """
        start = text.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in output:\n{text}")

        # Simple brace-matching
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]

        raise ValueError(f"Unbalanced JSON braces in output:\n{text}")
