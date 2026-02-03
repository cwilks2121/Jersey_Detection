from hugging_face_model import HuggingFaceModel
import json

MODEL_NAME = "deepseek-ai/DeepSeek-OCR" 

model = HuggingFaceModel(
    model_name=MODEL_NAME
)

image_path = "1125-8.jpg"

prompt = (
    "Extract the player's jersey NUMBER and LAST NAME from the image. "
    "Return JSON with keys: number, last_name, confidence. "
    "Use uppercase for last_name."
)

result = model.extract_jersey_information(
    image_path=image_path,
    prompt=prompt,
    system_prompt="You are a strict OCR engine. Return JSON only.",
    temperature=0.0,
    max_new_tokens=200
)

print("=== RESULT ===")
print(json.dumps(result, indent=2))