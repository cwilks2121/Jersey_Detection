from ollama_model import OllamaModel

model = OllamaModel(model_name="qwen3-vl")

prompt = (
    "You are doing jersey OCR. Extract ONLY the player's LAST NAME, JERSEY NUMBER, and JERSEY COLOR.\n"
    "Return JSON with keys: number (integer or null), last_name (string or null), color (string or null),"
    "confidence (0..1), raw_text (string).\n"
    "Rules:\n"
    "- last_name should be uppercase.\n"
    "- If unsure, set field to null and lower confidence.\n"
    "- Ignore sponsors/team name unless it is clearly the player's last name.\n"
)

output =model.extract_jersey_information(
    image_path="1125-8.jpg",
    prompt=prompt
)

print(output)
