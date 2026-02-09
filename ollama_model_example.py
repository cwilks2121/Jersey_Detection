from ollama_model import OllamaModel
from dataframe_creation import DataFrameCreator
from compute_statistics import compute_accuracy_and_hallucination
from pathlib import Path

qwen_model = OllamaModel(model_name="qwen3-vl")
llama_model = OllamaModel(model_name="llama3.2-vision:11b")
llava_model = OllamaModel(model_name="llava:13b")
qwen2_model = OllamaModel(model_name="qwen2.5vl:7b")

prompt = (
    "You are doing jersey OCR. Extract ONLY the player's LAST NAME, JERSEY NUMBER, and JERSEY COLOR.\n"
    "Return JSON with keys: number (integer list or null), last_name (string list or null), color (string list or null),"
    "confidence (0..1), raw_text (string). There can be multiple players in the image. Therefore, you must return a list "
    "of values for each JSON key even if only one player is detected.\n"
    "Rules:\n"
    "- last_name should be uppercase.\n"
    "- If unsure, set field to null and lower confidence.\n"
    "- Ignore sponsors/team name unless it is clearly the player's last name.\n"
)

image_folder = Path("images/")
image_files = [str(f) for f in image_folder.iterdir() if f.is_file()]

qwen_df_creator = DataFrameCreator()
llama_df_creator = DataFrameCreator()
llava_df_creator = DataFrameCreator()
qwen2_df_creator = DataFrameCreator()

for img_path in image_files:
    qwen_output = qwen_model.extract_jersey_information(
        image_path=img_path,
        prompt=prompt
    )
    qwen_df_creator.append_df_from_output(qwen_output, img_path=img_path)

    llama_output = llama_model.extract_jersey_information(
        image_path=img_path,
        prompt=prompt
    )
    llama_df_creator.append_df_from_output(llama_output, img_path=img_path)

    llava_output = llava_model.extract_jersey_information(
        image_path=img_path,
        prompt=prompt
    )
    llava_df_creator.append_df_from_output(llava_output, img_path=img_path)

    qwen2_output = qwen2_model.extract_jersey_information(
        image_path=img_path,
        prompt=prompt
    )
    qwen2_df_creator.append_df_from_output(qwen2_output, img_path=img_path)


qwen_df = qwen_df_creator.get_raw_df()
qwen_accuracy, qwen_hallucination_rate = compute_accuracy_and_hallucination(qwen_df)
print(qwen_df)
print(f"Accuracy: {qwen_accuracy}")
print(f"Hallucination Rate: {qwen_hallucination_rate}")

llama_df = llama_df_creator.get_raw_df()
llama_accuracy, llama_hallucination_rate = compute_accuracy_and_hallucination(llama_df)
print(llama_df)
print(f"Accuracy: {llama_accuracy}")
print(f"Hallucination Rate: {llama_hallucination_rate}")

llava_df = llava_df_creator.get_raw_df()
llava_accuracy, llava_hallucination_rate = compute_accuracy_and_hallucination(llava_df)
print(llava_df)
print(f"Accuracy: {llava_accuracy}")
print(f"Hallucination Rate: {llava_hallucination_rate}")

qwen2_df = qwen2_df_creator.get_raw_df()
qwen2_accuracy, qwen2_hallucination_rate = compute_accuracy_and_hallucination(qwen2_df)
print(qwen2_df)
print(f"Accuracy: {qwen2_accuracy}")
print(f"Hallucination Rate: {qwen2_hallucination_rate}")
