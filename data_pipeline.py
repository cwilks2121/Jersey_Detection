from ollama_model import OllamaModel
from dataframe_creation import DataFrameCreator
from compute_statistics import compute_f1_and_hallucination
from pathlib import Path
from yolo_detection import detect_players_and_annotate
import time

gemma_model = OllamaModel(model_name="gemma3:12b")

with open('system_prompt.txt', 'r') as file:
    prompt = str(file.read())

image_folder = Path("images/")
image_files = [str(f) for f in image_folder.iterdir() if f.is_file()]
# image_files = ["images/_MG_3492-4-8-42-2-26-15.jpg", "images/_MG_3656-4-29-12.jpg"]

gemma_df_creator = DataFrameCreator()
start_time = time.time()

for img_path in image_files:
    gemma_bboxed_path = detect_players_and_annotate(img_path)
    gemma_output = gemma_model.extract_jersey_information(
        image_path=gemma_bboxed_path,
        prompt="Follow the system prompt to extract jersey information from the image.",
        system_prompt=prompt
    )
    gemma_df_creator.append_df_from_output(gemma_output, img_path=img_path)
    print(f"Processed {img_path}")

end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds")

gemma_df = gemma_df_creator.get_raw_df()
gemma_f1, gemma_hallucination_rate = compute_f1_and_hallucination(gemma_df)
print(gemma_df)
print(f"F1 Score: {gemma_f1}")
print(f"Hallucination Rate: {gemma_hallucination_rate}")

