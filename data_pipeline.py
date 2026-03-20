from ollama_model import OllamaModel
from dataframe_creation import DataFrameCreator
from compute_statistics import compute_f1_and_hallucination
from pathlib import Path
import yolo_detection
import time
import pandas as pd
from html_summary import generate_html_summary
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from ultralytics import YOLO
import easyocr

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

model_name = "gemma3:27b"
model = OllamaModel(model_name=model_name)

with open('system_prompt.txt', 'r') as file:
    system_prompt = str(file.read())

image_folder = Path("images/")
image_files = [str(f) for f in image_folder.iterdir() if f.is_file()]

yolo_model = YOLO("yolo26n.pt")
ocr_model = easyocr.Reader(["en"], gpu=True)

def _process_image(img_path: str) -> tuple[str, dict]:
    predicted_digits = yolo_detection.extract_digits_from_boxes(
        image_path=img_path,
        yolo_model=yolo_model,
        ocr_model=ocr_model
    )
    bboxed_path = yolo_detection.detect_players_and_annotate(
        image_path=img_path,
        yolo_model=yolo_model,
        conf_threshold=0.3
    )
    model_output = model.extract_jersey_information(
        image_path=bboxed_path,
        system_prompt=system_prompt,
        prompt=f"Follow the system prompt. The detected digits from an OCR are provided in the following list: {predicted_digits}. " \
                "These are only to be used as a guide for jersey numbers, not a final decision."
    )

    return img_path, model_output

df_creator = DataFrameCreator()
start_time = time.time()

max_workers = int(os.getenv("PIPELINE_WORKERS", "1"))
if max_workers <= 1:
    for img_path in image_files:
        _, model_output = _process_image(img_path)
        df_creator.append_df_from_output(model_output, img_path=img_path)
        print(f"Processed {img_path}")
else:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_image, img_path): img_path for img_path in image_files}
        for future in as_completed(futures):
            img_path = futures[future]
            _, model_output = future.result()
            df_creator.append_df_from_output(model_output, img_path=img_path)
            print(f"Processed {img_path}")

end_time = time.time()
elapsed_time = end_time - start_time

model_df = df_creator.get_raw_df()
model_f1, model_hallucination_rate = compute_f1_and_hallucination(model_df)

num_images = len(image_files)

total_correct = model_df["correctly_identified_numbers"].apply(len).sum()
total_false_alarms = model_df["false_positives"].apply(len).sum()
total_ground_truth_players = model_df["number_ground_truth"].apply(len).sum()

print(model_df)
print("Model used for analysis:", model_name)
print("Total time for report to run:", f"{elapsed_time / 60:.2f} minutes")
print("Average time per image:", f"{elapsed_time / num_images:.2f} seconds")
print(f"F1 Score: {model_f1}")
print(f"Hallucination Rate: {model_hallucination_rate}")
print(f"Correct Detections: {total_correct} / {total_ground_truth_players}")
print(f"False Alarms: {total_false_alarms}")

generate_html_summary(
    system_prompt=system_prompt,
    model_name=model_name,
    elapsed_time=elapsed_time,
    num_images=num_images,
    model_f1=model_f1,
    model_hallucination_rate=model_hallucination_rate,
    total_correct=total_correct,
    total_ground_truth_players=total_ground_truth_players,
    total_false_alarms=total_false_alarms,
    model_df=model_df
)
