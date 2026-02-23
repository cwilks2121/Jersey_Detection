from ollama_model import OllamaModel
from dataframe_creation import DataFrameCreator
from compute_statistics import compute_f1_and_hallucination
from pathlib import Path
from yolo_detection import detect_players_and_annotate
import time
import pandas as pd
from IPython.display import display, HTML
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

model_name = "gemma3:27b"
gemma_model = OllamaModel(model_name=model_name)

with open('system_prompt.txt', 'r') as file:
    system_prompt = str(file.read())

image_folder = Path("images/")
image_files = [str(f) for f in image_folder.iterdir() if f.is_file()]

gemma_df_creator = DataFrameCreator()
start_time = time.time()

for img_path in image_files:
    gemma_bboxed_path = detect_players_and_annotate(img_path)
    gemma_output = gemma_model.extract_jersey_information(
        image_path=gemma_bboxed_path,
        system_prompt=system_prompt
    )
    gemma_df_creator.append_df_from_output(gemma_output, img_path=img_path)
    print(f"Processed {img_path}")

end_time = time.time()
elapsed_time = end_time - start_time

gemma_df = gemma_df_creator.get_raw_df()
gemma_f1, gemma_hallucination_rate = compute_f1_and_hallucination(gemma_df)

num_images = len(image_files)

total_correct = gemma_df["correctly_identified_numbers"].apply(len).sum()
total_false_alarms = gemma_df["false_positives"].apply(len).sum()
total_ground_truth_players = gemma_df["number_ground_truth"].apply(len).sum()

print(gemma_df)
print("Model used for analysis:", model_name)
print("Total time for report to run:", f"{elapsed_time / 60:.2f} minutes")
print("Average time per image:", f"{elapsed_time / num_images:.2f} seconds")
print(f"F1 Score: {gemma_f1}")
print(f"Hallucination Rate: {gemma_hallucination_rate}")
print(f"Correct Detections: {total_correct} / {total_ground_truth_players}")
print(f"False Alarms: {total_false_alarms}")

prompt_html = (
    "<pre style='white-space:pre-wrap; margin:0; "
    "font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; "
    "font-size: 11px; line-height: 1.25;'>"
    + (system_prompt.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
    + "</pre>"
)

summary_rows = [
    ("Model used for analysis", model_name),
    ("Total time for report to run", f"{elapsed_time / 60:.2f} min"),
    ("Average time per image", f"{elapsed_time / num_images:.2f} s"),
    ("F1 Score", f"{gemma_f1 * 100:.2f} %"),
    ("Hallucination Rate", f"{gemma_hallucination_rate * 100:.2f} %"),
    ("Correct detections", f"{total_correct} / {total_ground_truth_players} ({100 * total_correct / total_ground_truth_players:.2f} %)"),
    ("False alarms", str(total_false_alarms)),
    ("Prompt", prompt_html),
]

# Build the HTML for the run summary table at the top of the report
summary_html = """
<style>
  .report-wrap { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }

  .summary-card { margin: 0 0 18px 0; padding: 14px 16px; border: 1px solid #ddd; border-radius: 10px; }
  .summary-title { font-size: 18px; font-weight: 700; margin: 0 0 10px 0; }
  table.summary { border-collapse: collapse; width: 100%; }
  table.summary td { padding: 8px 10px; border-top: 1px solid #eee; vertical-align: top; }
  table.summary td.k { width: 28%; font-weight: 600; color: #222; text-align: left; }
  table.summary td.v { color: #111; text-align: left; }

  .note { font-size: 12px; color: #666; margin-top: 10px; }

  /* Center images inside cells in main table */
  table.dataframe img { display: block; margin: 0 auto; }
</style>
<div class="report-wrap">
  <div class="summary-card">
    <div class="summary-title">Run Summary</div>
    <table class="summary">
"""

# Add each (key, value) row into the summary table
for k, v in summary_rows:
    summary_html += f'<tr><td class="k">{k}</td><td class="v">{v}</td></tr>\n'

google_sheets_light_green = "#d9ead3"
google_sheets_light_yellow = "#fff2cc"
google_sheets_light_red   = "#f4cccc"

def _row_fill(row):
    """
      Pick a background color for an entire row based on accuracy_percent.
      - No accuracy_percent -> no color
      - 1.0                 -> green
      - Between 0 and 1     -> yellow
      - 0 or negative       -> red
    """
    # No accuracy_percent -> no color
    val = row.get("f1_score", np.nan)
    if val is None:
      val = np.nan
    row_accuracy_percent = float(val)
    if row_accuracy_percent is None or np.isnan(row_accuracy_percent):
        return [""] * len(row)

    if np.isclose(row_accuracy_percent, 1.0):
        color = google_sheets_light_green
    elif 0.0 < row_accuracy_percent < 1.0:
        color = google_sheets_light_yellow
    else:
        color = google_sheets_light_red

    return [f"background-color: {color};"] * len(row)

def img_tag(path, width=220):
    if not path or pd.isna(path):
        return ""
    # Make it a web-friendly relative path
    p = str(path).replace("\\", "/")
    return f'<img src="{p}" style="max-width:{width}px; height:auto; border-radius:6px;" />'

df_styler = (
    gemma_df.style
      .apply(_row_fill, axis=1)
      .format({"boxed_image": lambda p: img_tag(p, width=260),
               "image": lambda p: img_tag(p, width=260)})
      .set_properties(**{"text-align": "center", "vertical-align": "middle"})
      .set_table_styles([
          {"selector": "th", "props": [("text-align", "center"), ("vertical-align", "middle")]},
          {"selector": "td", "props": [("text-align", "center"), ("vertical-align", "middle")]},
          {"selector": "table", "props": [("width", "100%"), ("border-collapse", "collapse")]},
      ])
      .hide(axis="index")
)

# Turn the styled dataframe into HTML (escape=False so <img> tags still render)
df_html = df_styler.to_html(escape=False)

with open("summary.html", "w") as f:
    f.write(summary_html + df_html + "</div>")
