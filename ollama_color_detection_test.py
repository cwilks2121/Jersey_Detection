from ollama_color_detection import colorDetection
from pathlib import Path
from yolo_detection import detect_players_and_annotate
import time

prompt = (
    "You will be given a image featuring athletes with boxes around."
    "From each image you should return the unique jersey colors that the athletes are wearing."
    "Ignore the shorts color, undershirts, green bounding box color as well as the green grass color, just include the jersey top color/colors."
    "There should be only one JSON element per image, but there can be multiple hex code color/color elements."
    
    "IMPORTANT INSTRUCTIONS:"
    "1. Detect ALL jersey colors that are FULLY VISIBLE in the image.\n"
    "2. Get the most specific color for the hex code that you can.\n"
    "3. Make sure the Hex Code and the raw color are The Same.\n"

    "RESPONSE FORMAT: Return a valid json for each unique jersey found in the image with the following format. "
    
    "hex_code_color: The color in hex of each element of the jersey"
    "color: The written out color\n"
    
    "Put each jersey (Can have multiple colors) into one element of the json."
    
    "An example of this reponse would be:"
    "jerseys: { hex_code_color: [#0096FF, #FF0000], color: [Blue, Red]},"
    "{ hex_code_color: [#FFFFF, #FFA500], color: [White, Orange]}."
    "This is two different jerseys."
    "There can be an unlimited amount of unique jerseys."
    
    "It should not be like:"
    "jerseys: { hex_code_color: [#0096FF, #0096FF], color: [Blue, Blue]} since this is the same color listed twice"
)

llava_model = colorDetection(model_name="llava:13b")

with open('system_prompt.txt', 'r') as file:
    prompt = str(file.read())

image_folder = Path("images/")
image_files = [str(f) for f in image_folder.iterdir() if f.is_file()]

start_time = time.time()

for img_path in image_files:
    llava_bboxed_path = detect_players_and_annotate(img_path)
    print(llava_bboxed_path)
    llava_output = llava_model.extract_color_info(
        image_path=llava_bboxed_path,
        prompt = prompt
    )
    print(llava_output)