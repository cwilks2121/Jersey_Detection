from PIL import Image, ImageDraw
from ultralytics import YOLO  # Assuming YOLO is a class in a module named YOLO

# Load the model
model = YOLO("models/number_finder.pt")

# Get model results
model_results = model(source="captcha_images/captcha-1.jpg", conf=0.35, verbose=False) 
img = Image.open("captcha_images/captcha-1.jpg")
draw = ImageDraw.Draw(img)

# Sort results based on the x-coordinate
detected_objects = []
captcha_result = ""
boxes_data = []

for result in model_results:
    for box in result.boxes:
        box_data = {"xyxy": box.xyxy[0].tolist(),
                    "label": str(box.cls.item())}
        boxes_data.append(box_data)

sorted_boxes_data = sorted(boxes_data, key=lambda x: x['xyxy'][0])

# Remove similar boxes
unique_sorted_boxes_data = [sorted_boxes_data[0]]
for i in range(1, len(sorted_boxes_data)):
    current_box = sorted_boxes_data[i]
    prev_box = unique_sorted_boxes_data[-1]
    # Check if the current box's x-coordinate is significantly different from the previous box
    if abs(current_box['xyxy'][0] - prev_box['xyxy'][0]) > 1:  # Adjust threshold as needed
        unique_sorted_boxes_data.append(current_box)

for box in unique_sorted_boxes_data:
    label_name = box["label"]
    box_coords = box["xyxy"]
    detected_objects.append((box_coords))
    draw.rectangle(box_coords, outline="red")
    draw.text((box_coords[0], box_coords[1] - 10), label_name, fill="red")
    captcha_result += str(int(label_name[0]) - 2)
img.show()  # Display the image with bounding boxes and labels