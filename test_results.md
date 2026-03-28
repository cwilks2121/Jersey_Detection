194 images
469 ground truth players

vlm 8b:
    f1 score: 80.71%
    error rate: 19.30%
    correct detections: 405/469 (86.35%)
    false alarms: 166
    average time per image: 23.04 s

vlm 4b:
    f1 score: 80.71%
    error rate: 21.45%
    correct detections: 400/469 (85.29%) 
    false alarms: 188
    average time per image: 9.48 s

yolo + vlm 8b:
    f1 score: 79.01%
    error rate: 22.16%
    correct detections: 389/469 (82.94%)
    false alarms: 185
    average time per image: 20.23 s

yolo + vlm 4b:
    f1 score: 76.43%
    error rate: 21.40%
    correct detections: 374/469 (79.74%)
    false alarms: 158
    average time per image: 8.83 s 

yolo + ocr + vlm 8b:
    f1 score: 77.61%
    error rate: 19.19%
    correct detections: 367/469 (78.25%) 
    false alarms: 136
    average time per image: 19.09 s

yolo + ocr + vlm 4b:
    f1 score: 76.94%
    error rate: 18.56%
    correct detections: 365/469 (77.83%) 
    false alarms: 137
    average time per image: 9.52 s

sam3 + vlm 8b:
    f1 score: 79.20%
    error rate: 21.21%
    correct detections: 394/469 (84.01%)
    false alarms: 183
    average time per image: 20.46 s

sam3 + vlm 4b:
    f1 score: 81.63%
    error rate: 19.35%
    correct detections: 404/469 (86.14%) 
    false alarms: 148
    average time per image: 10.25 s 

sam3 + ocr + vlm 8b:
    f1 score: 74.19%
    error rate: 18.26%
    correct detections: 349/496 (74.41%) 
    false alarms: 135
    average time per image: 19.06 s

sam3 + ocr + vlm 4b:
    f1 score: 73.05%
    error rate: 15.46%
    correct detections: 343/469 (73.13%) 
    false alarms: 108
    average time per image: 8.95 s

