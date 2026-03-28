Tasks
- Print out number of tokens and time used per image
DONE - Explore what information we can extract from each image (last name, number, color, etc.)
DONE - Print out accuracy and hallucination rate
DONE - Allow for printing of multiple jersey classifications in one image
- Work out a demo for kimi-k2.5, gemma3, and qwen3-vl
- Create a pipeline using Yolo and one of the models
DONE - Change system prompt and prompt to be more specific
- Create a graphic comparing kimi-k2.5, gemma3, and qwen3-vl performance
DONE - Extract ground truth from image names
- Find best ways to represent and pass in our data through the pipeline


Stuff to do
- Use an ollama model in conjuction with YOLO to downscale images, find individuals in the image, and achieve higher accuracy.
- Do prompt engineering to find the best prompt.
- Determine a good way to take the returned pandas dataframe and save and interpret the results.
- Explore cheap cloud models that 40NorthLabs may potentially be interested in.
- Explore model parameters like payload and options.
- Design a single script that can test multiple models at once.
- Output tokens and other useful informations into our output dataframe.
- Investigate how we can use OCRs in our pipeline. Particularly for candidate characters and then have the VLM determine what the best candidate is.

Week 4
- Try and get qwen3-vl to work using Ollama.
- Match html output that Scott has for his model.
- Update the system architecture.

- Test with deep segmentation to get rid of clutter in the image.
- Add text recognition to the pipeline after the YOLO layer
- Play around with the confidence level on YOLO on detecting people
- Filter out labels on the YOLO model to only detect humans (done)
- Get qwen VL to start working at a reasonable speed
- See if I can use GPU resources instead of a for loop with Ollama

1. Build a number region dataset: https://universe.roboflow.com/yakovk/jersey-numbers-i1wn5/dataset/2
-- yolo detect train data=jersey-numbers.v2i.yolo26/data.yaml model=yolo26n.pt imgsz=960 epochs=80 batch=16

Remove the cropped region of jersey and just use paddle ocr on the yolo cropped region
Change the system prompt to check the predicted number above the bounding boxes created by paddleocr
Utilize GPU resources to run faster
Test confidence thresholds on both YOLO and paddleocr

Utilize llama.cpp to run qwen3-vl:8b
Try out the model Ken found on finding jersey digits and whole jerseys
Test out multi threading on llama.cpp

Download Deepseek model

salloc \
>   --partition=dlair-gpu-np \
>   --qos=cs6953-gpu-np \
>   --account=cs6953-gpu-np \
>   --gres=gpu:1 \
>   --mem=32G \
>   --cpus-per-task=8 \
>   --time=10:00:00