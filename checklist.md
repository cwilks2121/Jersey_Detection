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
