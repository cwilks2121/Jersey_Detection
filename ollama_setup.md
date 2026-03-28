1. Download Ollama from https://ollama.com/download/mac
2. Find a model from https://ollama.com/search
3. Run 'ollama pull <model name>' in the terminal
4. Run 'ollama list' in the terminal to see your downloaded models
5. Run 'ollama serve' in the terminal to start your ollama local server
5. Pass in that model name and your local ollama url into the OllamaModel class

./build/bin/llama-server \
  -m ./models/qwen3-vl-8b/Qwen3-VL-8B-Instruct-BF16.gguf \
  --mmproj ./models/qwen3-vl-8b/mmproj-BF16.gguf \
  -c 16834 \
  -ngl 999 \
  -fa auto \
  --host 0.0.0.0 \
  --port 8081

./build/bin/llama-server \
  -m ./models/qwen3-vl-4b/Qwen3-VL-4B-Instruct-Q4_K_M.gguf \
  --mmproj ./models/qwen3-vl-4b/mmproj-F16.gguf \
  -c 16834 \
  -ngl 999 \
  -fa auto \
  --host 0.0.0.0 \
  --port 8081

./build/bin/llama-server \
  -m ./models/qwen3-vl-2b/Qwen3-VL-2B-Instruct-Q4_K_M.gguf \
  --mmproj ./models/qwen3-vl-2b/mmproj-F16.gguf \
  -c 16834 \
  -ngl 999 \
  -fa auto \
  --host 0.0.0.0 \
  --port 8081

ngrok http 8081