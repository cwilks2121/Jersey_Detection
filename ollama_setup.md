1. Download Ollama from https://ollama.com/download/mac
2. Find a model from https://ollama.com/search
3. Run 'ollama pull <model name>' in the terminal
4. Run 'ollama list' in the terminal to see your downloaded models
5. Run 'ollama serve' in the terminal to start your ollama local server
5. Pass in that model name and your local ollama url into the OllamaModel class

./build/bin/llama-server \
  -m ./models/qwen3-vl-bf16/Qwen3-VL-8B-Instruct-BF16.gguf \
  --mmproj ./models/qwen3-vl-bf16/mmproj-BF16.gguf \
  -c 16384 \
  -ngl 999 \
  -fa auto \
  --host 0.0.0.0 \
  --port 8080

./build/bin/llama-server \
  -m ./models/gemma3-27b/gemma-3-27b-it-Q4_K_M.gguf \
  --mmproj ./models/gemma3-27b/mmproj-BF16.gguf \
  -c 16384 \
  -ngl 999 \
  --flash-attn auto \
  --port 8080

  ngrok http 8080