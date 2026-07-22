import json, sys, os
from http.server import HTTPServer, BaseHTTPRequestHandler
from llama_cpp import Llama

MODEL_PATH = os.environ.get("MODEL_PATH", "/caixiaoyao/ollama_models/qwen2.5-14b/qwen2.5-14b-instruct-q6_k.gguf")
PORT = int(os.environ.get("PORT", "11434"))

print(f"Loading model: {MODEL_PATH}")
llm = Llama(model_path=MODEL_PATH, n_gpu_layers=48, verbose=False, n_ctx=16384)
print("Model loaded")

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data":[{"id":"qwen2.5:14b"}]}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            messages = body.get("messages", [])
            prompt = ""
            for m in messages:
                role = m["role"]
                content = m["content"]
                if role == "system":
                    prompt = f"<|im_start|>system\n{content}\n<|im_end|>\n"
                elif role == "user":
                    prompt += f"<|im_start|>user\n{content}\n<|im_end|>\n"
                elif role == "assistant":
                    prompt += f"<|im_start|>assistant\n{content}\n<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

            try:
                result = llm(prompt, max_tokens=512, temperature=0.7, stop=["<|im_end|>", "<|im_start|>"])
                text = result["choices"][0]["text"].strip()
                resp = {
                    "choices": [{
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop"
                    }]
                }
            except Exception as e:
                resp = {"error": str(e), "choices": [{"message": {"role": "assistant", "content": f"Error: {e}"}, "finish_reason": "stop"}]}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())
        else:
            self.send_response(404)
            self.end_headers()

HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
