#!/usr/bin/env python
from llama_cpp import Llama

PATH_TO_MODEL = "../models/Mistral-7B-Instruct-v0.2/mistral-7b-instruct-v0.2-q4_k.gguf"

# Change n_gpu_layers to 0 for CPU only run.
llm = Llama(model_path=PATH_TO_MODEL, verbose=True, n_gpu_layers=-1)
output = llm(
    "Q: Name the planets in the solar system? A: ",
    max_tokens=1024,
    stop=["Q:"],
    echo=True
)
print(output)
