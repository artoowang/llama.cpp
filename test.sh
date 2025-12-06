#!/bin/bash

QWEN3_8B_GGUF=~ollama/.ollama/models/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f

PROMPT="""
<|im_start|>system
Speak like Yoda, in reverse.
<|im_end|>
<|im_start|>user
Hello
<|im_end|>
<|im_start|>assistant
"""

# Runs model without using conversation mode. I think this is equivalent to asking the model to
# complete the prompt?
./build/bin/llama-cli -m "$QWEN3_8B_GGUF" -no-cnv --verbose-prompt -p "$PROMPT"
