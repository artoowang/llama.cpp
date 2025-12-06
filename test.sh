#!/bin/bash

QWEN3_8B_GGUF=~ollama/.ollama/models/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f

PROMPT=$(cat <<EOF
<|im_start|>system

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{'type': 'function', 'function': {
    'description': 'Starts a new timer. If user does not specify a timer name, use the length of the timer as the name (e.g., "1 minute timer").',
    'name': 'HassStartTimer',
    'parameters': {
        'properties': {
            'hours': {'type': 'integer'},
            'minutes': {'type': 'integer'},
            'name': {'type': 'string'},
            'seconds': {'type': 'integer'}
        },
        'required': [],
    },
},
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{'name': <function-name>, 'arguments': <args-json-object>}
</tool_call>
<|im_end|>
<|im_start|>user
Timer, one minute.
<|im_end|>
<|im_start|>assistant
EOF
)

# Runs model without using conversation mode. I think this is equivalent to asking the model to
# complete the prompt?
./build/bin/llama-cli -m "$QWEN3_8B_GGUF" -no-cnv --verbose-prompt -p "$PROMPT"
