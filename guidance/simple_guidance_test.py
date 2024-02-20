#!/usr/bin/env python

import guidance
import logging
import pdb

from mistral import (
    PATH_7B_INSTRUCT_0_2_Q4_K,
    MistralChat,
)
from guidance import (
    gen,
    select,
)

logging.basicConfig()
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.DEBUG)


def run_model(model: MistralChat) -> str | None:
    out_model = model + "One plus one equals " + \
        gen(max_tokens=100, name="result")
    return out_model["result"]


model = MistralChat(PATH_7B_INSTRUCT_0_2_Q4_K, n_ctx=4096,
                    verbose=True, n_gpu_layers=-1)
_LOGGER.info(f"MistralChat model: {model}")
pdb.set_trace()
# _LOGGER.info(run_model(model))
