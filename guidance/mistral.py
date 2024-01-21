from guidance import models

PATH_7B_INSTRUCT_0_2_Q4_K = "../models/Mistral-7B-Instruct-v0.2/mistral-7b-instruct-v0.2-q4_k.gguf"


class MistralChat(models.LlamaCpp, models.Chat):
    """
    A customized guidance.models.LlamaCppChat tailored for Mistral style chat
    tokens. See https://docs.mistral.ai/models/. The style here also matches
    what mistral-7b-instruct-v0.2-q4_k.gguf reports in its
    tokenizer.chat_template.
    """

    _current_msg_id: int | None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_msg_id = None

    def get_role_start(self, role_name, **kwargs):
        if self._current_msg_id is None:
            self._current_msg_id = 0
        else:
            self._current_msg_id += 1
        self._check_current_msg_id(role_name)

        if role_name == "user":
            if self._current_msg_id == 0:
                return "<s>[INST] "
            else:
                return "[INST] "
        elif role_name == "assistant":
            return ""

    def get_role_end(self, role_name=None):
        self._check_current_msg_id(role_name)
        if role_name == "user":
            return " [/INST]"
        elif role_name == "assistant":
            return "</s>"

    def _check_current_msg_id(self, role_name):
        assert self._current_msg_id is not None
        if role_name == "user":
            assert self._current_msg_id % 2 == 0
        elif role_name == "assistant":
            assert self._current_msg_id % 2 == 1
        else:
            assert False, f"Unrecognized role: {role_name}"
