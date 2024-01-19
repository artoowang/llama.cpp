#!/usr/bin/env python
from guidance import gen, models, select, user, assistant, system

import colorlog
import guidance
import logging
import pdb

PATH_TO_MODEL = "../models/Mistral-7B-Instruct-v0.2/mistral-7b-instruct-v0.2-q4_k.gguf"

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_log_handler = colorlog.StreamHandler()
_log_handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s:%(name)s:%(reset)s %(message)s"
))
_logger.addHandler(_log_handler)


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
        if role_name == "user":
            assert self._current_msg_id % 2 == 0
        elif role_name == "assistant":
            assert self._current_msg_id % 2 == 1
        else:
            assert False, f"Unrecognized role: {role_name}"


@guidance
def generate_entities(lm):
    while True:
        # Generate a double quoted entitit ID, and store the quoted string in a
        # list.
        lm += ('"' +
               gen(name="entities", list_append=True, max_tokens=10, stop='"') +
               '"')
        # Generate the delimiter: if it is the end of the list "]", then stop.
        lm += gen(max_tokens=10, stop=[",", "]"], save_stop_text="stop")
        if lm["stop"] == "]":
            break
        # Otherwise, add the delimiter and proceed to the next entity.
        lm += ", "
    return lm


model = MistralChat(
    PATH_TO_MODEL, n_ctx=4096, verbose=True, n_gpu_layers=-1)

with user():
    system_prompt = """You are a smart home agent named Jarvis, powered by
Home Assistant.

Following are the entities in this smart home:

Name,ID,State
"Date & Time","sensor.date_time","2024-01-14, 00:58"
"Random Number","sensor.random_number","6"
"TV","switch.tv","off"
"Mos Eisley","switch.mos_eisley","on"
"Kitchen Temperature","sensor.kitchen_temperature","56.1"
"Kitchen Humidity","sensor.kitchen_humidity","61.0"
"Kitchen PM 2.5","sensor.kitchen_pm25","0"
"Kitchen CO2","sensor.kitchen_co2","408"
"Energy Usage","sensor.utility_kwh","143505.062"
"Power Usage","sensor.utility_watts","287"
"Hallway Occupancy","binary_sensor.hallway_occupancy","off"
"Bedroom Sensor Occupancy","binary_sensor.bedroom_occupancy","off"
"Hallway Temperature","sensor.hallway_temperature","55.4"
"Hallway Humidity","sensor.hallway_humidity","63"
"Hallway Air quality index","sensor.hallway_air_quality_index","46"
"Hallway VOCs","sensor.hallway_voc","692"
"Hallway Carbon dioxide","sensor.hallway_co2","586"
"Bedroom Temperature","sensor.bedroom_temperature","55.3"
"Espresso Machine","switch.aukey_espresso","unknown"
"Air Purifier","switch.aukey_air_purifier","unknown"
"Entry Light","switch.entry_light","off" 

Here is the valid action List:
turn_off, turn_on, toggle

Respond to the user messages with the following JSON template: 
{"action": "", "entities": [], "response": ""}
and update the JSON fields by:
* Setting action to one of the Action List item, or empty if none of the actions apply.
* Setting entities to the entities IDs (not names) related to the user message.
* Setting response to a sentence responding to the user's message.

"""
    model += system_prompt
    _logger.info(f"System prompt:\n{system_prompt}")

    # model += "What is the current temperature?"
    model += "What is the current temperature in kitchen and bedroom?"
    # model += "Turn on all switches."

with assistant():
    model += f"""
{{
    "action": "{select(["none", "turn_on", "turn_off", "toggle"], name="action")}",
    "entities": [{generate_entities()}],
    "response": "{gen(name="response", max_tokens=256, stop='"')}"
}}
"""

_logger.debug(f"Model state:\n{model}")

action = model["action"]
entities = model["entities"]
response = model["response"]

_logger.info(f"""Result:
action: {action}
entities: {entities}
response: {response}
""")
