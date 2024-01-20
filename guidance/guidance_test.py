#!/usr/bin/env python
from datetime import datetime
from guidance import (
    # Modules.
    models,
    # Libraries.
    capture,
    gen,
    # Grammars.
    select,
    # Roles.
    assistant,
    user,
)

import colorlog
import guidance
import logging
import pdb

PATH_TO_MODEL = "../models/Mistral-7B-Instruct-v0.2/mistral-7b-instruct-v0.2-q4_k.gguf"
DOMAINS = [
    "binary_sensor",
    "sensor",
    "switch",
]
ENTITIES = [
    "sensor.date_time",
    "sensor.random_number",
    "switch.tv",
    "switch.mos_eisley",
    "sensor.kitchen_temperature",
    "sensor.kitchen_humidity",
    "sensor.kitchen_pm25",
    "sensor.kitchen_co2",
    "sensor.utility_kwh",
    "sensor.utility_watts",
    "binary_sensor.hallway_occupancy",
    "binary_sensor.bedroom_occupancy",
    "sensor.hallway_temperature",
    "sensor.hallway_humidity",
    "sensor.hallway_air_quality_index",
    "sensor.hallway_voc",
    "sensor.hallway_co2",
    "sensor.bedroom_temperature",
    "switch.aukey_espresso",
    "switch.aukey_air_purifier",
    "switch.entry_light",
]

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

_log_handler = colorlog.StreamHandler()
_log_handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s:%(name)s:%(reset)s %(message)s"
))
_log_handler.setLevel(logging.INFO)
_logger.addHandler(_log_handler)

_file_handler = logging.FileHandler("guidance_test.log", mode="w")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s:%(name)s: %(message)s"
))
_logger.addHandler(_file_handler)


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


_model = MistralChat(
    PATH_TO_MODEL, n_ctx=4096, verbose=True, n_gpu_layers=-1)


@guidance(stateless=True)
def entity_id(lm: models.Model) -> models.Model:
    lm += select(ENTITIES)
    return lm


@guidance
def entity_id_list(lm: models.Model) -> models.Model:
    MAX_NUM_ENTITIES = 10
    lm = lm.remove("entity_id_list")
    num_generated = 0
    while True:
        lm += select([capture(entity_id(), "__LIST_APPEND:entity_id_list") + ", ", ""])
        entities = lm.get("entity_id_list", default=[])
        new_num_generated = len(entities)
        if new_num_generated == num_generated:
            # Model has stopped generating a new entity ID, and we are done.
            break
        num_generated = new_num_generated
        if num_generated >= MAX_NUM_ENTITIES:
            _logger.warning("We have reached the maximum number of entities "
                            f"for entity_id_list(): {MAX_NUM_ENTITIES}")
            break

    if lm["entity_id_list"] is None:
        # Nothing has been generated, so the list is None.
        # TODO: Currently we can't initialize with
        # lm = lm.set("entity_id_list", []) before the loop, since capture()
        # seems to have issue working with that.
        return lm.set("entity_id_list", [])

    # Detect and potentially remove duplicates
    entities = lm["entity_id_list"]
    entities_no_duplicates = list(set(entities))
    num_duplicates = len(entities) - len(entities_no_duplicates)
    if num_duplicates > 0:
        _logger.warning(f"{num_duplicates} duplicate(s) have been removed.")
        _logger.debug(f"entities: {entities}\nentities_no_duplicates: {entities_no_duplicates}")
        lm = lm.set("entity_id_list", entities_no_duplicates)

    return lm


@guidance(stateless=True)
def current_home_states(lm: models.Model) -> models.Model:
    update_prompt = """
Following are the latest entity states in this smart home:

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
"""
    return lm + update_prompt


@guidance
def assistant_response(lm: models.Model) -> models.Model:
    newline = "\n"
    lm += f"""
Action: {gen(stop=newline)}
Entities: {entity_id_list()}
Response: {gen(stop=newline)}
"""
    _logger.info(f"Model state:\n{lm}\n")
    return lm


# Send the initial prompt.
with user():
    _model += """
You are a smart home agent named Jarvis, powered by
Home Assistant.

"""
    _model += current_home_states()
    _model += f"""

Here are the valid actions:
turn_off, turn_on, toggle

Respond to the user's message with the following:
Action: the applicable action. If there is no action applicable, reply "none".
Entities: the related entity IDs (not the entity names).
Response: a sensentce responding to the user's message. 

For example:

User:
Turn on the entry light

Assistant:
Action: turn_on
Entities: switch.entry_light, 
Response: Sure, I have turned on the Entry Light.

"""

with assistant():
    _model += """
Action: none
Entities:
Response: Hi, I am Jarvis, your smart home assistant. How may I help you?
"""

# TODO: For now, let's just print the model state entirely.
# We might want to revisit this and just output useful text only.
_logger.info(f"Model state:\n{_model}\n")

# Start user loop.
while True:
    with user():
        user_prompt = input("User: ")
        if len(user_prompt) == 0:
            _logger.info("Exitting ...")
            break

        # TODO: Test updating the states when the user prompt starts with "UPDATE".
        # This adds a fake sensor.test called "Test sensor", with its value set to the current timestamp.
        if user_prompt.startswith("UPDATE"):
            _model += current_home_states()
            test_sensor_prompt = f'"Test sensor",sensor.test,"{datetime.now().timestamp()}"\n'
            _logger.info(f"Adding test sensor state: {test_sensor_prompt}")
            _model += test_sensor_prompt

        _model += user_prompt

    with assistant():
        _model += assistant_response()
