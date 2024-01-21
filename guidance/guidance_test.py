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
from homeassistant import (
    entity_id_list,
)
from mistral import (
    PATH_7B_INSTRUCT_0_2_Q4_K,
    MistralChat
)

import guidance
import pdb
import utils

_logger = utils.get_logger()
_model = MistralChat(
    PATH_7B_INSTRUCT_0_2_Q4_K, n_ctx=4096, verbose=True, n_gpu_layers=-1)


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
Entities: switch.entry_light
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
