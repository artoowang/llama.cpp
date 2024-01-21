#!/usr/bin/env python
from guidance import (
    # Modules.
    models,
    # Libraries.
    capture,
    char_set,
    gen,
    one_or_more,
    zero_or_more,
    # Grammars.
    select,
    token_limit,
    # Roles.
    user, assistant
)
from homeassistant import (
    assistant_response,
    entity_id,
    entity_id_list,
    entity_id_list_stateless,
    service_or_none,
    user_assistant_examples,
)
from mistral import (
    PATH_7B_INSTRUCT_0_2_Q4_K,
    MistralChat,
)

import guidance
import utils

_logger = utils.get_logger()
_model = models.LlamaCpp(
    PATH_7B_INSTRUCT_0_2_Q4_K, n_ctx=4096, verbose=True, n_gpu_layers=-1)


_model += f"""
You are a smart home assistant.
Following are the latest entity states in the smart home:

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

Greet the user, and then answer the user's instructions.

Assistant: Hi, I am your smart home assistant. How may I help you?
""" + user_assistant_examples()

_model += "User: What are the IDs of all switches?"
_model += assistant_response()
_logger.info(f"""Model state:
{_model}
Response: {_model["response"]}
""")

_model += "User: What is the current temperature?"
_model += assistant_response()
_logger.info(f"""Model state:
{_model}
Response: {_model["response"]}
""")

_model += "User: Turn off all switches"
_model += assistant_response()
_logger.info(f"""Model state:
{_model}
Response: {_model["response"]}
""")
