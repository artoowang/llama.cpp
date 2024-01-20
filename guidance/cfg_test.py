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

import guidance
import logging

logging.basicConfig(level=logging.DEBUG)

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

_model = models.LlamaCpp(
    PATH_TO_MODEL, n_ctx=4096, verbose=True, n_gpu_layers=-1)


# @guidance(stateless=True)
# def entity_domain(lm: models.Model) -> models.Model:
#     lm += one_or_more(char_set("A-Za-z0-9_"))
#     return lm


# @guidance(stateless=True)
# def entity_name(lm: models.Model) -> models.Model:
#     lm += one_or_more(char_set("A-Za-z0-9_"))
#     return lm


@guidance(stateless=True)
def entity_id(lm: models.Model) -> models.Model:
    lm += select(ENTITIES)
    return lm


@guidance(stateless=True)
def entity_id_list(lm: models.Model) -> models.Model:
    lm += token_limit(zero_or_more(capture(entity_id(), "__LIST_APPEND:entity_id_list") + ", "), 100)
    return lm


_model += f"""
Following are the latest entity states in a smart home:

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

The ID of Espresso Machine is switch.aukey_espresso
"""

_model += f"The ID of the light is {entity_id()}\n"
_model += "The IDs of all temperatures are sensor.kitchen_temperature, sensor.hallway_temperature, sensor.bedroom_temperature, \n"
_model += f"The IDs of all switches are {entity_id_list()}\n"

print(_model)
print(f"""
entity_id_list: {_model["entity_id_list"]}
""")
