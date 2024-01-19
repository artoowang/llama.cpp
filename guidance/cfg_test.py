#!/usr/bin/env python
from guidance import (
    # Modules.
    models,
    # Libraries.
    char_set,
    gen,
    one_or_more,
    zero_or_more,
    # Grammars.
    select,
    # Roles.
    user, assistant
)

import guidance
import logging

logging.basicConfig(level=logging.DEBUG)

PATH_TO_MODEL = "../models/Mistral-7B-Instruct-v0.2/mistral-7b-instruct-v0.2-q4_k.gguf"

_model = models.LlamaCpp(
    PATH_TO_MODEL, n_ctx=4096, verbose=True, n_gpu_layers=-1)


@guidance(stateless=True)
def entity_domain(lm: models.Model) -> models.Model:
    lm += one_or_more(char_set("A-Za-z0-9_"))
    return lm


@guidance(stateless=True)
def entity_name(lm: models.Model) -> models.Model:
    lm += one_or_more(char_set("A-Za-z0-9_"))
    return lm


@guidance(stateless=True)
def entity_id(lm: models.Model) -> models.Model:
    lm += entity_domain() + "." + entity_name()
    return lm


@guidance(stateless=True)
def entity_id_list(lm: models.Model) -> models.Model:
    lm += "[" + zero_or_more(entity_id() + select(["", ", "])) + "]"
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
The ID of the light is {entity_id()}
The IDs of all temperatures are [sensor.kitchen_temperature, sensor.hallway_temperature, sensor.bedroom_temperature]
The IDs of all switches are {entity_id_list()}
"""

print(_model)
