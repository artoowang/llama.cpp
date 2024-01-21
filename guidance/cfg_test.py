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
    entity_id,
    entity_id_list,
    entity_id_list_stateless,
)
from mistral import (
    PATH_7B_INSTRUCT_0_2_Q4_K,
)

import guidance
import utils

_logger = utils.get_logger()
_model = models.LlamaCpp(
    PATH_7B_INSTRUCT_0_2_Q4_K, n_ctx=4096, verbose=True, n_gpu_layers=-1)


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

# _model += f"The ID of the light is {entity_id()}\n"
_model += "The IDs of all temperatures are sensor.kitchen_temperature, sensor.hallway_temperature, sensor.bedroom_temperature, \n"
# _model += f"Repeat switch.tv 5 times: switch.tv, {entity_id_list()}\n"
_model += f"The IDs of all switches are {entity_id_list()}\n"
_model += f"The ID of the light is {entity_id_list()}\n"

_logger.info(_model)
_logger.info(f"""
entity_id_list: {_model["entity_id_list"]}
""")
