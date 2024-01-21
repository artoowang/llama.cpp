from guidance import (
    # Modules.
    models,
    # Libraries.
    capture,
    gen,
    zero_or_more,
    # Grammars.
    token_limit,
    select,
    # Roles.
    assistant,
    user,
)

import guidance
import utils

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

_logger = utils.get_logger()


@guidance(stateless=True)
def entity_id(lm: models.Model) -> models.Model:
    lm += select(ENTITIES)
    return lm


@guidance(stateless=True)
def entity_id_list_stateless(lm: models.Model) -> models.Model:
    lm += token_limit(zero_or_more(capture(entity_id(), "__LIST_APPEND:entity_id_list") + ", "), 100)
    return lm


@guidance
def entity_id_list(lm: models.Model) -> models.Model:
    MAX_NUM_ENTITIES = 10
    lm = lm.remove("entity_id_list")
    num_generated = 0
    while True:
        lm += select([capture(entity_id(), "__LIST_APPEND:entity_id_list") + select([", ", ""]), ""])
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
