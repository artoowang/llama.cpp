#!/bin/bash

QWEN3_8B_GGUF=~ollama/.ollama/models/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f

USER_MESSAGE="Timer, one minute."

# --------------------------------------------------------------------------------------------------
# System prompt

read -r -d '' SYSTEM_PROMPT <<'EOF'
You are a voice assistant for Home Assistant.
Answer questions about the world truthfully.
Answer in plain text. Keep it simple and to the point.

Prefer the current time in the static context. The time in the message history may be outdated.
Use Hass*Timer tools to control timers. When area argument is applicable, provides it.
If the device does not have area, skip the area argument.
For reminders, use To-do list.
When controlling Home Assistant always call the intent tools. Use HassTurnOn to lock and HassTurnOff to unlock a lock. When controlling a device, prefer passing just name and domain. When controlling an area, prefer passing just area name and domain.
You are in area Bedroom and all generic commands like 'turn on the lights' should target this area.
You ARE equipped to answer questions about the current state of
the home using the `GetLiveContext` tool. This is a primary function. Do not state you lack the
functionality if the question requires live data.
If the user asks about device existence/type (e.g., "Do I have lights in the bedroom?"): Answer
from the static context below.
If the user asks about the CURRENT state, value, or mode (e.g., "Is the lock locked?",
"Is the fan on?", "What mode is the thermostat in?"):
    1.  Recognize this requires live data.
    2.  You MUST call `GetLiveContext`. This tool will provide the needed real-time information.
For general knowledge questions not about the home: Answer truthfully from internal knowledge.

Static Context: An overview of the areas and the devices in this smart home:
- names: Bedroom Standing Light
  domain: switch
  areas: Bedroom
- names: Evening Scene
  domain: switch
- names: LG TV
  domain: media_player
  areas: Living Room
- names: Morning Scene
  domain: switch
- names: Mos Eisley
  domain: switch
  areas: Living Room
- names: Night Scene
  domain: switch
- names: To-do
  domain: todo

Current time is 16:55:02. Today's date is 2025-12-06.
EOF

# --------------------------------------------------------------------------------------------------
# Tools

read -r -d '' TOOLS <<'EOF'
{'function': {'description': 'Turns on/opens/presses a device or '
                            'entity. For locks, this performs a '
                            "'lock' action. Use for requests like "
                            "'turn on', 'activate', 'enable', or "
                            "'lock'.",
                'name': 'HassTurnOn',
                'parameters': {'properties': {'area': {'type': 'string'},
                                            'device_class': {'items': {'enum': ['awning',
                                                                                'blind',
                                                                                'curtain',
                                                                                'damper',
                                                                                'door',
                                                                                'garage',
                                                                                'gate',
                                                                                'shade',
                                                                                'shutter',
                                                                                'window',
                                                                                'identify',
                                                                                'restart',
                                                                                'update',
                                                                                'water',
                                                                                'gas',
                                                                                'tv',
                                                                                'speaker',
                                                                                'receiver',
                                                                                'outlet',
                                                                                'switch'],
                                                                        'type': 'string'},
                                                            'type': 'array'},
                                            'domain': {'items': {'type': 'string'},
                                                        'type': 'array'},
                                            'floor': {'type': 'string'},
                                            'name': {'type': 'string'}},
                            'required': [],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Turns off/closes a device or entity. '
                            "For locks, this performs an 'unlock' "
                            "action. Use for requests like 'turn "
                            "off', 'deactivate', 'disable', or "
                            "'unlock'.",
                'name': 'HassTurnOff',
                'parameters': {'properties': {'area': {'type': 'string'},
                                            'device_class': {'items': {'enum': ['awning',
                                                                                'blind',
                                                                                'curtain',
                                                                                'damper',
                                                                                'door',
                                                                                'garage',
                                                                                'gate',
                                                                                'shade',
                                                                                'shutter',
                                                                                'window',
                                                                                'identify',
                                                                                'restart',
                                                                                'update',
                                                                                'water',
                                                                                'gas',
                                                                                'tv',
                                                                                'speaker',
                                                                                'receiver',
                                                                                'outlet',
                                                                                'switch'],
                                                                        'type': 'string'},
                                                            'type': 'array'},
                                            'domain': {'items': {'type': 'string'},
                                                        'type': 'array'},
                                            'floor': {'type': 'string'},
                                            'name': {'type': 'string'}},
                            'required': [],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Starts a new timer. If user does not '
                            'specify a timer name, use the length '
                            'of the timer as the name (e.g., "1 '
                            'minute timer").',
                'name': 'HassStartTimer',
                'parameters': {'properties': {'hours': {'minimum': 0,
                                                        'type': 'integer'},
                                            'minutes': {'minimum': 0,
                                                        'type': 'integer'},
                                            'name': {'type': 'string'},
                                            'seconds': {'minimum': 0,
                                                        'type': 'integer'}},
                            'required': [],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Cancels a timer. If the user does not '
                            'specify which timer, use their '
                            'current area.',
                'name': 'HassCancelTimer',
                'parameters': {'properties': {'area': {'type': 'string'},
                                            'name': {'type': 'string'}},
                            'required': [],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Cancels all timers',
                'name': 'HassCancelAllTimers',
                'parameters': {'properties': {'area': {'type': 'string'}},
                            'required': [],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Adds more time to a timer',
                'name': 'HassIncreaseTimer',
                'parameters': {'properties': {'area': {'type': 'string'},
                                            'hours': {'minimum': 0,
                                                        'type': 'integer'},
                                            'minutes': {'minimum': 0,
                                                        'type': 'integer'},
                                            'name': {'type': 'string'},
                                            'seconds': {'minimum': 0,
                                                        'type': 'integer'},
                                            'start_hours': {'minimum': 0,
                                                            'type': 'integer'},
                                            'start_minutes': {'minimum': 0,
                                                                'type': 'integer'},
                                            'start_seconds': {'minimum': 0,
                                                                'type': 'integer'}},
                            'required': [],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Removes time from a timer',
                'name': 'HassDecreaseTimer',
                'parameters': {'properties': {'area': {'type': 'string'},
                                            'hours': {'minimum': 0,
                                                        'type': 'integer'},
                                            'minutes': {'minimum': 0,
                                                        'type': 'integer'},
                                            'name': {'type': 'string'},
                                            'seconds': {'minimum': 0,
                                                        'type': 'integer'},
                                            'start_hours': {'minimum': 0,
                                                            'type': 'integer'},
                                            'start_minutes': {'minimum': 0,
                                                                'type': 'integer'},
                                            'start_seconds': {'minimum': 0,
                                                                'type': 'integer'}},
                            'required': [],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Pauses a running timer',
                'name': 'HassPauseTimer',
                'parameters': {'properties': {'area': {'type': 'string'},
                                            'name': {'type': 'string'},
                                            'start_hours': {'minimum': 0,
                                                            'type': 'integer'},
                                            'start_minutes': {'minimum': 0,
                                                                'type': 'integer'},
                                            'start_seconds': {'minimum': 0,
                                                                'type': 'integer'}},
                            'required': [],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Resumes a paused timer',
                'name': 'HassUnpauseTimer',
                'parameters': {'properties': {'area': {'type': 'string'},
                                            'name': {'type': 'string'},
                                            'start_hours': {'minimum': 0,
                                                            'type': 'integer'},
                                            'start_minutes': {'minimum': 0,
                                                                'type': 'integer'},
                                            'start_seconds': {'minimum': 0,
                                                                'type': 'integer'}},
                            'required': [],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Reports the current status of timers',
                'name': 'HassTimerStatus',
                'parameters': {'properties': {},
                            'required': [],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Broadcast a message through the home',
                'name': 'HassBroadcast',
                'parameters': {'properties': {'message': {'type': 'string'}},
                            'required': ['message'],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Resumes a media player',
                'name': 'HassMediaUnpause',
                'parameters': {'properties': {'area': {'type': 'string'},
                                            'device_class': {'items': {'enum': ['tv',
                                                                                'speaker',
                                                                                'receiver'],
                                                                        'type': 'string'},
                                                            'type': 'array'},
                                            'domain': {'items': {'enum': ['media_player'],
                                                                'type': 'string'},
                                                        'type': 'array'},
                                            'floor': {'type': 'string'},
                                            'name': {'type': 'string'}},
                            'required': [],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Pauses a media player',
                'name': 'HassMediaPause',
                'parameters': {'properties': {'area': {'type': 'string'},
                                            'device_class': {'items': {'enum': ['tv',
                                                                                'speaker',
                                                                                'receiver'],
                                                                        'type': 'string'},
                                                            'type': 'array'},
                                            'domain': {'items': {'enum': ['media_player'],
                                                                'type': 'string'},
                                                        'type': 'array'},
                                            'floor': {'type': 'string'},
                                            'name': {'type': 'string'}},
                            'required': [],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Skips a media player to the next item',
                'name': 'HassMediaNext',
                'parameters': {'properties': {'area': {'type': 'string'},
                                            'device_class': {'items': {'enum': ['tv',
                                                                                'speaker',
                                                                                'receiver'],
                                                                        'type': 'string'},
                                                            'type': 'array'},
                                            'domain': {'items': {'enum': ['media_player'],
                                                                'type': 'string'},
                                                        'type': 'array'},
                                            'floor': {'type': 'string'},
                                            'name': {'type': 'string'}},
                            'required': [],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Replays the previous item for a media '
                            'player',
                'name': 'HassMediaPrevious',
                'parameters': {'properties': {'area': {'type': 'string'},
                                            'device_class': {'items': {'enum': ['tv',
                                                                                'speaker',
                                                                                'receiver'],
                                                                        'type': 'string'},
                                                            'type': 'array'},
                                            'domain': {'items': {'enum': ['media_player'],
                                                                'type': 'string'},
                                                        'type': 'array'},
                                            'floor': {'type': 'string'},
                                            'name': {'type': 'string'}},
                            'required': [],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Sets the volume percentage of a media '
                            'player',
                'name': 'HassSetVolume',
                'parameters': {'properties': {'area': {'type': 'string'},
                                            'device_class': {'items': {'enum': ['tv',
                                                                                'speaker',
                                                                                'receiver'],
                                                                        'type': 'string'},
                                                            'type': 'array'},
                                            'domain': {'items': {'enum': ['media_player'],
                                                                'type': 'string'},
                                                        'type': 'array'},
                                            'floor': {'type': 'string'},
                                            'name': {'type': 'string'},
                                            'volume_level': {'description': 'The '
                                                                            'volume '
                                                                            'percentage '
                                                                            'of '
                                                                            'the '
                                                                            'media '
                                                                            'player',
                                                            'maximum': 100,
                                                            'minimum': 0,
                                                            'type': 'integer'}},
                            'required': ['volume_level'],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Increases or decreases the volume of '
                            'a media player',
                'name': 'HassSetVolumeRelative',
                'parameters': {'properties': {'area': {'type': 'string'},
                                            'floor': {'type': 'string'},
                                            'name': {'type': 'string'},
                                            'volume_step': {'anyOf': [{'enum': ['down',
                                                                                'up'],
                                                                        'type': 'string'},
                                                                        {'maximum': 100,
                                                                        'minimum': -100,
                                                                        'type': 'integer'}]}},
                            'required': ['volume_step'],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Searches for media and plays the '
                            'first result',
                'name': 'HassMediaSearchAndPlay',
                'parameters': {'properties': {'area': {'type': 'string'},
                                            'floor': {'type': 'string'},
                                            'media_class': {'enum': ['album',
                                                                    'app',
                                                                    'artist',
                                                                    'channel',
                                                                    'composer',
                                                                    'contributing_artist',
                                                                    'directory',
                                                                    'episode',
                                                                    'game',
                                                                    'genre',
                                                                    'image',
                                                                    'movie',
                                                                    'music',
                                                                    'playlist',
                                                                    'podcast',
                                                                    'season',
                                                                    'track',
                                                                    'tv_show',
                                                                    'url',
                                                                    'video'],
                                                            'type': 'string'},
                                            'name': {'type': 'string'},
                                            'search_query': {'type': 'string'}},
                            'required': ['search_query'],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Add item to a todo list',
                'name': 'HassListAddItem',
                'parameters': {'properties': {'item': {'type': 'string'},
                                            'name': {'type': 'string'}},
                            'required': ['item', 'name'],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Complete item on a todo list',
                'name': 'HassListCompleteItem',
                'parameters': {'properties': {'item': {'type': 'string'},
                                            'name': {'type': 'string'}},
                            'required': ['item', 'name'],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Query a to-do list to find out what '
                            'items are on it. Use this to answer '
                            "questions like 'What's on my task "
                            "list?' or 'Read my grocery list'. "
                            'Filters items by status '
                            '(needs_action, completed, all).',
                'name': 'todo_get_items',
                'parameters': {'properties': {'status': {'default': 'needs_action',
                                                        'description': 'Filter '
                                                                        'returned '
                                                                        'items '
                                                                        'by '
                                                                        'status, '
                                                                        'by '
                                                                        'default '
                                                                        'returns '
                                                                        'incomplete '
                                                                        'items',
                                                        'enum': ['needs_action',
                                                                'completed',
                                                                'all'],
                                                        'type': 'string'},
                                            'todo_list': {'enum': ['To-do'],
                                                            'type': 'string'}},
                            'required': ['todo_list'],
                            'type': 'object'}},
'type': 'function'},
{'function': {'name': 'get_current_weather',
                'parameters': {'properties': {},
                            'required': [],
                            'type': 'object'}},
'type': 'function'},
{'function': {'name': 'get_weather_forecast',
                'parameters': {'properties': {'forecast_type': {'enum': ['hourly',
                                                                        'daily'],
                                                                'type': 'string'}},
                            'required': [],
                            'type': 'object'}},
'type': 'function'},
{'function': {'description': 'Provides real-time information about '
                            'the CURRENT state, value, or mode of '
                            'devices, sensors, entities, or areas. '
                            'Use this tool for: 1. Answering '
                            'questions about current conditions '
                            "(e.g., 'Is the light on?'). 2. As the "
                            'first step in conditional actions '
                            "(e.g., 'If the weather is rainy, turn "
                            "off sprinklers' requires checking the "
                            'weather first).',
                'name': 'GetLiveContext',
                'parameters': {'properties': {},
                            'required': [],
                            'type': 'object'}},
'type': 'function'}
EOF

# --------------------------------------------------------------------------------------------------
# Full prompt

read -r -d '' PROMPT <<EOF
<|im_start|>system
${SYSTEM_PROMPT}

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
${TOOLS}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{'name': <function-name>, 'arguments': <args-json-object>}
</tool_call>
<|im_end|>
<|im_start|>user
${USER_MESSAGE}
<|im_end|>
<|im_start|>assistant
EOF

# echo "${PROMPT}"
# exit 0

# Runs model without using conversation mode. I think this is equivalent to asking the model to
# complete the prompt?
# ./build/bin/llama-cli -m "$QWEN3_8B_GGUF" -no-cnv --verbose-prompt -p "$PROMPT"
./build/bin/llama-simple -m "$QWEN3_8B_GGUF" -n 1000 "$PROMPT"
