#include <cstring>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/log/log_sink_registry.h"
#include "absl/strings/string_view.h"
#include "color_log_sink.h"
#include "model_context.h"

namespace {

const std::string kStaticSystemPrompt = R"(
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
)";

const std::string kTools = R"(
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
)";

const std::string kStaticSystemAndToolPrompt = R"(
)" + kStaticSystemPrompt + R"(

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
)" + kTools + R"(
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{'name': <function-name>, 'arguments': <args-json-object>}
</tool_call>
)";

}  // namespace

ABSL_FLAG(std::string, model, "", "Path to GGUF model.");
ABSL_FLAG(int, ngl, 99, "Number of layers to offload to GPU.");

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage(
      " --model model.gguf [--ngl n_gpu_layers] user_message");
  const std::vector<char*> pos_args = absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  // Disable the default LogSink to stderr, and use our own color log sink.
  absl::SetStderrThreshold(static_cast<absl::LogSeverity>(100));
  ColorLogSink color_log_sink;
  absl::AddLogSink(&color_log_sink);

  // path to the model gguf file
  const std::string model_path = absl::GetFlag(FLAGS_model);
  if (model_path.empty()) {
    LOG(ERROR) << "Model path is required.";
    return 1;
  }

  // number of layers to offload to the GPU
  const int ngl = absl::GetFlag(FLAGS_ngl);

  // User message. pos_args[0] is the application name.
  if (pos_args.size() < 2) {
    LOG(ERROR) << "Need user message.";
    return 1;
  }
  std::string user_message = pos_args[1];

  // load dynamic backends
  ggml_backend_load_all();

  std::optional<ModelContext> model_ctx_opt =
      ModelContext::Create(ngl, model_path);
  if (!model_ctx_opt.has_value()) {
    LOG(ERROR) << "Failed to create model and context.";
    return 1;
  }
  ModelContext model_ctx(std::move(model_ctx_opt.value()));

  // Add static system prompt.
  model_ctx.AddSystemMessage(kStaticSystemAndToolPrompt);
  model_ctx.TakeSnapshot();

  // Main interaction loop.
  while (!user_message.empty()) {
    // Restore the snapshot, add dynamic system prompt, user message, and then
    // generate the response.
    model_ctx.RestoreSnapshot();
    model_ctx.AddSystemMessage("The current time is 2025-12-13 18:03.");
    model_ctx.AddUserMessage(user_message);
    const std::string result = model_ctx.SampleUntilEndOfGeneration();
    LOG(INFO) << "Result:\n" << result;

    // Next user message.
    printf("User: ");
    std::getline(std::cin, user_message);
  }

  model_ctx.PrintPerformanceMetric();
  LOG(INFO) << "Done";

  return 0;
}
