#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"

namespace {

const std::string kSystemPrompt = R"(
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

const std::string kUserMessage = "Timer, one minute.";

const std::string kPrompt = R"(
<|im_start|>system
)" + kSystemPrompt + R"(

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
)" + kTools + R"(
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{'name': <function-name>, 'arguments': <args-json-object>}
</tool_call>
<|im_end|>
<|im_start|>user
)" + kUserMessage + R"(
<|im_end|>
<|im_start|>assistant
)";

void print_usage(int, char** argv) {
  printf("\nexample usage:\n");
  printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n",
         argv[0]);
  printf("\n");
}

void print_llama_batch(const llama_batch& batch) {
  printf("llama_batch{n_tokens=%d%s%s%s%s%s%s}\n", batch.n_tokens,
         batch.token != nullptr ? "" : ",tokens=null",
         batch.embd != nullptr ? "" : ",embd=null",
         batch.pos != nullptr ? "" : ",pos=null",
         batch.n_seq_id != nullptr ? "" : ",n_seq_id=null",
         batch.seq_id != nullptr ? "" : ",seq_id=null",
         batch.logits != nullptr ? "" : ",logits=null");
}

}  // namespace

int main(int argc, char** argv) {
  // path to the model gguf file
  std::string model_path;
  // prompt to generate text from
  std::string prompt = kPrompt;
  // number of layers to offload to the GPU
  int ngl = 99;
  // number of tokens to predict
  int n_predict = 32;

  // parse command line arguments

  {
    int i = 1;
    for (; i < argc; i++) {
      if (strcmp(argv[i], "-m") == 0) {
        if (i + 1 < argc) {
          model_path = argv[++i];
        } else {
          print_usage(argc, argv);
          return 1;
        }
      } else if (strcmp(argv[i], "-n") == 0) {
        if (i + 1 < argc) {
          try {
            n_predict = std::stoi(argv[++i]);
          } catch (...) {
            print_usage(argc, argv);
            return 1;
          }
        } else {
          print_usage(argc, argv);
          return 1;
        }
      } else if (strcmp(argv[i], "-ngl") == 0) {
        if (i + 1 < argc) {
          try {
            ngl = std::stoi(argv[++i]);
          } catch (...) {
            print_usage(argc, argv);
            return 1;
          }
        } else {
          print_usage(argc, argv);
          return 1;
        }
      } else {
        // prompt starts here
        break;
      }
    }
    if (model_path.empty()) {
      print_usage(argc, argv);
      return 1;
    }
    if (i < argc) {
      prompt = argv[i++];
      for (; i < argc; i++) {
        prompt += " ";
        prompt += argv[i];
      }
    }
  }

  // load dynamic backends

  ggml_backend_load_all();

  // initialize the model

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = ngl;

  llama_model* model =
      llama_model_load_from_file(model_path.c_str(), model_params);

  if (model == NULL) {
    fprintf(stderr, "%s: error: unable to load model\n", __func__);
    return 1;
  }

  const llama_vocab* vocab = llama_model_get_vocab(model);
  // tokenize the prompt

  // find the number of tokens in the prompt
  const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                       NULL, 0, true, true);

  // allocate space for the tokens and tokenize the prompt
  std::vector<llama_token> prompt_tokens(n_prompt);
  if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(),
                     prompt_tokens.size(), true, true) < 0) {
    fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
    return 1;
  }

  // initialize the context

  llama_context_params ctx_params = llama_context_default_params();
  // n_ctx is the context size
  ctx_params.n_ctx = n_prompt + n_predict - 1;
  // n_batch is the maximum number of tokens that can be processed in a single
  // call to llama_decode
  ctx_params.n_batch = n_prompt;
  // enable performance counters
  ctx_params.no_perf = false;

  llama_context* ctx = llama_init_from_model(model, ctx_params);

  if (ctx == NULL) {
    fprintf(stderr, "%s: error: failed to create the llama_context\n",
            __func__);
    return 1;
  }

  // initialize the sampler

  auto sparams = llama_sampler_chain_default_params();
  sparams.no_perf = false;
  llama_sampler* smpl = llama_sampler_chain_init(sparams);

  llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

  // print the prompt token-by-token

  for (auto id : prompt_tokens) {
    char buf[128];
    int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    if (n < 0) {
      fprintf(stderr, "%s: error: failed to convert token to piece\n",
              __func__);
      return 1;
    }
    std::string s(buf, n);
    printf("%s", s.c_str());
  }
  printf("\n");

  // prepare a batch for the prompt

  llama_batch batch =
      llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
  print_llama_batch(batch);

  if (llama_model_has_encoder(model)) {
    printf("Model has encoder\n");
    if (llama_encode(ctx, batch)) {
      fprintf(stderr, "%s : failed to eval\n", __func__);
      return 1;
    }

    llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
    if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
      decoder_start_token_id = llama_vocab_bos(vocab);
    }

    batch = llama_batch_get_one(&decoder_start_token_id, 1);
    print_llama_batch(batch);
  }

  // The next position to decode.
  int n_pos = 0;

  // Evaluate the initial batch with the transformer model.
  const int64_t prompt_decode_start = ggml_time_us();
  if (llama_decode(ctx, batch)) {
    fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
    return 1;
  }
  printf("Prompt decode: %.2f s\n",
         (ggml_time_us() - prompt_decode_start) / 1e6f);
  n_pos += batch.n_tokens;

  // main loop

  printf("Main loop starts\n");

  const auto t_main_start = ggml_time_us();
  int n_decode = 0;
  llama_token new_token_id;

  printf("ZZZ: n_prompt=%d n_predict=%d\n", n_prompt, n_predict);
  while (n_pos < n_prompt + n_predict) {
    printf("ZZZ: n_pos=%d n_decode=%d batch.n_tokens=%d\n", n_pos, n_decode,
           batch.n_tokens);

    // sample the next token
    {
      new_token_id = llama_sampler_sample(smpl, ctx, -1);

      // is it an end of generation?
      if (llama_vocab_is_eog(vocab, new_token_id)) {
        break;
      }

      char buf[128];
      int n =
          llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
      if (n < 0) {
        fprintf(stderr, "%s: error: failed to convert token to piece\n",
                __func__);
        return 1;
      }
      std::string s(buf, n);
      // printf("%s", s.c_str());
      printf("ZZZ: output=%s\n", s.c_str());
      fflush(stdout);

      // prepare the next batch with the sampled token
      batch = llama_batch_get_one(&new_token_id, 1);
      // TODO: Printing this will be mixed with the generated text.
      // print_llama_batch(batch);

      n_decode += 1;
    }

    // Evaluate the next batch with the transformer model.
    if (llama_decode(ctx, batch)) {
      fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
      return 1;
    }
    n_pos += batch.n_tokens;
  }

  printf("Main loop ends\n");

  const auto t_main_end = ggml_time_us();

  fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
          __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f,
          n_decode / ((t_main_end - t_main_start) / 1000000.0f));

  fprintf(stderr, "\n");
  llama_perf_sampler_print(smpl);
  llama_perf_context_print(ctx);
  fprintf(stderr, "\n");

  llama_sampler_free(smpl);
  llama_free(ctx);
  llama_model_free(model);

  return 0;
}
