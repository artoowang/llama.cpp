#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
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

// Holds a llama_batch along with the storage for its tokens.
class Batch {
 public:
  // Creates a Batch from a prompt string. Returns std::nullopt on failure.
  static std::optional<Batch> CreateFromPrompt(const llama_vocab* vocab,
                                               const std::string& prompt) {
    // Tokenize the prompt.
    // Find the number of tokens in the prompt.
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                         nullptr, 0, true, true);
    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(),
                       tokens.size(), true, true) < 0) {
      fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
      return std::nullopt;
    }

    return Batch(std::move(tokens), vocab);
  }

  // Disable copy semantics.
  Batch(const Batch& other) = delete;
  Batch& operator=(const Batch& other) = delete;

  // Move semantics.
  Batch(Batch&& other) = default;
  Batch& operator=(Batch&& other) = default;

  // Returns the llama_batch pointing to the tokens stored in this object.
  llama_batch Get() const { return batch_; }

  // Prints the tokens stored in this object.
  void PrintTokens() const {
    // print the prompt token-by-token
    for (auto id : tokens_) {
      char buf[128];
      int n = llama_token_to_piece(vocab_, id, buf, sizeof(buf), 0, true);
      if (n < 0) {
        fprintf(stderr, "%s: error: failed to convert token to piece\n",
                __func__);
        return;
      }
      std::string s(buf, n);
      printf("%s", s.c_str());
    }
    printf("\n");
  }

  // Prints the Batch information.
  void Print() const { PrintLlamaBatch(batch_); }

  // Prints the given `llama_batch` information.
  static void PrintLlamaBatch(const llama_batch& batch) {
    printf("llama_batch{n_tokens=%d%s%s%s%s%s%s}\n", batch.n_tokens,
           batch.token != nullptr ? "" : ",tokens=null",
           batch.embd != nullptr ? "" : ",embd=null",
           batch.pos != nullptr ? "" : ",pos=null",
           batch.n_seq_id != nullptr ? "" : ",n_seq_id=null",
           batch.seq_id != nullptr ? "" : ",seq_id=null",
           batch.logits != nullptr ? "" : ",logits=null");
  }

 private:
  // Constructs a Batch from a vector of tokens and the vocabulary. The object
  // owns the tokens, but does not own the vocabulary, and the caller needs to
  // make sure the vocabulary remains valid.
  Batch(std::vector<llama_token>&& tokens, const llama_vocab* vocab)
      : vocab_(vocab),
        tokens_(std::move(tokens)),
        batch_(llama_batch_get_one(tokens_.data(), tokens_.size())) {}

  const llama_vocab* vocab_;
  std::vector<llama_token> tokens_;
  llama_batch batch_;
};

class ModelContext {
 private:
  // The following requires the deleter to be able to handle nullptr, which they
  // seem to do. Note that C++ standard guarantees delete nullptr is safe.
  using ModelPtr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
  using ContextPtr = std::unique_ptr<llama_context, decltype(&llama_free)>;
  using SampelrPtr =
      std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;

 public:
  static std::optional<ModelContext> Create(int32_t n_gpu_layers,
                                            const std::string& model_path) {
    // Initialize the model.
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    ModelPtr model(llama_model_load_from_file(model_path.c_str(), model_params),
                   llama_model_free);
    if (model == nullptr) {
      fprintf(stderr, "%s: error: unable to load model\n", __func__);
      return std::nullopt;
    }

    // TODO: We currently do not support encoder.
    if (llama_model_has_encoder(model.get())) {
      fprintf(stderr, "Encoder is currently not supported.");
      return std::nullopt;
    }

    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    ctx_params.n_ctx = 8192;
    // n_batch is the maximum number of tokens that can be processed in a single
    // call to llama_decode
    ctx_params.n_batch = 8192;
    // enable performance counters
    ctx_params.no_perf = false;
    ContextPtr ctx(llama_init_from_model(model.get(), ctx_params), llama_free);
    if (ctx == nullptr) {
      fprintf(stderr, "%s: error: failed to create the llama_context\n",
              __func__);
      return std::nullopt;
    }

    return ModelContext(std::move(model), std::move(ctx));
  }

  // Disable copy semantics.
  ModelContext(const ModelContext& other) = delete;
  ModelContext& operator=(const ModelContext& other) = delete;

  // Move semantics.
  ModelContext(ModelContext&& other) = default;
  ModelContext& operator=(ModelContext&& other) = default;

  // Processes the given `prompt` into the model.
  bool ProcessPrompt(const std::string& prompt) {
    std::optional<Batch> prompt_batch_opt =
        Batch::CreateFromPrompt(vocab_, prompt);
    if (!prompt_batch_opt.has_value()) {
      fprintf(stderr, "Failed to create batch for prompt\n");
      return false;
    }
    Batch prompt_batch(std::move(prompt_batch_opt.value()));

    // TODO: Test
    prompt_batch.PrintTokens();
    prompt_batch.Print();

    // Evaluate the initial batch with the transformer model.
    const int64_t prompt_decode_start = ggml_time_us();
    if (llama_decode(ctx_.get(), prompt_batch.Get())) {
      fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
      return false;
    }

    // TODO: Test
    printf("Prompt decode: %.2f s\n",
           (ggml_time_us() - prompt_decode_start) / 1e6f);

    return true;
  }

  // Access the raw objects. These are only here in order to call llama_*()
  // methods. Ideally, all functionalities should be implemented within this
  // class instead.
  const llama_vocab* GetVocab() const { return vocab_; }
  const llama_model* GetModel() const { return model_.get(); }
  llama_context* GetContext() { return ctx_.get(); }
  llama_sampler* GetSampler() { return sampler_.get(); }

 private:
  ModelContext(ModelPtr model, ContextPtr ctx)
      : model_(std::move(model)),
        ctx_(std::move(ctx)),
        sampler_(nullptr, llama_sampler_free),
        vocab_(llama_model_get_vocab(model_.get())) {
    // Initialize the sampler.
    llama_sampler_chain_params params = llama_sampler_chain_default_params();
    params.no_perf = false;
    sampler_.reset(llama_sampler_chain_init(params));
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_greedy());
  }

  ModelPtr model_;
  std::unique_ptr<llama_context, decltype(&llama_free)> ctx_;
  std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler_;
  const llama_vocab* vocab_;
};

void PrintUsage(int, char** argv) {
  printf("\nexample usage:\n");
  printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n",
         argv[0]);
  printf("\n");
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
          PrintUsage(argc, argv);
          return 1;
        }
      } else if (strcmp(argv[i], "-n") == 0) {
        if (i + 1 < argc) {
          try {
            n_predict = std::stoi(argv[++i]);
          } catch (...) {
            PrintUsage(argc, argv);
            return 1;
          }
        } else {
          PrintUsage(argc, argv);
          return 1;
        }
      } else if (strcmp(argv[i], "-ngl") == 0) {
        if (i + 1 < argc) {
          try {
            ngl = std::stoi(argv[++i]);
          } catch (...) {
            PrintUsage(argc, argv);
            return 1;
          }
        } else {
          PrintUsage(argc, argv);
          return 1;
        }
      } else {
        // prompt starts here
        break;
      }
    }
    if (model_path.empty()) {
      PrintUsage(argc, argv);
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

  std::optional<ModelContext> model_ctx_opt =
      ModelContext::Create(ngl, model_path);
  if (!model_ctx_opt.has_value()) {
    fprintf(stderr, "Failed to create model and context\n");
    return 1;
  }
  ModelContext model_ctx(std::move(model_ctx_opt.value()));

  model_ctx.ProcessPrompt(prompt);

  // main loop
  printf("Main loop starts\n");

  const int64_t t_main_start = ggml_time_us();
  int n_decode = 0;
  llama_token new_token_id;

  while (true) {
    // sample the next token
    new_token_id = llama_sampler_sample(model_ctx.GetSampler(),
                                        model_ctx.GetContext(), -1);

    // is it an end of generation?
    if (llama_vocab_is_eog(model_ctx.GetVocab(), new_token_id)) {
      break;
    }

    char buf[128];
    int n = llama_token_to_piece(model_ctx.GetVocab(), new_token_id, buf,
                                 sizeof(buf), 0, true);
    if (n < 0) {
      fprintf(stderr, "%s: error: failed to convert token to piece\n",
              __func__);
      return 1;
    }
    std::string s(buf, n);
    printf("%s", s.c_str());
    fflush(stdout);

    // prepare the next batch with the sampled token
    llama_batch batch = llama_batch_get_one(&new_token_id, 1);

    n_decode += 1;

    // Evaluate the next batch with the transformer model.
    if (llama_decode(model_ctx.GetContext(), batch)) {
      fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
      return 1;
    }
  }

  printf("Main loop ends\n");

  const auto t_main_end = ggml_time_us();

  fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
          __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f,
          n_decode / ((t_main_end - t_main_start) / 1000000.0f));

  fprintf(stderr, "\n");
  llama_perf_sampler_print(model_ctx.GetSampler());
  llama_perf_context_print(model_ctx.GetContext());
  fprintf(stderr, "\n");

  return 0;
}
