#include "model_context.h"

#include <iomanip>

#include "absl/log/log.h"
#include "batch.h"

std::optional<ModelContext> ModelContext::Create(
    int32_t n_gpu_layers, const std::string& model_path) {
  // Initialize the model.
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = n_gpu_layers;
  ModelPtr model(llama_model_load_from_file(model_path.c_str(), model_params),
                 llama_model_free);
  if (model == nullptr) {
    LOG(ERROR) << "Unable to load model.";
    return std::nullopt;
  }

  // TODO: We currently do not support encoder.
  if (llama_model_has_encoder(model.get())) {
    LOG(ERROR) << "Encoder is currently not supported.";
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
    LOG(ERROR) << "Failed to create the llama_context.";
    return std::nullopt;
  }

  return ModelContext(std::move(model), std::move(ctx));
}

bool ModelContext::AddSystemMessage(absl::string_view system_message) {
  const int64_t start_us = ggml_time_us();
  std::optional<Batch> batch = Batch::CreateFromPrompt(
      vocab_, {"<|im_start|>system\n", system_message, "\n<|im_end|>\n"});

  if (!batch.has_value()) {
    LOG(ERROR) << "Failed to create batch for system message: "
               << system_message;
    return false;
  }
  if (llama_decode(ctx_.get(), batch->Get())) {
    LOG(ERROR) << "Failed to decode.";
    return false;
  }

  // TODO: Test
  const int64_t end_us = ggml_time_us();
  LOG(INFO) << "System message processing: " << std::setprecision(2)
            << (end_us - start_us) / 1e6f << " s";
  batch->PrintTokens();
  batch->Print();

  return true;
}

bool ModelContext::AddUserMessage(absl::string_view user_message) {
  const int64_t start_us = ggml_time_us();
  std::optional<Batch> batch = Batch::CreateFromPrompt(
      vocab_, {"<|im_start|>user\n", user_message, "\n<|im_end|>\n"});

  if (!batch.has_value()) {
    LOG(ERROR) << "Failed to create batch for user message: " << user_message;
    return false;
  }
  if (llama_decode(ctx_.get(), batch->Get())) {
    LOG(ERROR) << "Failed to decode.";
    return false;
  }

  // TODO: Test
  const int64_t end_us = ggml_time_us();
  LOG(INFO) << "User message processing: " << std::setprecision(2)
            << (end_us - start_us) / 1e6f << " s";
  batch->PrintTokens();
  batch->Print();

  return true;
}

std::string ModelContext::SampleUntilEndOfGeneration() {
  const int64_t t_main_start = ggml_time_us();
  std::string result;

  // Prep the model to output assistant response.
  std::optional<Batch> batch =
      Batch::CreateFromPrompt(vocab_, {"<|im_start|>assistant\n"});
  // TODO: Test
  batch->PrintTokens();
  batch->Print();
  if (!batch.has_value()) {
    LOG(ERROR) << "Failed to create batch for assistant response.";
    return result;
  }
  if (llama_decode(ctx_.get(), batch->Get())) {
    LOG(ERROR) << "Failed to decode.";
    return result;
  }

  LOG(INFO) << "Generating tokens:";
  int n_decode = 0;
  while (true) {
    // sample the next token
    llama_token new_token_id =
        llama_sampler_sample(sampler_.get(), ctx_.get(), -1);

    // is it an end of generation?
    if (llama_vocab_is_eog(vocab_, new_token_id)) {
      break;
    }

    char buf[128];
    int n =
        llama_token_to_piece(vocab_, new_token_id, buf, sizeof(buf), 0, true);
    if (n < 0) {
      LOG(ERROR) << "Failed to convert token to piece.";
      break;
    }
    std::string s(buf, n);
    result.append(s);
    printf("%s", s.c_str());
    fflush(stdout);

    // prepare the next batch with the sampled token
    llama_batch batch = llama_batch_get_one(&new_token_id, 1);
    n_decode += 1;

    // Evaluate the next batch with the transformer model.
    if (llama_decode(ctx_.get(), batch)) {
      LOG(ERROR) << "Failed to eval.";
      break;
    }
  }
  // Make sure we start with a new line after the tokens.
  printf("\n");

  // TODO: The total time currently also includes printing log messages.
  const auto t_main_end = ggml_time_us();
  LOG(INFO) << "Decoded " << n_decode << " tokens in " << std::setprecision(2)
            << (t_main_end - t_main_start) / 1e6f
            << " s, speed: " << n_decode / ((t_main_end - t_main_start) / 1e6f)
            << " t/s";

  return result;
}

bool ModelContext::TakeSnapshot() {
  const int64_t start_us = ggml_time_us();
  const size_t state_size = llama_state_get_size(ctx_.get());

  snapshot_.resize(state_size);
  const size_t result =
      llama_state_get_data(ctx_.get(), snapshot_.data(), snapshot_.size());
  if (result != state_size) {
    LOG(ERROR) << "Failed to snapshot the state. Number of bytes copied: "
               << result << ", expected: " << state_size;
    return false;
  }

  const int64_t end_us = ggml_time_us();
  LOG(INFO) << "Snapshot state " << state_size << " bytes in "
            << std::setprecision(2) << (end_us - start_us) / 1e6f << " s";
  return true;
}

bool ModelContext::RestoreSnapshot() {
  if (snapshot_.empty()) {
    LOG(ERROR) << "No snapshot to restore.";
    return false;
  }

  const int64_t start_us = ggml_time_us();
  const size_t result =
      llama_state_set_data(ctx_.get(), snapshot_.data(), snapshot_.size());
  if (result != snapshot_.size()) {
    LOG(ERROR) << "Failed to restore the state. Number of bytes read: "
               << result << ", expected: " << snapshot_.size();
    return false;
  }

  const int64_t end_us = ggml_time_us();
  LOG(INFO) << "Restored snapshot of " << snapshot_.size() << " bytes in "
            << std::setprecision(2) << (end_us - start_us) / 1e6f << " s";
  return true;
}

void ModelContext::PrintPerformanceMetric() const {
  LOG(INFO) << "llama_perf:";
  llama_perf_sampler_print(sampler_.get());
  llama_perf_context_print(ctx_.get());
}

ModelContext::ModelContext(ModelPtr model, ContextPtr ctx)
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
