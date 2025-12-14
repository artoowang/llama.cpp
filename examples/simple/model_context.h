#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "llama.h"

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
                                            const std::string& model_path);

  // Disable copy semantics.
  ModelContext(const ModelContext& other) = delete;
  ModelContext& operator=(const ModelContext& other) = delete;

  // Move semantics.
  ModelContext(ModelContext&& other) = default;
  ModelContext& operator=(ModelContext&& other) = default;

  // Adds given system message into the model.
  bool AddSystemMessage(absl::string_view system_message);

  // Adds given user message into the model.
  bool AddUserMessage(absl::string_view user_message);

  // Samples the model until receiving end-of-generation token. Returns the
  // sampled result in string.
  std::string SampleUntilEndOfGeneration();

  // Takes a snapshot of the current model state. The snapshot is stored within
  // this object. Only one snapshot is stored at a time: a new snapshot will
  // overwrite the previous one. Returns false at failure.
  bool TakeSnapshot();

  // Restores the previously taken snapshot. Returns false if there is no prior
  // snapshot, or if the restoration fails.
  bool RestoreSnapshot();

  // Prints performance metrics.
  void PrintPerformanceMetric() const;

  // Access the raw objects.
  // TODO: These are only here in order to call llama_*() methods. Ideally, all
  // functionalities should be implemented within this class instead.
  //
  // const llama_vocab* GetVocab() const { return vocab_; }
  // const llama_model* GetModel() const { return model_.get(); }
  // llama_context* GetContext() { return ctx_.get(); }
  // llama_sampler* GetSampler() { return sampler_.get(); }

 private:
  ModelContext(ModelPtr model, ContextPtr ctx);

  ModelPtr model_;
  std::unique_ptr<llama_context, decltype(&llama_free)> ctx_;
  std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler_;
  const llama_vocab* vocab_;
  std::vector<uint8_t> snapshot_;
};
