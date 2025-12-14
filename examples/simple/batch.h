#include <optional>
#include <vector>

#include "absl/strings/string_view.h"
#include "llama.h"

// Holds a llama_batch along with the storage for its tokens.
class Batch {
 public:
  // Creates a Batch from a prompt string. Returns std::nullopt on failure.
  static std::optional<Batch> CreateFromPrompt(
      const llama_vocab* vocab, const std::vector<absl::string_view>& prompts);

  // Prints the given `llama_batch` information.
  static void PrintLlamaBatch(const llama_batch& batch);

  // Disable copy semantics.
  Batch(const Batch& other) = delete;
  Batch& operator=(const Batch& other) = delete;

  // Move semantics.
  Batch(Batch&& other) = default;
  Batch& operator=(Batch&& other) = default;

  // Returns the llama_batch pointing to the tokens stored in this object.
  llama_batch Get() const { return batch_; }

  // Prints the tokens stored in this object.
  void PrintTokens() const;

  // Prints the Batch information.
  void Print() const { PrintLlamaBatch(batch_); }

 private:
  // Constructs a Batch from a vector of tokens and the vocabulary. The object
  // owns the tokens, but does not own the vocabulary, and the caller needs to
  // make sure the vocabulary remains valid.
  Batch(std::vector<llama_token>&& tokens, const llama_vocab* vocab);

  const llama_vocab* vocab_;
  std::vector<llama_token> tokens_;
  llama_batch batch_;
};
