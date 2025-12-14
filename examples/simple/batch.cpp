#include "batch.h"

#include "absl/log/log.h"

std::optional<Batch> Batch::CreateFromPrompt(
    const llama_vocab* vocab, const std::vector<absl::string_view>& prompts) {
  // Tokenize the prompts.
  int total_tokens = 0;
  // TODO: This might not be needed.
  std::vector<int> prompt_tokens(prompts.size());
  for (size_t i = 0; i < prompts.size(); ++i) {
    // Find the number of tokens in the prompt.
    int num_tokens = -llama_tokenize(vocab, prompts[i].data(),
                                     prompts[i].size(), nullptr, 0, true, true);
    prompt_tokens[i] = num_tokens;
    total_tokens += num_tokens;
  }

  // Allocate space for the tokens and tokenize the prompt.
  std::vector<llama_token> tokens(total_tokens);
  int max_tokens = tokens.size();
  llama_token* tokens_ptr = tokens.data();
  for (size_t i = 0; i < prompts.size(); ++i) {
    int num_tokens = llama_tokenize(vocab, prompts[i].data(), prompts[i].size(),
                                    tokens_ptr, max_tokens, true, true);
    if (num_tokens != prompt_tokens[i]) {
      LOG(ERROR) << "failed to tokenize the prompt";
      return std::nullopt;
    }
    max_tokens -= num_tokens;
    tokens_ptr += num_tokens;
  }

  return Batch(std::move(tokens), vocab);
}

void Batch::PrintLlamaBatch(const llama_batch& batch) {
  LOG(INFO) << "llama_batch{n_tokens=" << batch.n_tokens
            << (batch.token != nullptr ? "" : ",tokens=null")
            << (batch.embd != nullptr ? "" : ",embd=null")
            << (batch.pos != nullptr ? "" : ",pos=null")
            << (batch.n_seq_id != nullptr ? "" : ",n_seq_id=null")
            << (batch.seq_id != nullptr ? "" : ",seq_id=null")
            << (batch.logits != nullptr ? "" : ",logits=null") << "}";
}

void Batch::PrintTokens() const {
  // print the prompt token-by-token
  std::string str;
  for (auto id : tokens_) {
    char buf[128];
    int n = llama_token_to_piece(vocab_, id, buf, sizeof(buf), 0, true);
    if (n < 0) {
      LOG(ERROR) << "error: failed to convert token to piece";
      return;
    }
    str.append(std::string(buf, n));
  }
  LOG(INFO) << "Batch tokens:\n" << str;
}

Batch::Batch(std::vector<llama_token>&& tokens, const llama_vocab* vocab)
    : vocab_(vocab),
      tokens_(std::move(tokens)),
      batch_(llama_batch_get_one(tokens_.data(), tokens_.size())) {}
