#include "color_log_sink.h"

#include <iostream>

#include "absl/strings/string_view.h"

void ColorLogSink::Send(const absl::LogEntry& entry) {
  const char* color = "\033[0m";  // reset
  switch (entry.log_severity()) {
    case absl::LogSeverity::kInfo:
      color = "\033[32m";
      break;  // green
    case absl::LogSeverity::kWarning:
      color = "\033[33m";
      break;  // yellow
    case absl::LogSeverity::kError:
      color = "\033[31m";
      break;  // red
    case absl::LogSeverity::kFatal:
      color = "\033[1;31m";
      break;  // bright red
  }

  const absl::string_view text_message_with_prefix =
      entry.text_message_with_prefix();
  const absl::string_view text_message = entry.text_message();
  const absl::string_view::size_type prefix_length =
      text_message_with_prefix.length() - text_message.length();
  const absl::string_view prefix =
      text_message_with_prefix.substr(0, prefix_length);

  std::cout << "\033[36m" << prefix << color << entry.text_message()
            << "\033[0m\n";
}
