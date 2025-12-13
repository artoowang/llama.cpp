#include "absl/log/log_sink.h"

class ColorLogSink : public absl::LogSink {
 public:
  void Send(const absl::LogEntry& entry) override;
};
