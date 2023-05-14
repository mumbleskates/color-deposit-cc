#ifndef CC_WIDDERS_UTIL_PROGRESS_H_
#define CC_WIDDERS_UTIL_PROGRESS_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>

#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace widders {

struct ProgressStats {
  // ETA to reach specified target.
  inline absl::Duration eta(int64_t to_target_progress);

  int64_t progress;             // Progress reported.
  absl::Duration elapsed;       // Total time elapsed.
  double recent_update_rate;    // Progress per second since last update.
  double harmonic_update_rate;  // Harmonic mean of recent and average rate.
};

struct ProgressOptions {
  // Function invoked whenever progress is updated.
  std::function<void(ProgressStats)> output_function;
  // Target duration between updates.
  absl::Duration target_output_interval = absl::Seconds(0.5);
  // Initial progress.
  int64_t initial_progress = 0;
  // Weighting given to update speed over the last update interval vs. over the
  // entire duration. The higher this number the more variable the ETA will be
  // based upon recent update speed vs. overall average speed so far.
  double harmonic_recent_weight = 0.2;
};

class Progress {
 public:
  explicit Progress(ProgressOptions options = {})
      : output_(std::move(options.output_function)),
        target_output_interval_(options.target_output_interval),
        harmonic_recent_weight_(options.harmonic_recent_weight),
        initial_progress_(options.initial_progress),
        start_time_(absl::Now()),
        last_output_progress_(options.initial_progress),
        last_output_time_(start_time_),
        next_check_time_(start_time_ + target_output_interval_) {}

  // Update progress to output function if we should.
  inline void update(int64_t progress);
  // Time elapsed since init.
  inline absl::Duration elapsed();

 private:
  const std::function<void(ProgressStats)> output_;
  const absl::Duration target_output_interval_;
  const double harmonic_recent_weight_;
  const int64_t initial_progress_;
  const absl::Time start_time_;

  int64_t last_output_progress_;
  absl::Time last_output_time_;
  absl::Time next_check_time_;
};

// -------------------------------------------------------------------------- //

inline absl::Duration ProgressStats::eta(int64_t to_target_progress) {
  return absl::Seconds(static_cast<double>(to_target_progress - progress) /
                       harmonic_update_rate);
}

inline void Progress::update(int64_t progress) {
  const absl::Time now_time = absl::Now();
  if (now_time < next_check_time_) return;
  const absl::Duration elapsed = now_time - start_time_;
  const absl::Duration since_last_output = now_time - last_output_time_;
  const int64_t progress_made = progress - last_output_progress_;
  const double recent_rate =
      progress_made / absl::ToDoubleSeconds(since_last_output);
  const double harmonic_rate =
      (1.0 + harmonic_recent_weight_) /
      (harmonic_recent_weight_ / recent_rate +
       absl::ToDoubleSeconds(elapsed) /
           static_cast<double>(progress - initial_progress_));
  last_output_time_ = now_time;
  last_output_progress_ = progress;
  next_check_time_ = now_time + target_output_interval_;
  output_({.progress = progress,
           .elapsed = elapsed,
           .recent_update_rate = recent_rate,
           .harmonic_update_rate = harmonic_rate});
}

inline absl::Duration Progress::elapsed() { return absl::Now() - start_time_; }

}  // namespace widders

#endif  // CC_WIDDERS_UTIL_PROGRESS_H_
