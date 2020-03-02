#ifndef CC_WIDDERS_UTIL_PROGRESS_H_
#define CC_WIDDERS_UTIL_PROGRESS_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>

#include "absl/time/clock.h"
#include "absl/time/time.h"

struct ProgressStats {
  // ETA to reach specified target.
  inline absl::Duration eta(int64_t to_target_progress);

  int64_t progress;             // Progress reported.
  absl::Duration elapsed;       // Total time elapsed.
  double recent_update_rate;    // Progress per second since last update.
  double harmonic_update_rate;  // Harmonic mean of recent and average rate.
};

class Progress {
 public:
  explicit Progress(std::function<void(ProgressStats)> output,
                    absl::Duration target_output_interval = absl::Seconds(0.5),
                    int64_t initial_progress = 0,
                    int64_t first_output_progress = 1024)
      : output_(std::move(output)),
        target_output_interval_(target_output_interval),
        initial_progress_(initial_progress),
        start_time_(absl::Now()),
        last_output_progress_(initial_progress),
        next_check_progress_(first_output_progress),
        last_output_time_(start_time_) {}

  // Update progress to output function if we should.
  inline void update(int64_t progress);
  // Time elapsed since init.
  inline absl::Duration elapsed();

 private:
  const std::function<void(ProgressStats)> output_;
  const absl::Duration target_output_interval_;
  const int64_t initial_progress_;
  const absl::Time start_time_;

  int64_t last_output_progress_;
  int64_t next_check_progress_;
  absl::Time last_output_time_;
};

// -------------------------------------------------------------------------- //

absl::Duration ProgressStats::eta(int64_t to_target_progress) {
  return absl::Seconds(static_cast<double>(to_target_progress - progress) /
                       harmonic_update_rate);
}

void Progress::update(int64_t progress) {
  if (progress < next_check_progress_) return;

  const absl::Time actual_now_time = absl::Now();
  const absl::Time now_time =
      std::max(actual_now_time, last_output_time_ + absl::Microseconds(10));
  const absl::Duration elapsed = now_time - start_time_;
  const absl::Duration since_last_output = now_time - last_output_time_;
  const int64_t progress_made = progress - last_output_progress_;
  const double recent_rate =
      progress_made / absl::ToDoubleSeconds(since_last_output);
  const double harmonic_rate =
      2.0 /
      (1 / recent_rate + absl::ToDoubleSeconds(elapsed) /
                             static_cast<double>(progress - initial_progress_));
  last_output_time_ = actual_now_time;
  last_output_progress_ = progress;
  next_check_progress_ =
      progress +
      static_cast<int64_t>(harmonic_rate *
                           absl::ToDoubleSeconds(target_output_interval_));
  output_({progress, elapsed, recent_rate, harmonic_rate});
}

absl::Duration Progress::elapsed() { return absl::Now() - start_time_; }

#endif  // CC_WIDDERS_UTIL_PROGRESS_H_
