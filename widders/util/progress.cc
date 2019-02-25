#include "widders/util/progress.h"

#include <algorithm>

absl::Duration ProgressStats::eta(int64_t to_target_progress) {
  return absl::Seconds((to_target_progress - progress) / harmonic_update_rate);
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
      2.0 / (1 / recent_rate +
             absl::ToDoubleSeconds(elapsed) / (progress - initial_progress_));
  last_output_time_ = actual_now_time;
  last_output_progress_ = progress;
  next_check_progress_ =
      progress +
      static_cast<int64_t>(harmonic_rate *
                           absl::ToDoubleSeconds(target_output_interval_));
  output_({progress, elapsed, recent_rate, harmonic_rate});
}

absl::Duration Progress::elapsed() { return absl::Now() - start_time_; }
