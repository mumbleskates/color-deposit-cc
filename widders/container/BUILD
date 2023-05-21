load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "kd_common",
    hdrs = [
        "kd_metrics.h",
        "kd_searchers.h",
        "kd_value_traits.h",
    ],
)

cc_library(
    name = "scapegoat_kd_map",
    hdrs = ["scapegoat_kd_map.h"],
    deps = [
        ":kd_common",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)
