package(default_visibility = ["//visibility:public"])

cc_library(
    name = "kd_metrics",
    hdrs = ["kd_metrics.h"],
)

cc_library(
    name = "scapegoat_kd_map",
    hdrs = ["scapegoat_kd_map.h"],
    deps = [
        ":kd_metrics",
        "//widders/algorithm:medians",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)
