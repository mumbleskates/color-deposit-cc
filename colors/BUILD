load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

cc_library(
    name = "color_conversion",
    hdrs = ["color_conversion.h"],
)

cc_binary(
    name = "color_deposit",
    srcs = ["color_deposit.cc"],
    deps = [
        ":color_conversion",
        "//third_party:lodepng",
        "//widders/container:scapegoat_kd_map",
        "//widders/util:progress",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)
