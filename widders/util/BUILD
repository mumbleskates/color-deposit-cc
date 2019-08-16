load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "progress",
    srcs = ["progress.cc"],
    hdrs = ["progress.h"],
    deps = [
        "@com_google_absl//absl/time",
    ],
)
