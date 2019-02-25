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
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)
