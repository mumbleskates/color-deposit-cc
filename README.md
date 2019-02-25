# Color-deposit, C++ edition

This is a cute little program that draws pretty pictures that include exactly one pixel of each RGB
color. This is accomplished by processing each color in some random, semi-random, or deterministic
order and, starting at the center of the image, iteratively placing each one in an empty space next
to an occupied space whose surrounding colors are *the most similar to it.* If a tie occurs, the
placement is chosen randomly from among the ties; this means that otherwise deterministic outputs
have huge variation.

[Excerpts and examples of the kinds of output you might see](https://imgur.com/a/LJuD4ya)

Currently this implementation is single-threaded; this is mostly because it is far from trivial to
perform these calculations in parallel, as each pixel's placement potentially (usually) affects the
placement of the very next pixel. This means that throwing a bunch of CPU cores at the calculation
of the same image will not only take up a potentially significant amount of extra memory, but may
actually be *slower* than a single threaded attempt: if threads calculate 100 pixels' 100 best
candidate locations, by the time the 50th pixel has been synced all of the 100 preferred locations
may have been removed or altered, and the expensive calculation will need to happen again anyway,
single-threaded now (in many color orderings this would be a very common occurrence). Other
possibilities for partially mitigating this problem exist, but they are even more complex and have
even more overhead (immutable k-d trees and a live work queue? 🤔 very expensive).

Thus, if you want to make use of many threads to make lots of pretty pictures, I would currently
recommend simply running lots of instances of the program (be aware that it will consume at least
512MB of RAM and sometimes half-again that much* -- the images are big).

I originally came across this general concept
[here](https://codegolf.stackexchange.com/a/22326/85579) and was shocked and skeptical that it could
possibly take dozens of hours on a many-core machine to make these kinds of images. Turns out it
doesn't, thanks to spatial search trees.

** If you desire to consume less RAM, you can change the `using Field =` declaration in
color_deposit.cc to `float` to reduce the memory consumption by nearly half. But in this modern era
RAM is plentiful and it doesn't speed up the current implementation much at all -- computation is
dominated by nearest-neighbor searching overhead, not color-distance calculations.

## Building and running

Before building you will need the submodules with the prerequisites: `git submodule update --init`

Best built with [bazel](https://bazel.build/). The binary you care about is in the bazel target
//colors:color_deposit, so to run the program when you have Bazel working, the command
`bazel run -c opt colors:color_deposit -- `{flags go here} should get you started. The `--help` flag
prints out some helpful guidance for the supported options, but those strings are in the source code
as well. I would personally recommend always enabling `--log_progress`.

The executable should you wish to use it directly, and the output images, can be found under
`bazel-out` -- for instance,
.../bazel-out/k8-opt/bin/colors/color_deposit.runfiles/color_deposit_cc/output.png . Different
names for the output file can be chosen in a non-flag commandline argument, but the file format will
always be PNG.

Some options to try (the variations are very numerous):

- (no options)
- `--log_progress --color_metric=rgb`
- `--log_progress --color_metric=xyz`
- `--log_progress --ordering=ordered:-g`

These take a long time:

- `--log_progress --ordering=ordered:+r+g+b`
- `--log_progress --ordering=ordered:+b+g+r --color_metric=rgb`
- `--log_progress --ordering=ordered:-l`
- `--log_progress --ordering=ordered:-v`
