load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

NSYNC_COMMIT = "82b118aa7ace3132e517e2c467f8732978cf4023"

NSYNC_SHA256 = ""

http_archive(
    name = "nsync",
    sha256 = NSYNC_SHA256,
    strip_prefix = "nsync-" + NSYNC_COMMIT,
    urls = ["https://github.com/wsmoses/nsync/archive/{commit}.tar.gz".format(commit = NSYNC_COMMIT)],
)

ENZYMEXLA_COMMIT = "4d276779410f2fb0abe1e2ec65ec40bc2dd52365"

ENZYMEXLA_SHA256 = ""

http_archive(
    name = "enzyme_ad",
    sha256 = ENZYMEXLA_SHA256,
    strip_prefix = "Enzyme-JAX-" + ENZYMEXLA_COMMIT,
    urls = ["https://github.com/EnzymeAD/Enzyme-JAX/archive/{commit}.tar.gz".format(commit = ENZYMEXLA_COMMIT)],
)

http_archive(
    name = "rules_python",
    sha256 = "778aaeab3e6cfd56d681c89f5c10d7ad6bf8d2f1a72de9de55b23081b2d31618",
    strip_prefix = "rules_python-0.34.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.34.0/rules_python-0.34.0.tar.gz",
)

# Hedron's Compile Commands Extractor for Bazel
# https://github.com/hedronvision/bazel-compile-commands-extractor
http_archive(
    name = "hedron_compile_commands",
    strip_prefix = "bazel-compile-commands-extractor-4f28899228fb3ad0126897876f147ca15026151e",

    # Replace the commit hash (0e990032f3c5a866e72615cf67e5ce22186dcb97) in both places (below) with the latest (https://github.com/hedronvision/bazel-compile-commands-extractor/commits/main), rather than using the stale one here.
    # Even better, set up Renovate and let it do the work for you (see "Suggestion: Updates" in the README).
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/4f28899228fb3ad0126897876f147ca15026151e.tar.gz",
    # When you first run this tool, it'll recommend a sha256 hash to put here with a message like: "DEBUG: Rule 'hedron_compile_commands' indicated that a canonical reproducible form can be obtained by modifying arguments sha256 = ..."
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")

hedron_compile_commands_setup()

load("@hedron_compile_commands//:workspace_setup_transitive.bzl", "hedron_compile_commands_setup_transitive")

hedron_compile_commands_setup_transitive()

load("@hedron_compile_commands//:workspace_setup_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive")

hedron_compile_commands_setup_transitive_transitive()

load("@hedron_compile_commands//:workspace_setup_transitive_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive_transitive")

hedron_compile_commands_setup_transitive_transitive_transitive()

load("@enzyme_ad//:workspace.bzl", "ENZYME_COMMIT", "ENZYME_SHA256", "JAX_COMMIT", "JAX_SHA256", "XLA_PATCHES")

XLA_PATCHES = XLA_PATCHES + [
    """
sed -i.bak0 "s/__cpp_lib_hardware_interference_size/HW_INTERFERENCE_SIZE/g" xla/backends/cpu/runtime/thunk_executor.h
""",
    """
sed -i.bak0 "s/__cpp_lib_hardware_interference_size/HW_INTERFERENCE_SIZE/g" xla/tsl/concurrency/async_value_ref.h
""",
    """
sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/HAVE_LINK_H=1\\/HAVE_LINK_H=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
""",
    """
sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/LLVM_ENABLE_THREADS=1\\/LLVM_ENABLE_THREADS=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
""",
    """
sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/HAVE_MALLINFO=1\\/DONT_HAVE_ANY_MALLINFO=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
""",
    """
sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/HAVE_PTHREAD_GETNAME_NP=1\\/FAKE_HAVE_PTHREAD_GETNAME_NP=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
""",
    """
sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/HAVE_PTHREAD_SETNAME_NP=1\\/FAKE_HAVE_PTHREAD_SETNAME_NP=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
""",
    """
sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.h -exec sed -i.bak0 's\\/ENABLE_CRASH_OVERRIDES 1\\/ENABLE_CRASH_OVERRIDES 0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
""",
    """
sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.h -exec sed -i.bak0 's\\/HAVE_PTHREAD_GETNAME_NP\\/FAKE_HAVE_PTHREAD_GETNAME_NP\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
""",
    """
sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.h -exec sed -i.bak0 's\\/HAVE_PTHREAD_SETNAME_NP\\/FAKE_HAVE_PTHREAD_SETNAME_NP\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
""",
    # """
    # sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\['find . -type f -name BUILD.bazel -exec sed -i.bak0 \\\\\\'s\\/\\\"CAPIIR\\\",\\/\\\"CAPIIR\\\",alwayslink=1,\\/g\\\\\\\\' {} +',/g" third_party/llvm/workspace.bzl
    # """,
]

LLVM_TARGETS = select({
    "@bazel_tools//src/conditions:windows": [
        "AMDGPU",
        "NVPTX",
    ],
    "@bazel_tools//src/conditions:darwin": [],
    "//conditions:default": [
        "AMDGPU",
        "NVPTX",
    ],
}) + [
    "AArch64",
    "X86",
    "ARM",
]

# Uncomment these lines to use a custom LLVM commit
# LLVM_COMMIT = "b39c5cb6977f35ad727d86b2dd6232099734ffd3"
# LLVM_SHA256 = ""
# http_archive(
#     name = "llvm-raw",
#     build_file_content = "# empty",
#     sha256 = LLVM_SHA256,
#     strip_prefix = "llvm-project-" + LLVM_COMMIT,
#     urls = ["https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)],
# )
#
#
# load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
# maybe(
#     http_archive,
#     name = "llvm_zlib",
#     build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
#     sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
#     strip_prefix = "zlib-ng-2.0.7",
#     urls = [
#         "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
#     ],
# )
#
# maybe(
#     http_archive,
#     name = "llvm_zstd",
#     build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
#     sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
#     strip_prefix = "zstd-1.5.2",
#     urls = [
#         "https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz"
#     ],
# )

http_archive(
    name = "jax",
    patch_args = ["-p1"],
    patches = ["@enzyme_ad//:patches/jax.patch"],
    sha256 = JAX_SHA256,
    strip_prefix = "jax-" + JAX_COMMIT,
    urls = ["https://github.com/google/jax/archive/{commit}.tar.gz".format(commit = JAX_COMMIT)],
)

# load("@jax//third_party/xla:workspace.bzl", "XLA_COMMIT", "XLA_SHA256")
XLA_COMMIT = "281c11225c4a0bb7b710a290610a06d71194febd"

XLA_SHA256 = ""

http_archive(
    name = "xla",
    patch_cmds = XLA_PATCHES,
    sha256 = XLA_SHA256,
    strip_prefix = "xla-" + XLA_COMMIT,
    urls = ["https://github.com/wsmoses/xla/archive/{commit}.tar.gz".format(commit = XLA_COMMIT)],
)

load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    requirements = {
        "3.9": "//build:requirements_lock_3_9.txt",
        "3.10": "//build:requirements_lock_3_10.txt",
        "3.11": "//build:requirements_lock_3_11.txt",
        "3.12": "//build:requirements_lock_3_12.txt",
        "3.13": "//build:requirements_lock_3_13.txt",
    },
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()
#
# load("@xla//third_party/py:python_init_pip.bzl", "python_init_pip")
# python_init_pip()
#
# load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")
# python_init_rules()
#
# load("@rules_python//python:repositories.bzl", "py_repositories")
#
# py_repositories()
#
# load("@rules_python//python/pip_install:repositories.bzl", "pip_install_dependencies")
#
# pip_install_dependencies()

http_archive(
    name = "enzyme",
    sha256 = ENZYME_SHA256,
    strip_prefix = "Enzyme-" + ENZYME_COMMIT + "/enzyme",
    urls = ["https://github.com/EnzymeAD/Enzyme/archive/{commit}.tar.gz".format(commit = ENZYME_COMMIT)],
)

http_archive(
    name = "build_bazel_rules_apple",
    sha256 = "34c41bfb59cdaea29ac2df5a2fa79e5add609c71bb303b2ebb10985f93fa20e7",
    url = "https://github.com/bazelbuild/rules_apple/releases/download/3.1.1/rules_apple.3.1.1.tar.gz",
)

load(
    "@build_bazel_rules_apple//apple:repositories.bzl",
    "apple_rules_dependencies",
)

apple_rules_dependencies()

http_archive(
    name = "upb",
    patch_cmds = [
        "sed -i.bak0 's/@bazel_tools\\/\\/platforms:windows/@platforms\\/\\/os:windows/g' BUILD",
        "sed -i.bak0 's/-Werror//g' BUILD",
    ],
    sha256 = "61d0417abd60e65ed589c9deee7c124fe76a4106831f6ad39464e1525cef1454",
    strip_prefix = "upb-9effcbcb27f0a665f9f345030188c0b291e32482",
    url = "https://github.com/protocolbuffers/upb/archive/9effcbcb27f0a665f9f345030188c0b291e32482.tar.gz",
)

load("@jax//third_party/xla:workspace.bzl", jax_xla_workspace = "repo")

jax_xla_workspace()

load("@xla//:workspace4.bzl", "xla_workspace4")

xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")

xla_workspace3()

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")
load("@xla//:workspace2.bzl", "xla_workspace2")

llvm_configure(
    name = "llvm-project",
    targets = LLVM_TARGETS,
)

xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")

xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")

xla_workspace0()

load("@jax//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")

flatbuffers()

load(
    "@tsl//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@tsl//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@tsl//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@tsl//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@tsl//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")
