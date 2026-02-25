"""
python-for-android recipe for llama-cpp-python.
Compiles llama.cpp with the Android NDK and packages the Python bindings.
"""
from pythonforandroid.recipe import PythonRecipe, Recipe
from pythonforandroid.logger import shprint
from pythonforandroid.util import current_directory, ensure_dir
from os.path import join, exists
from os import environ
import sh


class LlamaCppPythonRecipe(PythonRecipe):
    name    = "llama_cpp_python"
    version = "0.3.9"

    # Download directly from PyPI sdist
    url = (
        "https://files.pythonhosted.org/packages/source/l/"
        "llama_cpp_python/llama_cpp_python-{version}.tar.gz"
    )

    # llama.cpp ships its own vendored cmake so we just need cmake available
    depends = ["python3", "setuptools"]
    call_hostpython_via_targetpython = False
    install_in_hostpython = False

    # Extra CMake flags for Android cross-compilation
    # Disable all accelerators that don't exist on Android
    _cmake_defines = [
        "-DLLAMA_NATIVE=OFF",
        "-DLLAMA_AVX=OFF",
        "-DLLAMA_AVX2=OFF",
        "-DLLAMA_AVX512=OFF",
        "-DLLAMA_FMA=OFF",
        "-DLLAMA_F16C=OFF",
        "-DLLAMA_METAL=OFF",
        "-DLLAMA_CUDA=OFF",
        "-DLLAMA_VULKAN=OFF",
        "-DLLAMA_HIPBLAS=OFF",
        "-DLLAMA_OPENBLAS=OFF",
        "-DLLAMA_SYCL=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DBUILD_SHARED_LIBS=ON",
    ]

    def get_recipe_env(self, arch):
        env = super().get_recipe_env(arch)
        toolchain = self.ctx.toolchain_prefix
        api       = self.ctx.ndk_api

        ndk_home  = self.ctx.ndk_dir
        tc_bin    = join(ndk_home, "toolchains", "llvm", "prebuilt",
                         "linux-x86_64", "bin")

        triple = arch.command_prefix          # e.g. aarch64-linux-android
        clang  = join(tc_bin, f"{triple}{api}-clang")
        clangpp = join(tc_bin, f"{triple}{api}-clang++")
        ar     = join(tc_bin, f"{triple}-ar")

        env["CC"]  = clang
        env["CXX"] = clangpp
        env["AR"]  = ar
        env["ANDROID_NDK"] = ndk_home

        # Tell llama-cpp-python to use our CMake chain file
        tc_file = join(
            ndk_home, "build", "cmake",
            "android.toolchain.cmake",
        )
        the_arch = "arm64-v8a" if "aarch64" in triple else "armeabi-v7a"
        cmake_args = self._cmake_defines + [
            f"-DCMAKE_TOOLCHAIN_FILE={tc_file}",
            f"-DANDROID_ABI={the_arch}",
            f"-DANDROID_PLATFORM=android-{api}",
            "-DANDROID_STL=c++_shared",
        ]
        env["CMAKE_ARGS"] = " ".join(cmake_args)

        # Force wheel build (skips cython, uses CMake)
        env["FORCE_CMAKE"]  = "1"
        env["LLAMA_NATIVE"] = "0"
        env["LLAMA_METAL"]  = "0"
        env["LLAMA_CUDA"]   = "0"
        env["LLAMA_OPENBLAS"] = "0"
        return env

    def build_arch(self, arch):
        env = self.get_recipe_env(arch)
        with current_directory(self.get_build_dir(arch.arch)):
            hostpython = self.ctx.hostpython
            shprint(
                sh.Command(hostpython),
                "setup.py",
                "build_ext",
                "--inplace",
                _env=env,
            )
            # Install into site-packages
            shprint(
                sh.Command(hostpython),
                "setup.py",
                "install",
                f"--prefix={self.ctx.get_python_install_dir(arch.arch)}",
                "--no-compile",
                _env=env,
            )


recipe = LlamaCppPythonRecipe()
