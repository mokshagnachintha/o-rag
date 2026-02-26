[app]
# Application name and package
title          = O-RAG
package.name   = orag
package.domain = com.yourname

source.dir    = .
source.include_exts = py,png,jpg,kv,atlas,db
# Include the pre-built ARM64 llama-server binary (no extension)
source.include_patterns = llama-server-arm64
source.exclude_dirs = __pycache__,.git,.venv,venv,.mypy_cache,quantization,bin,p4a-recipes,llama_cpp_src,build_log_dl,build_log_dl2,llamacpp_bin,quantize

version = 1.0.0

# Python dependencies
# Pure-Python packages (certifi, requests, etc.) are pip-installed by p4a.
# Only packages that need C compilation are listed as recipes.
requirements =
    python3,
    kivy==2.3.0,
    hostpython3,
    setuptools,
    sqlite3,
    openssl,
    requests,
    certifi,
    idna,
    charset-normalizer,
    urllib3,
    pypdf,
    plyer,
    huggingface-hub,
    tqdm,
    filelock,
    packaging,
    fsspec

# Android bootstrap
p4a.bootstrap = sdl2

# Orientation: portrait only
orientation = portrait

# Android-specific
android.permissions =
    READ_EXTERNAL_STORAGE,
    WRITE_EXTERNAL_STORAGE,
    MANAGE_EXTERNAL_STORAGE,
    READ_MEDIA_IMAGES,
    READ_MEDIA_VIDEO,
    INTERNET

android.api         = 34
android.minapi      = 26
android.ndk         = 25b
android.ndk_api     = 26
android.sdk         = 34
android.archs       = arm64-v8a
android.allow_backup = False

# Extra Java options for model loading
android.add_jvm_options = -Xmx768m

# App icon
icon.filename      = %(source.dir)s/assets/icon.png
presplash.filename = %(source.dir)s/assets/presplash.png

[buildozer]
log_level = 2
warn_on_root = 1
