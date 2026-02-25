[app]
# Application name and package
title          = Offline RAG
package.name   = offlinerag
package.domain = com.yourname

source.dir    = .
source.include_exts = py,png,jpg,kv,atlas,db,gguf
source.exclude_dirs = __pycache__,.git,.venv,venv,.mypy_cache,quantize,llamacpp_bin

version = 1.0.0

# Python dependencies that Buildozer will install via p4a
requirements =
    python3,
    kivy==2.3.0,
    hostpython3,
    setuptools,
    pip,
    liblzma,
    sqlite3,
    openssl,
    requests,
    certifi,
    idna,
    charset-normalizer,
    urllib3,
    pymupdf==1.24.11,
    llama_cpp_python

# Point buildozer to our custom p4a recipes folder
p4a.local_recipes = ./p4a-recipes

# Android bootstrap (SDL2 is the default Kivy bootstrap)
p4a.bootstrap = sdl2

# Orientation: portrait only (saves memory vs landscape switching)
orientation = portrait

# Android-specific
android.permissions =
    READ_EXTERNAL_STORAGE,
    WRITE_EXTERNAL_STORAGE,
    MANAGE_EXTERNAL_STORAGE

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
