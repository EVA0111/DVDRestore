"""Every folder DVD Restore uses, in one place. The engine lives in core/, the project root is one level up:
    app/       Electron front-end            models/    network weights (+ the network code some of them need)
    core/      this engine (Python)          engines/   TensorRT engines, built once per model and size (cache)
    runtime/   portable version only: python/, ffmpeg/, mkvtoolnix/   temp/  work folders (cache)
Bundled tools in runtime/ are used first, so a portable copy runs without anything installed."""
import os

CORE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CORE)
MODELS = os.path.join(ROOT, "models")
ENGINES = os.path.join(ROOT, "engines")
TEMP = os.path.join(ROOT, "temp")
RUNTIME = os.path.join(ROOT, "runtime")
SETTINGS = os.path.join(ROOT, "settings.json")
MY_FILTERS = os.path.join(ROOT, "my_filters")      # your own filters: *.py functions, plugins\*.dll


def model(name):
    """path of a weights file in models/"""
    return os.path.join(MODELS, name)


def tool(*parts):
    """path of a bundled tool in runtime/ (may not exist: then the installed one is used)"""
    return os.path.join(RUNTIME, *parts)
