"""EVA toolbox: the library of the Chain step, and how it is kept up to date / extended.

  catalog.json   curation of the plugin and vs-jetpack functions (hide with a reason, defaults, required settings,
                 input-format adapters, notes) - plain JSON, edit freely
  shims.py       runtime fixes for bugs in installed filter packages (applied before toolbox blocks run)
  recipes/*.py   your own blocks (BLOCK + run(clip, p, job)); "Save as block" on a Custom code block writes one
Updating = upgrading the pip packages the filters come from (vsjetpack, vapoursynth-*, vs-*) with the app's own Python,
then the compatibility scan (blocks_scan.py) - see packages(), update_command(), install_command()."""
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(HERE, "catalog.json")
RECIPES = os.path.join(HERE, "recipes")
_CAT = {"mtime": None, "data": None}


def catalog():
    """catalog.json (re-read when it changes)"""
    try:
        m = os.path.getmtime(CATALOG)
        if _CAT["mtime"] != m:
            _CAT["data"], _CAT["mtime"] = json.load(open(CATALOG, encoding="utf-8")), m
    except Exception as e:                               # noqa: BLE001
        _CAT["data"] = {"_error": f"catalog.json: {e}"}
    d = _CAT["data"] or {}
    for k in ("hide", "defaults", "needs", "adapt", "notes"):
        d.setdefault(k, {})
    return d


# ---------------------------------------------------------------- recipes (your own blocks)
def recipes():
    """{stem: module or error text} of recipes/*.py (files starting with "_" are templates)"""
    out = {}
    if not os.path.isdir(RECIPES):
        return out
    for fn in sorted(os.listdir(RECIPES)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        stem = fn[:-3]
        try:
            spec = importlib.util.spec_from_file_location(f"eva_recipe_{stem}", os.path.join(RECIPES, fn))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not isinstance(getattr(mod, "BLOCK", None), dict) or not callable(getattr(mod, "run", None)):
                raise ValueError("needs a BLOCK dict and a run(clip, p, job) function")
            out[stem] = mod
        except Exception as e:                           # noqa: BLE001
            out[stem] = f"{type(e).__name__}: {e}"
    return out


def safe_stem(name):
    s = re.sub(r"[^a-z0-9_]+", "_", (name or "").strip().lower()).strip("_")
    return s[:48] or "my_block"


def save_recipe(name, code, desc=""):
    """a Custom code block as a toolbox block: recipes/<name>.py (fixed code, no settings). Returns the file name."""
    os.makedirs(RECIPES, exist_ok=True)
    stem = safe_stem(name)
    path = os.path.join(RECIPES, stem + ".py")
    k = 2
    while os.path.exists(path):
        path = os.path.join(RECIPES, f"{stem}_{k}.py"); k += 1
    body = "\n".join("    " + l if l.strip() else "" for l in (code or "clip = clip").splitlines())
    text = (f'"""EVA toolbox recipe saved from a Custom code block. Edit freely (see _template.py for settings)."""\n'
            "import vapoursynth as vs\n\ncore = vs.core\n\n"
            f"BLOCK = {{\"name\": {json.dumps(name or stem)}, \"cat\": \"My blocks\", \"desc\": {json.dumps(desc or 'Saved from a Custom code block.')}, \"params\": []}}\n\n\n"
            "def run(clip, p, job):\n"
            "    import vsgraph as dr                                  # the app's stages, as in Custom code\n"
            "    ns = {}\n"
            "    for m in (\"vstools\", \"vsdenoise\", \"vsaa\", \"vsdehalo\", \"vsdeinterlace\", \"vsscale\", \"vsmasktools\", \"vsrgtools\",\n"
            "              \"vsdeband\", \"vskernels\", \"vsexprtools\"):\n"
            "        try:\n"
            "            ns[m] = __import__(m)\n"
            "        except Exception:                                 # noqa: BLE001\n"
            "            pass\n"
            "    globals().update(ns)\n"
            f"{body}\n"
            "    return clip\n")
    compile(text, path, "exec")                          # refuse code that does not even parse
    open(path, "w", encoding="utf-8").write(text)
    return os.path.basename(path)


# ---------------------------------------------------------------- updates
def packages():
    """the pip packages the filters come from: [(name, version)]"""
    from importlib.metadata import distributions
    out = {}
    for d in distributions():
        n = (d.metadata.get("Name") or "").strip()
        ln = n.lower()
        if ln.startswith(("vapoursynth-", "vs-", "vs_")) or ln in ("vsjetpack", "vapoursynth", "dfttest2", "vsjetengine", "jetpytools"):
            out[n] = d.version
    return sorted(out.items(), key=lambda x: x[0].lower())


def update_command():
    """upgrade every filter package (VapourSynth itself is left as it is: the app is tested with this version)"""
    names = [n for n, _ in packages() if n.lower() != "vapoursynth"]
    return [sys.executable, "-m", "pip", "install", "--upgrade", "--upgrade-strategy", "only-if-needed", *names]


def install_command(name):
    n = (name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._\-\[\]=<>~!]*", n):
        raise ValueError(f"not a package name: {name!r}")
    return [sys.executable, "-m", "pip", "install", "--upgrade", n]


# ---------------------------------------------------------------- groups and plain names (names.json)
NAMES = os.path.join(HERE, "names.json")
_NAMES = {"mtime": None, "data": None}


def names():
    """names.json (re-read when it changes): groups, plain names, family rules, namespace fallback"""
    try:
        m = os.path.getmtime(NAMES)
        if _NAMES["mtime"] != m:
            _NAMES["data"], _NAMES["mtime"] = json.load(open(NAMES, encoding="utf-8")), m
    except Exception:                                    # noqa: BLE001
        _NAMES["data"] = {}
    d = _NAMES["data"] or {}
    for k, v in (("groups", []), ("names", {}), ("rules", []), ("fallback", {})):
        d.setdefault(k, v)
    return d


def _human(fn):
    """function name -> words: fine_dehalo2 -> Fine dehalo2, DegrainMedian -> Degrain median"""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z][a-z])", " ", fn.replace("_", " ")).strip()
    return (s[:1].upper() + s[1:]) if s else fn


def label(call):
    """(group key, plain name) of a plugin / vs-jetpack function"""
    d = names()
    if call in d["names"]:
        g, n = d["names"][call]
        return g, n
    ns, _, fn = call.rpartition(".")
    for rx, g, tpl in d["rules"]:
        m = re.match(rx, call)
        if m:
            return g, tpl.replace("{ns}", {"mv": "MVTools", "mvu": "MVUtensils"}.get(ns, ns)).format(None, *[x[:1].upper() + x[1:] for x in m.groups()])
    return d["fallback"].get(ns, "adv"), f"{_human(fn)} ({ns})"
