"""EVA toolbox: small runtime fixes for bugs in the installed filter packages (applied before any toolbox block runs;
nothing installed is edited). Each fix checks that the bug is still there, so an updated package simply skips it.
Add a fix: write a function below and list it in FIXES."""


def fix_dfttest2_typeis():
    """dfttest2 (vsdenoise.DFTTest, QTGMC's denoiser) falls back to `TypeIs = Any` on Python < 3.13 and then subscripts
    it: "TypeError: type 'Any' is not subscriptable". Give it typing_extensions.TypeIs."""
    import typing
    import typing_extensions
    import dfttest2.dfttest2 as d2
    if getattr(d2, "TypeIs", None) is typing.Any:
        d2.TypeIs = typing_extensions.TypeIs
        return True
    return False


def fix_vinverse_amnt():
    """vsdeinterlace.vinverse (and deblend, which uses it) only defines undershoot / overshoot when `amnt` is given, but
    amnt defaults to None: "UnboundLocalError: undershoot". None means "no limit", which is a clamp to the full range."""
    import inspect
    import vsdeinterlace
    import vsdeinterlace.funcs as fm
    orig = fm.vinverse
    if getattr(orig, "_eva_fixed", False):
        return False
    src = inspect.getsource(orig)
    if "if amnt is not None:" not in src or "undershoot=undershoot" not in src:
        return False                                    # fixed upstream

    def vinverse(clip, *args, **kwargs):
        sig = inspect.signature(orig).bind_partial(clip, *args, **kwargs)
        if sig.arguments.get("amnt") is None:
            kwargs["amnt"] = 255                         # 8-bit scale, scaled to the clip: the whole range = no limit
        return orig(clip, *args, **kwargs)
    vinverse._eva_fixed = True
    vinverse.__doc__, vinverse.__wrapped__ = orig.__doc__, orig
    fm.vinverse = vinverse                               # deblend() looks it up here
    vsdeinterlace.vinverse = vinverse
    # functions that took the original as a default argument when their module loaded (vsdeinterlace.deblend's decomber)
    import importlib
    import pkgutil
    for mi in pkgutil.iter_modules(vsdeinterlace.__path__, "vsdeinterlace."):
        try:
            m = importlib.import_module(mi.name)
        except Exception:                                # noqa: BLE001
            continue
        for obj in list(vars(m).values()):
            if inspect.isfunction(obj):
                if obj.__defaults__ and any(d is orig for d in obj.__defaults__):
                    obj.__defaults__ = tuple(vinverse if d is orig else d for d in obj.__defaults__)
                if obj.__kwdefaults__ and any(d is orig for d in obj.__kwdefaults__.values()):
                    obj.__kwdefaults__ = {k: (vinverse if d is orig else d) for k, d in obj.__kwdefaults__.items()}
    return True


def fix_fine_dehalo2_coord():
    """vsdehalo.fine_dehalo2 calls grow_mask(..., coord=...) but grow_mask's parameter is `coords`; the stray keyword
    reaches std.Maximum: "Maximum: Function does not take argument(s) named coord"."""
    import inspect
    import vsdehalo.mask as dm
    orig = dm.grow_mask
    if getattr(orig, "_eva_fixed", False):
        return False
    if "coord" in inspect.signature(orig).parameters:
        return False                                    # fixed upstream

    def grow_mask(*args, coord=None, **kwargs):
        if coord is not None and kwargs.get("coords") is None:
            try:
                kwargs["coords"] = [int(x) for x in coord]          # Coordinates.VERTICAL -> [0, 1, 0, 0, 0, 0, 1, 0]
            except TypeError:
                kwargs["coords"] = [int(x) for x in coord.value]
        return orig(*args, **kwargs)
    grow_mask._eva_fixed = True
    dm.grow_mask = grow_mask
    return True


FIXES = [fix_dfttest2_typeis, fix_vinverse_amnt, fix_fine_dehalo2_coord]
_DONE = {}


def apply():
    """run every fix once per process; returns {name: True (applied) / False (not needed) / error text}"""
    for f in FIXES:
        if f.__name__ in _DONE:
            continue
        try:
            _DONE[f.__name__] = f()
        except Exception as e:                           # noqa: BLE001  (package missing: nothing to fix)
            _DONE[f.__name__] = f"skipped: {type(e).__name__}: {e}"
    return dict(_DONE)
