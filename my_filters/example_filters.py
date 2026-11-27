"""Example of your own filters for EVA toolkit (Repair -> + VapourSynth filters -> + Add filter -> Your own filters).
Every function here whose first argument is the clip becomes a filter: its other arguments are its settings (with
their defaults), its docstring is the help shown in the window. Edit this file or add your own .py files next to it;
plugins (.dll) go into the plugins folder. Settings are typed as Python values (2, 0.5, True, 'text')."""
import vapoursynth as vs

core = vs.core


def soft_sharpen(clip, amount=0.5, radius=1):
    """Gentle unsharp mask on the brightness only: amount 0-2, radius 1-3."""
    blur = core.std.BoxBlur(clip, planes=[0], hradius=radius, vradius=radius)
    return core.std.Expr([clip, blur], [f"x x y - {amount} * +", "", ""])


def lift_shadows(clip, lift=8):
    """Brightens the darkest tones by up to `lift` levels (8-bit scale), leaves the rest alone."""
    peak = (1 << clip.format.bits_per_sample) - 1 if clip.format.sample_type == vs.INTEGER else 1
    k = lift * peak / 255
    return core.std.Expr(clip, [f"x {k} 1 x {peak} / - 4 pow * +", "", ""])
