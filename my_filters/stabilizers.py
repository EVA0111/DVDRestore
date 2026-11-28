"""Stabilizers for EVA toolkit (filters: Your own filters -> stabilizers). Built on MVTools' Depan functions (global motion
estimate from the picture, then shifting / zooming / rotating each frame back). Run them before crop and resize, so edges
that move in do not clip."""
import vapoursynth as vs

core = vs.core


def stab(clip, dxmax=4, dymax=4, mirror=0):
    """Stab (from AviSynth, as in havsfunc): removes small, fast jitter and film-gate weave of old transfers. dxmax / dymax = the
    largest shift in pixels that is corrected; mirror 0-15 fills the moved-in border with mirrored picture (1 top, 2 bottom,
    4 left, 8 right, add them: 15 = all sides) instead of black."""
    avg = core.std.AverageFrames(clip, [1, 1, 1])                           # a steadier reference (TemporalSoften radius 1)
    ref = core.rgvs.Repair(clip, avg, mode=[1])
    inter = core.std.Interleave([ref, clip])
    data = core.mv.DepanEstimate(inter, trust=0, dxmax=int(dxmax), dymax=int(dymax))
    out = core.mv.DepanCompensate(inter, data=data, offset=-1, mirror=int(mirror))
    return core.std.SelectEvery(out, cycle=2, offsets=0)


def steady_camera(clip, cutoff=1.0, damping=0.9, dxmax=60, dymax=30, zoommax=1.05, mirror=15, trust=4):
    """Smooths a shaky camera (handheld shots, wobble) with DepanEstimate + DepanStabilise: the slow camera path is kept
    (pans stay), the fast shake goes. cutoff = lowest frequency (Hz) still corrected: lower = smoother, 0.5-2.
    damping 0-2 (higher = settles faster). dxmax / dymax / zoommax = the most shift (px) and zoom that is corrected. mirror 15 fills moved-in borders from the picture."""
    data = core.mv.DepanEstimate(clip, trust=float(trust), dxmax=int(dxmax), dymax=int(dymax), zoommax=float(zoommax))
    return core.mv.DepanStabilise(clip, data=data, cutoff=float(cutoff), damping=float(damping), dxmax=int(dxmax), dymax=int(dymax),
                                  zoommax=float(zoommax), mirror=int(mirror))


def jitter_only(clip, dxmax=3, dymax=3, mirror=15, passes=2):
    """Stab applied `passes` times (1-3): for heavy jitter that one pass leaves partly in."""
    for _ in range(max(1, min(3, int(passes)))):
        clip = stab(clip, dxmax, dymax, mirror)
    return clip
