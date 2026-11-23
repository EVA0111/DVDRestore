"""EVA toolbox recipe template - copy this file to a new name (no leading underscore) and it appears in the toolbox
under "My blocks". Files starting with "_" are ignored.

BLOCK  name, category, description and settings (kinds: choice, int, float, bool, text, code - like the app's blocks)
run()  gets the clip, the block's settings (defaults filled in), the job (episode settings) and returns a clip.
The Script step writes  clip = dr.block(clip, job, 'recipe:<file name>', {settings}, ctx)  - the same call as the chain.
vapoursynth (vs, core) and vs-jetpack can be imported as usual."""
import vapoursynth as vs

core = vs.core

BLOCK = {
    "name": "Example: soft denoise + sharpen",
    "cat": "My blocks",
    "desc": "A light vs-jetpack denoise followed by a faithful sharpen - a starting point for your own block.",
    "params": [
        {"key": "sigma", "label": "Denoise strength", "kind": "float", "default": 2.0, "min": 0.5, "max": 16, "step": 0.5},
        {"key": "sharp", "label": "Sharpen", "kind": "float", "default": 0.4, "min": 0, "max": 2, "step": 0.1},
    ],
}


def run(clip, p, job):
    from vsdenoise import DFTTest
    clip = DFTTest().denoise(clip, float(p["sigma"]), tr=1)
    import vsgraph
    return vsgraph.sharpen(clip, dict(job, sharp={"luma": float(p["sharp"]), "chroma": 0.0, "mode": "faithful", "radius": "fine"}))
