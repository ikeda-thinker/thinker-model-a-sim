"""Generate box-shaped placeholder STL meshes for thinker_ros2_description."""
import sys
from pathlib import Path

root = Path(sys.argv[1])
# name -> (size xyz, center xyz) in metres, rough guesses
parts = {
    "fingers/thinker-TF1/meshes/visual/base.stl": ((0.03, 0.03, 0.037), (0, 0, 0.0187)),
    "fingers/thinker-TF1/meshes/collision/base.stl": ((0.03, 0.03, 0.037), (0, 0, 0.0187)),
    "fingers/thinker-TF1/meshes/visual/x.stl": ((0.012, 0.012, 0.012), (0, 0, 0)),
    "fingers/thinker-TF1/meshes/visual/y.stl": ((0.012, 0.012, 0.012), (0, 0, 0)),
    "fingers/thinker-TF1/meshes/visual/z.stl": ((0.008, 0.008, 0.0276), (0, 0, 0.0138)),
    "nails/thinker-TF1/meshes/visual/default-nail.stl": ((0.01, 0.008, 0.012), (0, 0, 0.006)),
    "nails/thinker-TF1/meshes/collision/default-nail.stl": ((0.01, 0.008, 0.012), (0, 0, 0.006)),
    "grippers/robotiq-hande/meshes/visual/finger-slider.stl": ((0.01, 0.02, 0.04), (0, 0, 0.02)),
}

def box(size, c):
    sx, sy, sz = (s / 2 for s in size)
    v = [(c[0] + x * sx, c[1] + y * sy, c[2] + z * sz) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    out = ["solid box"]
    for a, b, c_, d in quads:
        for tri in ((a, b, c_), (a, c_, d)):
            out.append("facet normal 0 0 0\nouter loop")
            out += ["vertex %g %g %g" % v[i] for i in tri]
            out.append("endloop\nendfacet")
    out.append("endsolid box")
    return "\n".join(out) + "\n"

for rel, (size, center) in parts.items():
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(box(size, center))
    print(p)
