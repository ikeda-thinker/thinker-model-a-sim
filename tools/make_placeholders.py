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
    # Hand-E の上面から 1mm 出ている指スライダーの凸部だけ (寸法は robotiq_hande_description の finger.dae から)。
    # THF はこれにアダプタ (fingers/thinker-TF1/meshes/*/hande-adapter.stl) で付く
    "grippers/robotiq-hande/meshes/visual/finger-slider.stl": ((0.0419, 0.006, 0.001), (-0.0113, -0.0142, 0.0005)),
}

# 爪は根元 (z=0) から URDF の tip_link (0, 0.0153, 0.034) へ斜めに細くなる柱。
# name -> ((根元 size xy, 中心 xy), (先端 size xy, 中心 xy), 長さ z)
nails = {
    "nails/thinker-TF1/meshes/visual/default-nail.stl": (((0.01, 0.008), (0, 0)), ((0.006, 0.003), (0, 0.0153)), 0.034),
    "nails/thinker-TF1/meshes/collision/default-nail.stl": (((0.01, 0.008), (0, 0)), ((0.006, 0.003), (0, 0.0153)), 0.034),
}


def box(size, c):
    sx, sy, sz = (s / 2 for s in size)
    v = [(c[0] + x * sx, c[1] + y * sy, c[2] + z * sz) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    return stl(v)


def prism(bottom, top, length):
    """z=0 の矩形 bottom と z=length の矩形 top を結ぶ六面体。"""
    (bs, bc), (ts, tc) = bottom, top
    v = []
    for x in (-1, 1):
        for y in (-1, 1):
            for z, (s, c) in ((0.0, (bs, bc)), (length, (ts, tc))):
                v.append((c[0] + x * s[0] / 2, c[1] + y * s[1] / 2, z))
    return stl(v)


def stl(v):
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

for rel, (bottom, top, length) in nails.items():
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(prism(bottom, top, length))
    print(p)
