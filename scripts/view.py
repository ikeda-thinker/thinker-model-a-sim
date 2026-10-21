"""View tma.urdf in a browser via viser (http://localhost:8080)."""
import sys
import time
from pathlib import Path

import trimesh
import viser
import yourdfpy
from viser.extras import ViserUrdf

ROOT = Path(__file__).resolve().parent.parent
PKGS = ROOT / "packages"

def apply_visual_color(geom, visual, material_map):
    """URDF の色を PBR の材質にして付ける。
    yourdfpy は単色のメッシュ (STL) にしか色を塗らず、材質付きで読まれる dae (Hand-E) は元の色のまま出る。
    単色で塗るとツヤがなく黒い部品がのっぺりするので、名前に metal を含む色は金属、glass を含む色はツヤツヤ、それ以外は半ツヤにする"""
    mat = visual.material
    if mat is None:
        return
    color = mat.color or (material_map[mat.name].color if mat.name in material_map else None)
    if color is None:
        return
    metal = "metal" in (mat.name or "")
    # glTF の baseColorFactor はリニアなので、URDF の色 (見た目の sRGB) をそのまま渡すと白っぽくなる。sRGB → リニアに直す
    rgb = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in color.rgba[:3]]
    geom.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[float(c) for c in rgb] + [float(color.rgba[3])],
            metallicFactor=1.0 if metal else 0.0,
            roughnessFactor=0.3 if metal else 0.05 if "glass" in (mat.name or "") else 0.45,
        )
    )


yourdfpy.urdf.apply_visual_color = apply_visual_color


def resolve(fname):
    if fname.startswith("file:///") and "/packages/" in fname:
        # absolute paths baked into the URDF -> this checkout's packages/
        return str(PKGS / fname.split("/packages/", 1)[1])
    if fname.startswith("file:///opt/ros/kilted/share/"):
        return str(PKGS / fname[len("file:///opt/ros/kilted/share/"):])
    if fname.startswith("package://"):
        rel = Path(fname[len("package://"):])
        p = PKGS / rel
        if not p.exists() and rel.parts[0] == "think_hand_f_description":
            p = p.with_suffix(".stl")  # placeholder boxes
        return str(p)
    return fname


urdf = yourdfpy.URDF.load(
    ROOT / (sys.argv[1] if len(sys.argv) > 1 else "tma.urdf"),
    filename_handler=resolve,
    build_collision_scene_graph=False,
    load_collision_meshes=False,
)
server = viser.ViserServer(host="0.0.0.0", port=8080)
viewer = ViserUrdf(server, urdf, root_node_name="/robot")
with server.gui.add_folder("Joints"):
    sliders = []
    for name, (lo, hi) in viewer.get_actuated_joint_limits().items():
        lo = -3.14159 if lo is None else lo
        hi = 3.14159 if hi is None else hi
        s = server.gui.add_slider(name, lo, hi, 0.0001 if name.endswith("_z_joint") else 0.001, min(max(0.0, lo), hi))
        s.on_update(lambda _: viewer.update_cfg([x.value for x in sliders]))
        sliders.append(s)
viewer.update_cfg([s.value for s in sliders])
print("open http://localhost:8080", flush=True)
while True:
    time.sleep(10)
