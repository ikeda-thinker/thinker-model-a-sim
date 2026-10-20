"""View thinker-TMA-F1-REFX4-U3.urdf in a browser via viser (http://localhost:8080)."""
import sys
import time
from pathlib import Path

import viser
import yourdfpy
from viser.extras import ViserUrdf

ROOT = Path(__file__).resolve().parent.parent
PKGS = ROOT / "packages"


def resolve(fname):
    if fname.startswith("file:///") and "/packages/" in fname:
        # absolute paths baked into the URDF -> this checkout's packages/
        return str(PKGS / fname.split("/packages/", 1)[1])
    if fname.startswith("file:///opt/ros/kilted/share/"):
        return str(PKGS / fname[len("file:///opt/ros/kilted/share/"):])
    if fname.startswith("package://"):
        rel = Path(fname[len("package://"):])
        p = PKGS / rel
        if not p.exists() and rel.parts[0] == "thinker_ros2_description":
            p = p.with_suffix(".stl")  # placeholder boxes
        return str(p)
    return fname


urdf = yourdfpy.URDF.load(
    ROOT / (sys.argv[1] if len(sys.argv) > 1 else "thinker-TMA-F1-REFX4-U3.urdf"),
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
        s = server.gui.add_slider(name, lo, hi, 0.001, min(max(0.0, lo), hi))
        s.on_update(lambda _: viewer.update_cfg([x.value for x in sliders]))
        sliders.append(s)
viewer.update_cfg([s.value for s in sliders])
print("open http://localhost:8080", flush=True)
while True:
    time.sleep(10)
