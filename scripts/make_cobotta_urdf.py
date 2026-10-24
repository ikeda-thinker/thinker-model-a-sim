"""tma.urdf の THF・爪を COBOTTA (DENSO) と純正の電動グリッパに付け替えた cobotta.urdf を作る。

    uv run scripts/make_cobotta_urdf.py

- アーム・グリッパの寸法と質量は DENSO の denso_cobotta_ros (cobotta.urdf.xacro / parallel.urdf.xacro) の値。
  メッシュは scripts/fetch_packages.sh で packages/denso_cobotta_descriptions に取得する (GPL-2.0 なのでコミットしない)。
  元の質量・慣性は「おおよその推定値」とのこと。グリッパは仮の値 (質量 1 kg、慣性 1 kg*m^2) だったので推定で入れ直した
- 関節名は tma.urdf に合わせて thinker_arm_joint_1〜6 にする (sim.py はアームを "thinker_arm_" で探す)。
  右指は元の URDF では左指の -1 倍の mimic だが、軸を反転して Hand-E と同じ「両指とも正で開く (0=閉)」にする
- THF・爪 (left/right_thf_* 以下) は tma.urdf からそのまま持ってくる。Hand-E の指への取り付け位置を、
  Hand-E の x (開閉方向) が COBOTTA の y になるよう z 軸まわりに 90° 回して COBOTTA の指に移す。
  COBOTTA の指は全閉 (関節値 0) で内側の面が接するので、全閉が Hand-E の開度 HANDE_EQUIV_MM (爪先どうしが接する開度) と
  同じ配置になるよう、THF を片側 HANDE_EQUIV_MM だけ外へずらす。全開 (15mm) は Hand-E の 23mm 相当
- THF と指をつなぐアダプタは実物が無いので仮の板 (箱)。D405 は付けない
"""
import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tma.urdf"
OUT = ROOT / "cobotta.urdf"
MESH = "package://denso_cobotta_descriptions/cobotta_description"

HANDE_EQUIV_MM = 8.0  # COBOTTA の全閉に対応する Hand-E の片指の開度。爪先のすきま ≈ 2 x 開度 - 16mm が 0 になる所 (tma/peg.py)
STROKE = 0.015  # COBOTTA のグリッパの片指ストローク [m]
FINGER_Z = 0.06  # J6 (フランジ) から指の関節 (指先の高さ) まで [m]

# (リンク, 親から見た関節の位置, 軸, 下限, 上限, 速度 [rad/s], 質量 [kg], 重心, (半径, 高さ) で円柱近似した慣性)。denso_cobotta_ros の値
ARM = [
    ("J1", (0, 0, 0), (0, 0, 1), -2.617994, 2.617994, 0.40317105721069, 1.29, (0, 0, 0.15), (0.06, 0.1)),
    ("J2", (0, 0, 0.18), (0, 1, 0), -1.047198, 1.745329, 0.392175482923126, 0.39, (-0.02, 0.1, 0.07), (0.03, 0.12)),
    ("J3", (0, 0, 0.165), (0, 1, 0), 0.3141593, 2.443461, 0.725707902979242, 0.35, (-0.01, 0.02, 0.03), (0.03, 0.1)),
    ("J4", (-0.012, 0.02, 0.088), (0, 0, 1), -2.96706, 2.96706, 0.739146938219599, 0.35, (0, 0, 0.055), (0.03, 0.1)),
    ("J5", (0, -0.02, 0.0895), (0, 1, 0), -1.658063, 2.356194, 0.739146938219599, 0.19, (0, -0.04, 0.015), (0.03, 0.03)),
    ("J6", (0, -0.0445, 0.042), (0, 0, 1), -2.96706, 2.96706, 1.1093312725676, 0.03, (0, 0, 0), (0.03, 0.0)),
]
BASE = (1.4, (-0.02131, 0.000002, 0.044011), (0.08, 0.06))
GRIPPER_BASE_MASS = 0.25  # 推定 (元は仮の 1 kg)
FINGER_MASS = 0.02  # 推定


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def cyl_inertial(link, mass, com, rh):
    r, h = rh
    ixx = mass * r * r / 4 + mass * h * h / 12
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", rpy="0 0 0", xyz=fmt(com))
    ET.SubElement(inertial, "mass", value=f"{mass:.6g}")
    ET.SubElement(inertial, "inertia", ixx=f"{ixx:.6g}", ixy="0", ixz="0", iyy=f"{ixx:.6g}", iyz="0", izz=f"{mass * r * r / 2:.6g}")


def box_inertial(link, mass, size, com=(0, 0, 0)):
    x, y, z = size
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", rpy="0 0 0", xyz=fmt(com))
    ET.SubElement(inertial, "mass", value=f"{mass:.6g}")
    k = mass / 12
    ET.SubElement(inertial, "inertia", ixx=f"{k * (y * y + z * z):.6g}", ixy="0", ixz="0", iyy=f"{k * (x * x + z * z):.6g}", iyz="0",
                  izz=f"{k * (x * x + y * y):.6g}")


def mesh_link(robot, name, mesh, origin=(0, 0, 0), material="cobotta_white"):
    link = ET.SubElement(robot, "link", name=name)
    for tag in ("visual", "collision"):
        el = ET.SubElement(link, tag)
        ET.SubElement(el, "origin", rpy="0 0 0", xyz=fmt(origin))
        ET.SubElement(ET.SubElement(el, "geometry"), "mesh", filename=f"{MESH}/{mesh}.dae")
        if tag == "visual":
            ET.SubElement(el, "material", name=material)
    return link


def box_shape(link, size, xyz, material):
    for tag in ("visual", "collision"):
        el = ET.SubElement(link, tag)
        ET.SubElement(el, "origin", rpy="0 0 0", xyz=fmt(xyz))
        ET.SubElement(ET.SubElement(el, "geometry"), "box", size=fmt(size))
        if tag == "visual":
            ET.SubElement(el, "material", name=material)


def joint(robot, name, jtype, parent, child, xyz=(0, 0, 0), rpy=(0, 0, 0), axis=None, limit=None, mimic=None):
    j = ET.SubElement(robot, "joint", name=name, type=jtype)
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=child)
    ET.SubElement(j, "origin", rpy=fmt(rpy), xyz=fmt(xyz))
    if axis is not None:
        ET.SubElement(j, "axis", xyz=fmt(axis))
    if limit is not None:
        lo, hi, effort, vel = limit
        ET.SubElement(j, "limit", effort=f"{effort:.6g}", lower=f"{lo:.10g}", upper=f"{hi:.10g}", velocity=f"{vel:.10g}")
    if mimic is not None:
        ET.SubElement(j, "mimic", joint=mimic, multiplier="1", offset="0")
    return j


def thf_subtree(src, root_link):
    """root_link から下の THF のリンク・関節 (tma.urdf のまま)。"""
    links = {l.get("name"): l for l in src.findall("link")}
    joints = src.findall("joint")
    out_links, out_joints, todo = [links[root_link]], [], [root_link]
    while todo:
        parent = todo.pop()
        for j in joints:
            if j.find("parent").get("link") == parent:
                child = j.find("child").get("link")
                out_joints.append(j)
                out_links.append(links[child])
                todo.append(child)
    return out_links, out_joints


def main():
    src = ET.parse(SRC).getroot()
    robot = ET.Element("robot", name="thinker_cobotta")
    robot.append(ET.Comment(" Generated by scripts/make_cobotta_urdf.py from tma.urdf (THF, nails) and denso_cobotta_ros (COBOTTA, gripper). Do not edit by hand "))
    for m in src.findall("material"):
        robot.append(copy.deepcopy(m))
    ET.SubElement(ET.SubElement(robot, "material", name="cobotta_white"), "color", rgba="0.92 0.92 0.9 1")

    # ルートは慣性なしにする (KDL / robot_state_publisher がルートの慣性を扱えない)
    ET.SubElement(robot, "link", name="base_link")
    base = mesh_link(robot, "thinker_arm_base_link", "base_link")
    cyl_inertial(base, BASE[0], BASE[1], BASE[2])
    joint(robot, "thinker_arm_base_joint", "fixed", "base_link", "thinker_arm_base_link")
    parent = "thinker_arm_base_link"
    for i, (name, xyz, axis, lo, hi, vel, mass, com, rh) in enumerate(ARM, 1):
        link = mesh_link(robot, f"thinker_arm_{name}", name)
        cyl_inertial(link, mass, com, rh)
        joint(robot, f"thinker_arm_joint_{i}", "revolute", parent, f"thinker_arm_{name}", xyz=xyz, axis=axis, limit=(lo, hi, 10, vel))
        parent = f"thinker_arm_{name}"

    # 純正グリッパ。指のメッシュはグリッパ本体の座標で作られているので、指の関節 (指先の高さ) から FINGER_Z 下げて置く
    gb = mesh_link(robot, "cobotta_gripper_base", "gripper_base")
    box_inertial(gb, GRIPPER_BASE_MASS, (0.055, 0.061, 0.035), com=(0, 0, 0.018))
    joint(robot, "cobotta_gripper_base_joint", "fixed", "thinker_arm_J6", "cobotta_gripper_base")
    # 仮のアダプタ: 指の外側の面 (y=13.9mm) から THF のタブの内側の面まで。幅は指と同じ、指の側面から THF のタブの高さまで
    adapter_y = (0.0139, 0.0229)
    adapter_z = (-0.020, 0.016)
    adapter_size = (0.0312, adapter_y[1] - adapter_y[0], adapter_z[1] - adapter_z[0])
    for side, sign in (("left", 1.0), ("right", -1.0)):
        finger = mesh_link(robot, f"cobotta_gripper_{side}_finger", f"{side}_finger", origin=(0, 0, -FINGER_Z), material="cobotta_white")
        center = (0.0, sign * (adapter_y[0] + adapter_y[1]) / 2, (adapter_z[0] + adapter_z[1]) / 2)
        box_shape(finger, adapter_size, center, "thinker_black")
        box_inertial(finger, FINGER_MASS, adapter_size, com=center)
        joint(
            robot, f"cobotta_gripper_{side}_finger_joint", "prismatic", "cobotta_gripper_base", f"cobotta_gripper_{side}_finger",
            xyz=(0, 0, FINGER_Z), axis=(0, sign, 0), limit=(0.0, STROKE, 20, 0.04),
            mimic="cobotta_gripper_left_finger_joint" if side == "right" else None,
        )

    # THF。Hand-E の指から見た THF の base の位置 (tma.urdf) を z まわりに 90° 回し (Hand-E の x → COBOTTA の y)、外へ HANDE_EQUIV_MM ずらす
    hande_joints = {j.get("name"): j for j in src.findall("joint")}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        j = hande_joints[f"{side}_thf_base_joint"]
        o = j.find("origin")
        x, y, z = map(float, o.get("xyz").split())
        roll, pitch, yaw = map(float, o.get("rpy").split())
        x += sign * HANDE_EQUIV_MM * 1e-3
        links, joints = thf_subtree(src, f"{side}_thf_base_link")
        for l in links:
            robot.append(copy.deepcopy(l))
        joint(robot, f"{side}_thf_base_joint", "fixed", f"cobotta_gripper_{side}_finger", f"{side}_thf_base_link",
              xyz=(-y, x, z), rpy=(roll, pitch, yaw + math.pi / 2))
        for jj in joints:
            robot.append(copy.deepcopy(jj))

    ET.indent(robot, space="  ")
    OUT.write_text('<?xml version="1.0" ?>\n' + ET.tostring(robot, encoding="unicode") + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
