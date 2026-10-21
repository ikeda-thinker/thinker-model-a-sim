"""Fusion 360 から書き出した STL (mm) を URDF 用のメッシュ (m) に直す。

    uv run tools/convert_meshes.py

- 爪 (packages/pile-nail-v1.stl): y が長さ方向 (根元の面 y=0 〜 先端 45mm)、z が内側 (相手の指) へのずれ、x が幅。
  根元に THF の板 (3x7mm) を差す穴 (8.4x5.4mm、深さ 8mm) がある。
  nail_link の座標 (z が長さ方向で根元の面が原点、y が内側、x が幅) に回して書き出す。
  曲がった形の凸包だと内側が埋まるので、当たり判定は長さ方向に切った区間ごとの凸包にする。
- THF (packages/think-hand-f.stl): 1 指ぶんのユニット。z が指の軸 (上端の板に爪を差す)、x が内側。
  座標はそのまま (mm → m だけ)。URDF の base_link の visual / collision の origin で位置を合わせる。
  見た目は外から見える部品だけ残して軽くし、当たり判定は本体の外形の凸包にする。
  上端の穴 (筐体の上端 z=16mm) の中の部品は凸包から引っ込んでいるが、穴から見えるので残す。これらはセンサの可動部で、見た目を別にして URDF の可動リンクに付ける
  (座標は THF と同じまま。各リンクの visual の origin で位置を合わせる)。
  - thf-gimbal (y_link: x/y に傾くだけ): 穴の中の球殻 (外面が回転中心 z≈0.3mm まわりの半径 17.7mm の球面で、筐体の縁に接するので沈めない)、
    その内側のカラー、ピン穴が回転中心にあるジンバルの内輪
  - thf-tip (z_link: 傾いて押し込みで沈む): 球殻より上に出ている軸 (上端の板に爪を差す)・スリーブ・フランジ
  実機は筐体が黒でねじ・タブ・先端の可動部が金属の地色なので、本体のうち小さい部品 (ねじ・ピン・タブ) は thf-metal に分けて URDF で色を変える。
  一番下のタブ (厚さ 6 x 幅 20mm、x=-3〜3, z=-41.3〜-26.8) に x 方向の M3 穴が 2 個 (y=±7, z=-37.3) あり、ここをアダプタに留める。
- Hand-E と THF のアダプタ (packages/thf-hande-adapter-v3.stl): 厚さ 8.8mm の板 (z=-4.4〜4.4)。
  z=+4.4 の面に Hand-E の指スライダーの凸部 (幅 6, 高さ 1mm) がはまる溝 (y=14.25 中心) があり、
  M3 の座ぐり穴 2 個 (x=3, 12, y=14.25、頭は z=-4.4 側) で Hand-E の指取付ねじ穴に留める。
  x=+22 の面に THF のタブが入るポケット (深さ 1mm、底 x=21) があり、y=±7, z=-1.1 のインサートに THF を留める。
  Hand-E の指フレーム (robotiq_hande_description の finger.dae と同じ。ねじ穴 x=-2.98, 6.02, y=-14.23、本体の上面 z=0) に回して書き出す。
- D405 (packages/realsense2_description/meshes/d405.stl): 1 つにつながったメッシュで、z=0 が前面 (奥行き 23mm)。
  実機はアルミの筐体の前面が黒いガラスなので、前を向いた前面付近の面を d405-front、残りを d405-body に分けて URDF で色を変える。座標・単位 (mm) はそのまま。
"""
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "packages/think_hand_f_description"

NAIL_SRC = ROOT / "packages/pile-nail-v1.stl"
NAIL_OUT = PKG / "nails/thinker-TF1/meshes"
NAIL_NAME = "pile-nail"
NAIL_CUTS_MM = (0.0, 9.0, 13.5, 18.0, 22.5, 27.0)  # 当たり判定の区間の区切り (根元の面からの長さ)。最後の区切りから滑り止めの手前までが刃
NAIL_PAD_MM = 10.0  # 先端の滑り止め (ゴム) の長さ。この区間は別の凸包 (pile-nail-pad) にして sim.py で摩擦を変える

HULL_GRID = 0.5e-3  # 当たり判定の凸包を作る前に点を丸める格子 [m]

THF_SRC = ROOT / "packages/think-hand-f.stl"
THF_OUT = PKG / "fingers/thinker-TF1/meshes"
THF_NAME = "thf"
THF_VISIBLE_MM = 2.0  # 全体の凸包からこの距離以内に頂点がある部品を外から見える部品とみなす
THF_BODY_Z_MM = (-31.0, 21.0)  # 当たり判定に使う本体の高さ範囲 (下のブラケット・ケーブル、上の板は除く)
THF_OPENING_MM = (13.0, 12.0)  # 半径がこれ以内で上端がこの高さより上の部品を、上端の穴から見える部品として残す [mm]
THF_TIP_Z_MM = 22.0  # 穴の中の部品のうち、上端がこれより上のものを thf-tip、残りを thf-gimbal にする。球殻の上端は 20.6mm
THF_METAL_MM = 30.0  # 外から見える部品のうち、一番長い辺がこれより短いもの (ねじ・ピン・タブ) を金属の部品 thf-metal にする。黒い筐体・カバーは 31mm 以上

D405_SRC = ROOT / "packages/realsense2_description/meshes/d405.stl"
D405_FRONT_MM = -1.5  # 中心がこれより前 (z が大きい) で前を向いた面を前面のガラスにする。前面の板は z=-1〜0
D405_FRONT_NZ = 0.5  # 前を向いているとみなす法線の z 成分の下限

ADAPTER_SRC = ROOT / "packages/thf-hande-adapter-v3.stl"
ADAPTER_OUT = THF_OUT
ADAPTER_NAME = "hande-adapter"
# STL → Hand-E の左指フレーム。x 軸まわりに 180° 回して溝の面を下 (z=0) に、座ぐり穴を Hand-E のねじ穴に合わせる
ADAPTER_SHIFT_MM = (-6.0, 0.0, 4.4)


def nail_to_link(m):
    """STL (mm) → nail_link (m)。(x, y, z) → (-x, z, y)。行列式 +1 の回転なので面の向きは保たれる"""
    T = np.eye(4)
    T[:3, :3] = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    m = m.copy()
    m.apply_transform(T)
    m.apply_scale(1e-3)
    return m


def coarse_hull(pts, grid):
    """格子 grid に丸めて点を減らしてから凸包を作る。凸包の接触計算は頂点数に比例して重くなるので"""
    return trimesh.convex.convex_hull(np.unique(np.round(pts / grid), axis=0) * grid)


def hull_between(m, z0, z1):
    """z0〜z1 の区間にある頂点と、両端の断面の点の凸包"""
    pts = [m.vertices[(m.vertices[:, 2] >= z0) & (m.vertices[:, 2] <= z1)]]
    for z in (z0, z1):
        s = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if s is not None:
            pts.append(s.vertices)
    return coarse_hull(np.concatenate(pts), HULL_GRID)


def convert_nail():
    m = nail_to_link(trimesh.load(NAIL_SRC))
    (NAIL_OUT / "visual").mkdir(parents=True, exist_ok=True)
    (NAIL_OUT / "collision").mkdir(parents=True, exist_ok=True)
    m.export(NAIL_OUT / "visual" / f"{NAIL_NAME}.stl")
    zmax = m.bounds[1, 2]
    cuts = [c * 1e-3 for c in NAIL_CUTS_MM] + [zmax - NAIL_PAD_MM * 1e-3, zmax]
    names = [f"{NAIL_NAME}-{i}" for i in range(len(cuts) - 2)] + [f"{NAIL_NAME}-pad"]
    for name, z0, z1 in zip(names, cuts[:-1], cuts[1:]):
        h = hull_between(m, z0, z1)
        h.export(NAIL_OUT / "collision" / f"{name}.stl")
        print(f"爪 collision {name}: z {z0 * 1e3:5.1f}〜{z1 * 1e3:5.1f}mm  y {h.bounds[0, 1] * 1e3:5.1f}〜{h.bounds[1, 1] * 1e3:5.1f}mm")
    tip = m.vertices[m.vertices[:, 2] > zmax - 1e-4]
    print(f"爪の先端 z={zmax * 1e3:.1f}mm, 先端の中心 y={tip[:, 1].mean() * 1e3:.1f}mm (URDF の tip_joint に使う)")


def convert_thf():
    m = trimesh.load(THF_SRC)
    parts = m.split(only_watertight=False)
    hull = m.convex_hull
    opening = [p for p in parts if np.hypot(p.vertices[:, 0], p.vertices[:, 1]).max() < THF_OPENING_MM[0] and p.bounds[1, 2] > THF_OPENING_MM[1]]
    ids = {id(p) for p in opening}
    visible = [
        p for p in parts
        if id(p) not in ids and np.abs(hull.nearest.signed_distance(p.vertices[:: max(1, len(p.vertices) // 50)])).min() < THF_VISIBLE_MM
    ]
    tip = [p for p in opening if p.bounds[1, 2] > THF_TIP_Z_MM]
    gimbal = [p for p in opening if p.bounds[1, 2] <= THF_TIP_Z_MM]
    # 面が 1 枚だけの切れ端は大きさで分けると金属に入るが、筐体の一部なので黒い側に残す
    metal = [p for p in visible if len(p.faces) > 1 and np.ptp(p.vertices, axis=0).max() < THF_METAL_MM]
    ids = {id(p) for p in metal}
    visible = [p for p in visible if id(p) not in ids]
    vis, vis_metal, vis_gimbal, vis_tip = (trimesh.util.concatenate(ps) for ps in (visible, metal, gimbal, tip))
    for v in (vis, vis_metal, vis_gimbal, vis_tip):
        v.apply_scale(1e-3)
    body = m.vertices[(m.vertices[:, 2] >= THF_BODY_Z_MM[0]) & (m.vertices[:, 2] <= THF_BODY_Z_MM[1])]
    col = coarse_hull(body * 1e-3, 2 * HULL_GRID)
    (THF_OUT / "visual").mkdir(parents=True, exist_ok=True)
    (THF_OUT / "collision").mkdir(parents=True, exist_ok=True)
    vis.export(THF_OUT / "visual" / f"{THF_NAME}.stl")
    vis_metal.export(THF_OUT / "visual" / f"{THF_NAME}-metal.stl")
    vis_gimbal.export(THF_OUT / "visual" / f"{THF_NAME}-gimbal.stl")
    vis_tip.export(THF_OUT / "visual" / f"{THF_NAME}-tip.stl")
    col.export(THF_OUT / "collision" / f"{THF_NAME}.stl")
    print(f"THF visual: {len(parts)} 部品中 本体 {len(visible)} / metal {len(metal)} / gimbal {len(gimbal)} / tip {len(tip)} 部品, "
          f"{len(vis.faces)} / {len(vis_metal.faces)} / {len(vis_gimbal.faces)} / {len(vis_tip.faces)} 面 / collision: {len(col.faces)} 面")


def convert_adapter():
    """見た目は本体だけ (中のインサート 2 個は外から見えないので除く)、当たり判定は本体の凸包"""
    m = trimesh.load(ADAPTER_SRC)
    body = max(m.split(only_watertight=False), key=lambda p: len(p.faces))
    T = np.eye(4)
    T[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    T[:3, 3] = ADAPTER_SHIFT_MM
    body.apply_transform(T)
    body.apply_scale(1e-3)
    col = coarse_hull(body.vertices, HULL_GRID)
    body.export(ADAPTER_OUT / "visual" / f"{ADAPTER_NAME}.stl")
    col.export(ADAPTER_OUT / "collision" / f"{ADAPTER_NAME}.stl")
    lo, hi = body.bounds * 1e3
    print(f"アダプタ: x {lo[0]:.1f}〜{hi[0]:.1f}, y {lo[1]:.1f}〜{hi[1]:.1f}, z {lo[2]:.1f}〜{hi[2]:.1f}mm / collision: {len(col.faces)} 面")


def convert_d405():
    m = trimesh.load(D405_SRC)
    front = (m.triangles_center[:, 2] > D405_FRONT_MM) & (m.face_normals[:, 2] > D405_FRONT_NZ)
    for name, mask in (("front", front), ("body", ~front)):
        m.submesh([np.flatnonzero(mask)], append=True).export(D405_SRC.with_name(f"d405-{name}.stl"))
    print(f"D405: 前面 {front.sum()} / 筐体 {(~front).sum()} 面")


if __name__ == "__main__":
    convert_nail()
    convert_thf()
    convert_adapter()
    convert_d405()
