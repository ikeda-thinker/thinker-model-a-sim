"""Fusion 360 から書き出した STL (mm) を URDF 用のメッシュ (m) に直す。

    uv run tools/convert_meshes.py

- 爪 (packages/pile-nail-v1.stl): y が長さ方向 (根元の面 y=0 〜 先端 45mm)、z が内側 (相手の指) へのずれ、x が幅。
  根元に THF の板 (3x7mm) を差す穴 (8.4x5.4mm、深さ 8mm) がある。
  nail_link の座標 (z が長さ方向で根元の面が原点、y が内側、x が幅) に回して書き出す。
  曲がった形の凸包だと内側が埋まるので、当たり判定は長さ方向に切った区間ごとの凸包にする。
- THF (packages/think-hand-f.stl): 1 指ぶんのユニット。z が指の軸 (上端の板に爪を差す)、x が内側。
  座標はそのまま (mm → m だけ)。URDF の base_link の visual / collision の origin で位置を合わせる。
  見た目は外から見える部品だけ残して軽くし、当たり判定は本体の外形の凸包にする。
"""
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "packages/thinker_ros2_description"

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
    visible = [p for p in parts if np.abs(hull.nearest.signed_distance(p.vertices[:: max(1, len(p.vertices) // 50)])).min() < THF_VISIBLE_MM]
    vis = trimesh.util.concatenate(visible)
    vis.apply_scale(1e-3)
    body = m.vertices[(m.vertices[:, 2] >= THF_BODY_Z_MM[0]) & (m.vertices[:, 2] <= THF_BODY_Z_MM[1])]
    col = coarse_hull(body * 1e-3, 2 * HULL_GRID)
    (THF_OUT / "visual").mkdir(parents=True, exist_ok=True)
    (THF_OUT / "collision").mkdir(parents=True, exist_ok=True)
    vis.export(THF_OUT / "visual" / f"{THF_NAME}.stl")
    col.export(THF_OUT / "collision" / f"{THF_NAME}.stl")
    print(f"THF visual: {len(parts)} 部品中 {len(visible)} 部品, {len(vis.faces)} 面 / collision: {len(col.faces)} 面")


if __name__ == "__main__":
    convert_nail()
    convert_thf()
