"""ばら積み版の把持環境。ビンに小物 (ねじ・ピン・ナット・ワッシャ・球・ブロック・カプセル) を落として山を作り、
山の上の方から 1 個を対象にする。対象の位置・向きはノイズ付きの推定値として方策に渡す。

    uv run tools/grasp_pile.py --viewer null --episodes 5   # スクリプト方策で把持 (形状別の成功率を表示)
    uv run tools/grasp_pile.py --viewer viser               # ブラウザで見る (http://localhost:8080)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import warp as wp

import newton
import newton.examples
import grasp_env as ge
import sim

N_OBJ = 30
BIN_CENTER = ge.OBJ_XY  # ビンの中心 (基準姿勢の爪先の真下)
BIN_HALF = 0.07  # 内側の半寸法 [m]
WALL_T = 0.004  # 壁の半厚 [m]
WALL_H = 0.04  # 壁の高さ [m]
DROP_GRID = 4  # 落とす位置の格子 (DROP_GRID x DROP_GRID)
DROP_PITCH = 0.03  # 格子の間隔 [m]
DROP_Z0 = 0.03  # 1 段目の高さ [m]
DROP_DZ = 0.02  # 段の間隔 [m]
SETTLE_FRAMES = 120  # 落として静止するまで待つフレーム数 (2 s)
TARGET_RADIUS = 0.025  # ビン中心からこの距離以内の物体を対象の候補にする [m]
DENSITY = 2500.0  # [kg/m^3]
MU = 0.8

# 形状の定義。parts は (種類, 寸法, ローカル位置), extents は向きから上面の高さを見積もるための半寸法 (x, y, z),
# axis は長軸 (ローカル)。None なら向きに意味がない
SHAPES = {
    "screw": dict(parts=[("cyl", (0.0025, 0.010), (0, 0, 0)), ("cyl", (0.0045, 0.0015), (0, 0, 0.0115))], extents=(0.0045, 0.0045, 0.013), axis=(0, 0, 1)),
    "pin": dict(parts=[("cyl", (0.002, 0.0125), (0, 0, 0))], extents=(0.002, 0.002, 0.0125), axis=(0, 0, 1)),
    "nut": dict(parts=[("cyl", (0.0055, 0.0025), (0, 0, 0))], extents=(0.0055, 0.0055, 0.0025), axis=None),
    "washer": dict(parts=[("cyl", (0.0065, 0.001), (0, 0, 0))], extents=(0.0065, 0.0065, 0.001), axis=None),
    "ball": dict(parts=[("sph", (0.005,), (0, 0, 0))], extents=(0.005, 0.005, 0.005), axis=None),
    "block": dict(parts=[("box", (0.005, 0.004, 0.008), (0, 0, 0))], extents=(0.005, 0.004, 0.008), axis=(0, 0, 1)),
    "capsule": dict(parts=[("cap", (0.003, 0.008), (0, 0, 0))], extents=(0.003, 0.003, 0.011), axis=(0, 0, 1)),
}
SHAPE_NAMES = list(SHAPES)


def quat_to_mat(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def random_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


class PileEnv(ge.GraspEnv):
    NJMAX, NCONMAX = 16384, 4096  # 山が静止するまでに接触が約 1400 点出る
    CAMERA_TARGET = (*BIN_CENTER, 0.02)
    yaw_period = np.pi  # 長軸は 180 度対称

    def add_scene(self, builder):
        cx, cy = BIN_CENTER
        o = BIN_HALF + WALL_T
        for sx, sy, hx, hy in ((1, 0, WALL_T, o), (-1, 0, WALL_T, o), (0, 1, o, WALL_T), (0, -1, o, WALL_T)):
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(cx + sx * o, cy + sy * o, WALL_H / 2), wp.quat_identity()),
                hx=hx, hy=hy, hz=WALL_H / 2,
                cfg=newton.ModelBuilder.ShapeConfig(mu=MU),
            )
        cfg = newton.ModelBuilder.ShapeConfig(density=DENSITY, mu=MU)
        self.obj_names = []
        for i in range(N_OBJ):
            name = SHAPE_NAMES[i % len(SHAPE_NAMES)]
            self.obj_names.append(name)
            body = builder.add_body(xform=wp.transform(wp.vec3(cx, cy, 0.5 + 0.01 * i), wp.quat_identity()), label=f"obj{i}")
            for kind, dim, pos in SHAPES[name]["parts"]:
                xf = wp.transform(wp.vec3(*pos), wp.quat_identity())
                if kind == "cyl":
                    builder.add_shape_cylinder(body, xform=xf, radius=dim[0], half_height=dim[1], cfg=cfg)
                elif kind == "cap":
                    builder.add_shape_capsule(body, xform=xf, radius=dim[0], half_height=dim[1], cfg=cfg)
                elif kind == "sph":
                    builder.add_shape_sphere(body, xform=xf, radius=dim[0], cfg=cfg)
                else:
                    builder.add_shape_box(body, xform=xf, hx=dim[0], hy=dim[1], hz=dim[2], cfg=cfg)

    def init_objects(self):
        labels = list(self.model.body_label)
        self.obj_bodies = [next(i for i, l in enumerate(labels) if l.endswith(f"obj{k}")) for k in range(N_OBJ)]
        child = self.model.joint_child.numpy()
        q_start = self.model.joint_q_start.numpy()
        self.obj_q_slots = [int(q_start[int(np.nonzero(child == b)[0][0])]) for b in self.obj_bodies]
        self.target_type = None

    def spawn_objects(self):
        rng = self.rng
        q = self.q0.copy()
        # 格子の位置に少しずらして、ランダムな向きで落とす
        slots = [(ix, iy, layer) for layer in range(N_OBJ) for ix in range(DROP_GRID) for iy in range(DROP_GRID)]
        order = rng.permutation(N_OBJ)
        for k, slot in zip(order, range(N_OBJ)):
            layer, rem = divmod(slot, DROP_GRID * DROP_GRID)
            ix, iy = divmod(rem, DROP_GRID)
            x = BIN_CENTER[0] + (ix - (DROP_GRID - 1) / 2) * DROP_PITCH + rng.uniform(-0.004, 0.004)
            y = BIN_CENTER[1] + (iy - (DROP_GRID - 1) / 2) * DROP_PITCH + rng.uniform(-0.004, 0.004)
            s = self.obj_q_slots[k]
            q[s : s + 3] = [x, y, DROP_Z0 + layer * DROP_DZ]
            q[s + 3 : s + 7] = random_quat(rng)
        self.assign_q(q)
        for _ in range(SETTLE_FRAMES):
            sim.Sim.step(self)

        # 山の上の方から 1 個を選ぶ (ビン中心に近いものの中で一番高いもの)
        bq = self.state_0.body_q.numpy()
        pos = bq[self.obj_bodies][:, :3]
        near = np.hypot(pos[:, 0] - BIN_CENTER[0], pos[:, 1] - BIN_CENTER[1]) < TARGET_RADIUS
        inside = pos[:, 2] < WALL_H + 0.05
        cand = np.nonzero(near & inside)[0]
        if len(cand) == 0:
            cand = np.argsort(np.hypot(pos[:, 0] - BIN_CENTER[0], pos[:, 1] - BIN_CENTER[1]))[:1]
        k = int(cand[np.argmax(pos[cand, 2])])
        self.target = k
        self.target_type = self.obj_names[k]
        self.box_body = self.obj_bodies[k]
        spec = SHAPES[self.target_type]
        rot = quat_to_mat(bq[self.box_body][3:7])

        self.obj_xy = pos[k, :2].copy()
        self.est_pos = pos[k] + rng.normal(0.0, ge.EST_POS_STD, 3)
        self.obj_z0 = float(pos[k, 2])
        # 向きから上面までの高さを見積もる (OBB の鉛直方向の半寸法)
        self.obj_half_h = float(np.abs(rot[2]) @ np.array(spec["extents"]))
        # 長軸の水平方向の角度。爪の並びを長軸に合わせると、長軸に直交する向きに閉じる
        self.yaw_valid = 0.0
        phi = 0.0
        if spec["axis"] is not None:
            axis = rot @ np.array(spec["axis"], dtype=float)
            if np.hypot(axis[0], axis[1]) > 0.4:  # ほぼ立っているものは向きに意味がない
                phi = float(np.arctan2(axis[1], axis[0]))
                self.yaw_valid = 1.0
        self.obj_yaw = phi
        self.est_yaw = phi + rng.normal(0.0, ge.EST_YAW_STD) if self.yaw_valid else 0.0
        self.grasp_yaw = ge.wrap_period(self.est_yaw, np.pi) if self.yaw_valid else 0.0


class PileDemo(ge.Demo):
    def __init__(self, viewer, args):
        super().__init__(viewer, args, env_cls=PileEnv)
        self.types = []

    def start_episode(self):
        super().start_episode()
        e = self.env
        print(f"    対象={e.target_type} 高さ={e.est_pos[2] * 1e3:.1f}mm 長軸yaw={'%+.0f' % np.degrees(e.obj_yaw) if e.yaw_valid else '-'}")

    def step(self):
        n = self.episode
        super().step()
        if self.episode > n:
            self.types.append(self.prev_type)
        self.prev_type = self.env.target_type

    def test_final(self):
        super().test_final()
        for name in SHAPE_NAMES:
            r = [ok for t, ok in zip(self.types, self.results) if t == name]
            if r:
                print(f"  {name:8s} {sum(r)}/{len(r)}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--urdf", default="thinker-TMA-F1-REFX4-U3.urdf")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pos-noise-mm", type=float, default=ge.EST_POS_STD * 1e3, help="推定位置のノイズ (標準偏差) [mm]")
    parser.add_argument("--yaw-noise-deg", type=float, default=float(np.degrees(ge.EST_YAW_STD)), help="推定 yaw のノイズ (標準偏差) [deg]")
    parser.set_defaults(num_frames=10**9)
    viewer, args = newton.examples.init(parser)
    ge.EST_POS_STD = args.pos_noise_mm * 1e-3
    ge.EST_YAW_STD = np.radians(args.yaw_noise_deg)
    demo = PileDemo(viewer, args)
    demo.prev_type = demo.env.target_type
    while viewer.is_running() and demo.episode < args.episodes:
        demo.step()
        demo.render()
    demo.test_final()
    viewer.close()
