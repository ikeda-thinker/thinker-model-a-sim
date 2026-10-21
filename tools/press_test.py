"""ハンドを下向きにして、下から台を持ち上げて爪先に押し当てる。押し込み量 (z) と傾き (x, y) を表示する。

TMA の主用途 (上からバラ積みの小物にアプローチ) を想定した確認。アームを IK で下ろす代わりに、
台を z 方向の直動関節に載せて持ち上げる (爪から見た相対運動は同じ)。

    uv run tools/press_test.py --viewer null              # 平らな台
    uv run tools/press_test.py --viewer null --tilt-deg 3 # 台を傾けて片当たりさせる
    uv run tools/press_test.py --viewer viser             # ブラウザで見る (http://localhost:8080)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import warp as wp

import newton
import newton.examples
import sim

# UR の定番の下向き姿勢。グリッパ先端が -z、2 本の爪は y 方向に並ぶ (FK で確認)
ARM_Q_DEG = (0.0, -90.0, 90.0, -90.0, -90.0, 0.0)
# 上の姿勢・Hande 全閉での爪先 (URDF の tip_link) の位置 [m]。FK + 爪の衝突メッシュから測った
NAIL_TIP = (0.2986, 0.135, 0.0933)
GRIPPER_OPEN_MM = 10.0  # 爪先どうしが離れるよう少し開く (全閉だと爪先がほぼ接する)

GAP_MM = 2.0  # 開始時の台と爪先のすき間
RISE_MM = 4.0  # 台を持ち上げる量 (すき間を引いた残りが押し込みになる)
RISE_SPEED = 5.0  # [mm/s]
SETTLE_S = 0.5  # 持ち上げ前に Hande を開いて落ち着かせる時間
TABLE_HALF = (0.03, 0.03, 0.005)  # 台の半寸法 [m]


class PressTest(sim.Sim):
    ARM_Q_DEG = ARM_Q_DEG
    CAMERA_TARGET = NAIL_TIP

    def add_scene(self, builder):
        # 台: 世界に z 方向の直動関節でつなぎ、位置目標で持ち上げる。上面が爪先の GAP_MM 下から始まる
        tilt = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.radians(self.args.tilt_deg)))
        top0 = NAIL_TIP[2] - GAP_MM * 1e-3
        anchor = wp.transform(wp.vec3(NAIL_TIP[0], NAIL_TIP[1], top0), wp.quat_identity())
        table = builder.add_link(xform=anchor, label="table")
        # 台の上面がアンカー (傾き軸) を通るように箱をずらす
        builder.add_shape_box(
            table,
            xform=wp.transform(wp.quat_rotate(tilt, wp.vec3(0.0, 0.0, -TABLE_HALF[2])), tilt),
            hx=TABLE_HALF[0],
            hy=TABLE_HALF[1],
            hz=TABLE_HALF[2],
            cfg=newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.8),
        )
        j = builder.add_joint_prismatic(
            -1,
            table,
            parent_xform=anchor,
            axis=(0.0, 0.0, 1.0),
            target_pos=0.0,
            target_ke=1e5,
            target_kd=1e3,
            actuator_mode=newton.JointTargetMode.POSITION,
            label="table_lift",
        )
        builder.add_articulation([j], label="table")

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        labels = list(self.model.joint_label)
        j = next(i for i, l in enumerate(labels) if l.endswith("table_lift"))
        self.table_q = int(self.model.joint_q_start.numpy()[j])
        self.table_dof = int(self.model.joint_qd_start.numpy()[j])
        self.set_gripper(GRIPPER_OPEN_MM)
        self.frame = 0

    def step(self):
        t = self.frame * self.frame_dt
        rise = np.clip((t - SETTLE_S) * RISE_SPEED, 0.0, RISE_MM) * 1e-3
        tgt = self.control.joint_target_q.numpy()
        tgt[self.table_dof] = rise
        self.control.joint_target_q.assign(tgt)
        super().step()
        self.frame += 1
        if self.frame % 6 == 0:
            n = self.nail_state()
            lift = self.state_0.joint_q.numpy()[self.table_q] * 1e3
            print(
                f"t={t:4.2f}s 台={lift:5.2f}mm "
                f"pushL={n['left_z']:+.2f} pushR={n['right_z']:+.2f}mm "
                f"xL={n['left_x']:+.2f} yL={n['left_y']:+.2f} xR={n['right_x']:+.2f} yR={n['right_y']:+.2f}deg"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--urdf", default="thinker-TMA-F1-REFX4-U3.urdf")
    parser.add_argument("--tilt-deg", type=float, default=0.0, help="台を x 軸まわりに傾ける (爪の並び方向に傾斜)")
    parser.set_defaults(num_frames=int((SETTLE_S + RISE_MM / RISE_SPEED + 0.5) * 60))
    viewer, args = newton.examples.init(parser)
    newton.examples.run(PressTest(viewer, args), args)
