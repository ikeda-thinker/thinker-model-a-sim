"""Run thinker-TMA-F1-REFX4-U3.urdf in Newton Physics.

    uv run tools/sim.py                 # OpenGL viewer
    uv run tools/sim.py --viewer viser  # browser viewer (http://localhost:8080)
    uv run tools/sim.py --viewer null --steps 500   # headless smoke test
"""
import argparse
import re
import tempfile
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton import JointTargetMode
from newton._src.solvers.mujoco.constants import SOLREF_MODE_FORCE_SPACE

NAIL_PUSH_MAX_MM = 3.9  # 最大押し込み量。z_joint の値そのまま 0〜3.9mm

# ネイルは目標 0 のバネ (受動) として扱い、変位が接触力を反映する触覚センサになるようにする。
# 剛性・減衰は仮の値 (実機の値が分かれば差し替える)
# x,y 関節 [N*m/rad]。爪先は回転軸から 15mm ほど中央側にずれているので、柔らかいと上から押したとき
# 爪が回って逃げ、押し込み (z) に力が乗らない。1.0 では逃げたので 10 にした
NAIL_ROT_KE = 10.0
NAIL_ROT_KD = 0.1
# z 関節 [N/m]。全押し込み 3.9mm で約 7.8N。
# 2000 N/m (= 2.0 N/mm) は think_hand_f_transient_sim/sim_main.py の SPRING_K_LEFT/RIGHT に合わせた
NAIL_Z_KE = 2000.0
# 把持時の目標押し込み量 [mm]。押し込みは正。同 sim_main.py の DELTA_REF の 0.5mm (約 1N)
NAIL_PUSH_REF_MM = 0.5
NAIL_Z_KD = 5.0

# 把持対象の箱。アーム初期姿勢 (全関節 0) での爪の間に置く。座標は実測 (FK + 爪の衝突メッシュ) から決めた
BOX_SIZE = 0.05  # 一辺 [m]
BOX_MASS = 0.1  # [kg]
BOX_POS = (0.4607, 0.405, 0.0665)  # 爪の x 中点、爪先の y、グリッパの z
# 接触の剛性・減衰 [N/m], [N*s/m]。Newton の force-space 指定 (質量に依存しない)。
# MuJoCo 既定の接触は剛さが加速度の単位で決まり、軽い爪 (10-30g) では数百 N/m 相当と柔らかく、
# 爪のバネ (2000 N/m) より柔らかいため変位の大半がめり込みに回っていた
CONTACT_KE = 5e4
CONTACT_KD = 200.0
HANDE_STROKE = 0.025  # Hande の片指ストローク [m]。関節値 0 が閉、0.025 が全開

ROOT = Path(__file__).resolve().parent.parent
PKGS = ROOT / "packages"


def resolve_meshes(urdf_path: Path) -> str:
    """Rewrite package:// URIs to absolute paths (falling back to placeholder .stl)."""

    def sub(m):
        p = PKGS / m.group(1)
        if not p.exists():
            p = p.with_suffix(".stl")
        return f'filename="{p.as_posix()}"'

    return re.sub(r'filename="package://([^"]+)"', sub, urdf_path.read_text(encoding="utf-8"))


def build_robot(urdf: str) -> newton.ModelBuilder:
    robot = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
    # newton の URDF パーサは絶対パスのみ確実に解決できるので、一時ファイルに書き出して渡す
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False, encoding="utf-8") as f:
        f.write(resolve_meshes(ROOT / urdf))
    robot.add_urdf(
        f.name,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        floating=False,  # UR3e のベースは固定
        enable_self_collisions=False,
        joint_damping_mode="passive",
    )
    Path(f.name).unlink(missing_ok=True)
    return robot


class Sim:
    # アームの初期姿勢・目標 [deg] (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3)。None なら全関節 0
    ARM_Q_DEG = None
    CAMERA_TARGET = BOX_POS

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.substeps = 20  # 接触を硬くしたので細かく (MuJoCo は時定数を 2*dt 未満にできない)
        self.sim_dt = self.frame_dt / self.substeps
        self.sim_time = 0.0

        builder = build_robot(args.urdf)
        # 全関節を位置制御 (PD) にする。ゲイン値は仮 (ネイルは下でバネとして上書き)
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 500.0
            builder.joint_target_kd[i] = 50.0
            builder.joint_target_mode[i] = int(JointTargetMode.POSITION)
        for j, label in enumerate(builder.joint_label):
            if builder.joint_type[j] != int(newton.JointType.FIXED) and (
                "thinker_arm_" in label or "hande_left_finger_joint" in label or "hande_right_finger_joint" in label
            ):
                d = builder.joint_qd_start[j]
                # アームは剛に固定、Hande は力で閉じる
                arm = "thinker_arm_" in label
                builder.joint_target_ke[d] = 20000.0 if arm else 2000.0
                builder.joint_target_kd[d] = 2000.0 if arm else 100.0
            m = re.search(r"f[12]_thinker_TF1_([xyz])_joint$", label)
            if m:
                d = builder.joint_qd_start[j]
                rot = m.group(1) != "z"
                builder.joint_target_ke[d] = NAIL_ROT_KE if rot else NAIL_Z_KE
                builder.joint_target_kd[d] = NAIL_ROT_KD if rot else NAIL_Z_KD
        self.arm_q_index = [
            builder.joint_q_start[j]
            for j, label in enumerate(builder.joint_label)
            if "thinker_arm_" in label and builder.joint_type[j] != int(newton.JointType.FIXED)
        ]
        if self.ARM_Q_DEG is not None:
            for k, deg in zip(self.arm_q_index, self.ARM_Q_DEG):
                builder.joint_q[k] = np.radians(deg)
        builder.add_ground_plane()
        self.add_scene(builder)

        self.model = builder.finalize()
        # ネイル関節 (f1/f2 の x,y,z) の q / dof インデックス。全て 1 自由度なので同じ並び
        labels = list(self.model.joint_label)
        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        self.nail_q_index, self.nail_dof_index = {}, {}
        for f in ("f1", "f2"):
            for a in ("x", "y", "z"):
                j = next(i for i, l in enumerate(labels) if l.endswith(f"{f}_thinker_TF1_{a}_joint"))
                self.nail_q_index[f"{f}_{a}"] = int(q_start[j])
                self.nail_dof_index[f"{f}_{a}"] = int(qd_start[j])

        gj = next(i for i, l in enumerate(labels) if l.endswith("robotiq_hande_left_finger_joint"))
        self.gripper_q, self.gripper_dof = int(q_start[gj]), int(qd_start[gj])
        self.box_body = next((i for i, l in enumerate(self.model.body_label) if l.endswith("box")), None)

        self.model.mujoco.solref_mode.fill_(SOLREF_MODE_FORCE_SPACE)
        self.model.shape_material_ke.fill_(CONTACT_KE)
        self.model.shape_material_kd.fill_(CONTACT_KD)
        # force-space の接触剛性は Newton の接触 (model.collide) を使うときだけ効く
        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=False, njmax=512, nconmax=256)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        # アームは初期姿勢を目標にして保持する (q と qd の並びはアームでは同じ)
        tgt = self.control.joint_target_q.numpy()
        q = self.model.joint_q.numpy()
        for k in self.arm_q_index:
            tgt[k] = q[k]
        self.control.joint_target_q.assign(tgt)
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.q0 = self.model.joint_q.numpy().copy()
        print(f"bodies={self.model.body_count} joints={self.model.joint_count} dofs={self.model.joint_dof_count}")

        self.viewer.set_model(self.model)
        self.look_at_nails()

    def add_scene(self, builder):
        """ロボット以外の物体を置く。サブクラスで差し替える。"""
        if self.args.box:
            h = BOX_SIZE / 2
            # 箱の下に静止した台を置く (持ち上げは手順の後)
            builder.add_shape_box(
                -1, xform=wp.transform(wp.vec3(BOX_POS[0], BOX_POS[1], BOX_POS[2] - h - 0.01), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.01
            )
            box = builder.add_body(xform=wp.transform(wp.vec3(*BOX_POS), wp.quat_identity()), label="box")
            cfg = newton.ModelBuilder.ShapeConfig(density=BOX_MASS / BOX_SIZE**3, mu=1.0)
            builder.add_shape_box(box, hx=h, hy=h, hz=h, cfg=cfg)

    def look_at_nails(self, dist=0.12, elevation_deg=25.0, azimuth_deg=60.0):
        """カメラを爪先 (CAMERA_TARGET) に向ける。azimuth は +y から x 方向へ振る角度。"""
        target = np.array(self.CAMERA_TARGET)
        az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
        offset = dist * np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
        pos = target + offset
        front = target - pos
        front /= np.linalg.norm(front)
        if hasattr(self.viewer, "set_camera"):  # ViewerGL
            self.viewer.set_camera(
                wp.vec3(*pos), pitch=float(np.degrees(np.arcsin(front[2]))), yaw=float(np.degrees(np.arctan2(front[1], front[0])))
            )
        elif hasattr(self.viewer, "_server"):  # ViewerViser

            def place(client):
                client.camera.position = tuple(pos)
                client.camera.look_at = tuple(target)

            self.viewer._server.on_client_connect(place)
            for client in self.viewer._server.get_clients().values():
                place(client)

    def set_gripper(self, opening_mm):
        """Hande の片指の開き [mm] (0=閉、25=全開)。右指は URDF の mimic で追従する。"""
        tgt = self.control.joint_target_q.numpy()
        tgt[self.gripper_dof] = np.clip(opening_mm * 1e-3, 0.0, HANDE_STROKE)
        self.control.joint_target_q.assign(tgt)

    def gripper_opening_mm(self):
        return float(self.state_0.joint_q.numpy()[self.gripper_q] * 1e3)

    def box_pos(self):
        return self.state_0.body_q.numpy()[self.box_body][:3]

    def nail_state(self):
        """各ネイルの {関節名: 値} を返す。x/y は回転 [deg] (±7)、z は押し込み量 [mm] (0=デフォルト、3.9=最大押し込み。関節値そのまま)。"""
        q = self.state_0.joint_q.numpy()
        out = {}
        for key, idx in self.nail_q_index.items():
            out[key] = float(np.degrees(q[idx])) if key.endswith(("_x", "_y")) else float(q[idx] * 1e3)
        return out

    def set_nail_target(self, finger, x_deg=0.0, y_deg=0.0, push_mm=0.0):
        """finger は "f1" か "f2"。バネの平衡位置をリミットでクランプして変える (センサ用途では呼ばない)。"""
        vals = {
            "x": np.radians(np.clip(x_deg, -7.0, 7.0)),
            "y": np.radians(np.clip(y_deg, -7.0, 7.0)),
            "z": np.clip(push_mm, 0.0, NAIL_PUSH_MAX_MM) * 1e-3,
        }
        tgt = self.control.joint_target_q.numpy()
        for axis, v in vals.items():
            tgt[self.nail_dof_index[f"{finger}_{axis}"]] = v
        self.control.joint_target_q.assign(tgt)

    def step(self):
        # 目標は control の既定値 (全関節 0) のまま。ここで self.control の目標を書き換えて制御する
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        q = self.state_0.joint_q.numpy()
        if not np.isfinite(q).all():
            raise ValueError("joint_q has non-finite values")
        print("nail state:", self.nail_state())


if __name__ == "__main__":
    import newton.examples

    parser = newton.examples.create_parser()
    parser.add_argument("--urdf", default="thinker-TMA-F1-REFX4-U3.urdf")
    parser.add_argument("--box", action=argparse.BooleanOptionalAction, default=True, help="把持対象の箱を置く")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Sim(viewer, args), args)
