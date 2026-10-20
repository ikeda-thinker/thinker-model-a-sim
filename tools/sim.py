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

NAIL_PUSH_MAX_MM = 3.9  # 最大押し込み量。z_joint は 0 (デフォルト) 〜 -3.9mm

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
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.substeps = 10
        self.sim_dt = self.frame_dt / self.substeps
        self.sim_time = 0.0

        builder = build_robot(args.urdf)
        # 全関節を位置制御 (PD) にする。ゲイン値は仮
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 500.0
            builder.joint_target_kd[i] = 50.0
            builder.joint_target_mode[i] = int(JointTargetMode.POSITION)
        builder.add_ground_plane()

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

        self.solver = newton.solvers.SolverMuJoCo(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.q0 = self.model.joint_q.numpy().copy()
        print(f"bodies={self.model.body_count} joints={self.model.joint_count} dofs={self.model.joint_dof_count}")

        self.viewer.set_model(self.model)

    def nail_state(self):
        """各ネイルの {関節名: 値} を返す。x/y は回転 [deg] (±7)、z は押し込み量 [mm] (0=デフォルト、-3.9=最大押し込み)。"""
        q = self.state_0.joint_q.numpy()
        out = {}
        for key, idx in self.nail_q_index.items():
            out[key] = float(np.degrees(q[idx])) if key.endswith(("_x", "_y")) else float(q[idx] * 1e3)
        return out

    def set_nail_target(self, finger, x_deg=0.0, y_deg=0.0, push_mm=0.0):
        """finger は "f1" か "f2"。リミットでクランプして位置目標にする。"""
        vals = {
            "x": np.radians(np.clip(x_deg, -7.0, 7.0)),
            "y": np.radians(np.clip(y_deg, -7.0, 7.0)),
            "z": np.clip(push_mm, -NAIL_PUSH_MAX_MM, 0.0) * 1e-3,
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
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Sim(viewer, args), args)
